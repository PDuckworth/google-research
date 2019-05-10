# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for DReGs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io
import tensorflow as tf

from numpy import newaxis
import tensorflow_probability as tfp
import model

tfd = tfp.distributions

flags = tf.flags

flags.DEFINE_string("MNIST_LOCATION", "/home/paul/Datasets/google-research/mnist",
                    "The directory of MNIST datasets.")
flags.DEFINE_string("OMNIGLOT_LOCATION", "/home/paul/Datasets/google-research/omniglot.mat",
                    "The directory of Omniglot datasets.")

FLAGS = flags.FLAGS

MNIST_LOCATION = lambda: FLAGS.MNIST_LOCATION
OMNIGLOT_LOCATION = lambda: FLAGS.OMNIGLOT_LOCATION



def binarize_batch_xs(batch_xs, or_not = False):
  """Randomly binarize a batch of data."""
  if or_not: return batch_xs  # just don't do anything :)
  return (batch_xs > np.random.random(size=batch_xs.shape)).astype(batch_xs.dtype)


def summarize_grads(grads):
  """Summarize the gradient vector."""

  grad_ema = tf.train.ExponentialMovingAverage(decay=0.99)
  vectorized_grads = tf.concat([tf.reshape(g, [-1]) for g, _ in grads if g is not None], axis=0)
  new_second_moments = tf.square(vectorized_grads)
  new_first_moments = vectorized_grads
  maintain_grad_ema_op = grad_ema.apply([new_first_moments, new_second_moments])
  first_moments = grad_ema.average(new_first_moments)
  second_moments = grad_ema.average(new_second_moments)
  variances = second_moments - tf.square(first_moments)
  return (maintain_grad_ema_op, tf.reduce_mean(variances),
          tf.reduce_mean(tf.square(first_moments)) / tf.reduce_mean(variances))


def load_omniglot(dynamic_binarization=True, shuffle=True, shuffle_seed=123):
  """Load Omniglot dataset.

  Args:
    dynamic_binarization: Return the data as floats, or return the data
      binarized with a fixed seed.
    shuffle: Shuffle the train set before extracting the last examples for the
      validation set.
    shuffle_seed: Seed for the shuffling.

  Returns:
    Tuple of (train, valid, test).
  """
  n_validation = 1345  # Default magic number

  def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order="fortran")

  # Try to load data locally
  if tf.gfile.Exists(os.path.join("/tmp", "omniglot.mat")):
    omni_raw = scipy.io.loadmat(os.path.join("/tmp", "omniglot.mat"))
  else:
    # Fall back to external
    with tf.gfile.GFile(OMNIGLOT_LOCATION(), "rb") as f:
      omni_raw = scipy.io.loadmat(f)

  train_data = reshape_data(omni_raw["data"].T.astype("float32"))
  test_data = reshape_data(omni_raw["testdata"].T.astype("float32"))

  if not dynamic_binarization:
    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

  if shuffle:
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(
        train_data.shape[0])
    train_data = train_data[permutation]

  train_data, valid_data = (train_data[:-n_validation],
                            train_data[-n_validation:])

  return train_data, valid_data, test_data


def load_mnist():
  """Load the MNIST training set."""

  def load_dataset(dataset="train_xs"):
    if os.path.exists("/tmp/%s.npy" % dataset):
      with tf.gfile.Open("/tmp/%s.npy" % dataset, "rb") as f:
        xs = np.load(f).reshape(-1, 784)
    else:
      with tf.gfile.Open(
          os.path.join(MNIST_LOCATION(), "%s.npy" % dataset), "rb") as f:
        xs = np.load(f).reshape(-1, 784)

    return xs.astype(np.float32)

  train_xs = load_dataset("train_xs")
  test_xs = load_dataset("test_xs")
  valid_xs = load_dataset("valid_xs")

  return train_xs, valid_xs, test_xs

def load_toy_data(datapoints = 200, dim=1):
  SEED = 11

  TRUE_MEAN = 2
  TRUE_SCALE = 1
  NUM_DATA_POINTS = 1024*2

  # Create fake data points using a tf distribution
  # true_distribution = tfp.distributions.Normal(loc=TRUE_MEAN, scale=TRUE_SCALE)
  # fake_some_samples = true_distribution.sample(NUM_DATA_POINTS, seed=SEED)
  # fake_data = tf.reshape(fake_some_samples, (NUM_DATA_POINTS,1))
  # data = tf.placeholder_with_default(fake_data, shape=[NUM_DATA_POINTS,1], name='fake_data')

  # Create fake data points using a np distribution
  np.random.seed(SEED)
  data = np.random.normal(loc=TRUE_MEAN, scale=TRUE_SCALE, size=(NUM_DATA_POINTS, 4)).astype(np.float32)

  # Add a new axis so that tf can evaluate the log probability at each data point for many parameter values.
  train_xs = data[:1000]
  test_xs = data[:1000]
  valid_xs = data[:512]

  return train_xs, valid_xs, test_xs

def main(unused_argv):
  with tf.Session() as sess:

    # MNIST dataset
    train_xs, valid_xs, test_xs = load_mnist()
    # print(train_xs.shape)
    # print(train_xs[0])

    # Toy dataset
    train_xs, valid_xs, test_xs = load_toy_data()

if __name__ == "__main__":
  tf.app.run(main)


