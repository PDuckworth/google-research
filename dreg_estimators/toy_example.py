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

"""Training/eval loop for DReGs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np

import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

import model as model
import utils as utils
from tensorflow.python.training import summary_io

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_enum("estimator", "iwae", [
    "iwae", "rws", "stl", "dreg", "dreg-cv", "rws-dreg", "rws-dreg-norm",
    "dreg-norm", "jk", "jk-dreg", "dreg-alpha"
], "Estimator type to use.")
flags.DEFINE_integer("num_samples", 64, "The numer of K samples to use.")
flags.DEFINE_integer("latent_dim", 1, "The dimension of the VAE latent space.")
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate for ADAM.")

tf.enable_eager_execution()

FLAGS = flags.FLAGS


class ToyMean(object):
  def __init__(self, size =1, name="toy_mean"):
    self.size = size
    self.name = name
  def __call__(self, *args, **kwargs):
    """Creates a normal distribution (conditioned?) on the inputs."""
    return tfd.Normal(loc=args, scale=tf.eye(self.size))

class ToyPrior(object):
  def __init__(self, mu = 1, size = 1, name="toy_prior"):
    self.size = size
    self.name = name
    self.mu = mu
    self.mu = tf.ones(self.size)*self.mu
    self.fcnet = tfd.Normal(loc = self.mu, scale = tf.eye(self.size))
  def __call__(self, *args, **kwargs):
    """Creates a normal distribution"""
    return tfd.Normal(loc=self.mu, scale=tf.eye(self.size))


DEFAULT_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer(),
    "b": tf.zeros_initializer()
}

class ToyNormalproposal(object):

  def __init__(self,
               size,
               hidden_layer_sizes,
               initializers=None,
               use_bias=True,
               name="toy_normal"):

    self.size = size
    self.name = name
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS

    self.fcnet = snt.Linear(output_size=hidden_layer_sizes,
                            use_bias=use_bias,
                            initializers=initializers,
                            name=name + "_fcnet")


  def condition(self, tensor_list):
    """Computes the parameters of a normal distribution based on the inputs."""
    # # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    concatted_inputs = tf.concat(tensor_list, axis=-1)
    input_dim = concatted_inputs.get_shape().as_list()[-1]
    raw_input_shape = tf.shape(concatted_inputs)[:-1]
    fcnet_input_shape = [tf.reduce_prod(raw_input_shape), input_dim]
    fcnet_inputs = tf.reshape(concatted_inputs, fcnet_input_shape)

    print("fcnet_inputs shape:", fcnet_inputs.shape)
    outs = self.fcnet(fcnet_inputs)

    # Reshape outputs to the original shape.
    output_size = tf.concat([raw_input_shape, [1]], axis=0)
    out = tf.reshape(outs, output_size)
    return out

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    # mu, sigma = self.condition(args)
    mu = self.condition(args)
    self.mu = mu

    A = np.ones(FLAGS.latent_dim) / 2.
    b = np.eye(FLAGS.latent_dim) * (3 / 2.)
    tf_A = tf.Variable(A.astype(np.float32).reshape(FLAGS.latent_dim, FLAGS.latent_dim))
    tf_b = tf.Variable(b.astype(np.float32).reshape(FLAGS.latent_dim, ))
    self.fcnet.get_variables()[0].assign(tf_A)
    self.fcnet.get_variables()[1].assign(tf_b)

    if kwargs.get("stop_gradient", False):
      mu = tf.stop_gradient(mu)
      # sigma = tf.stop_gradient(sigma)
    # return tfd.Normal(loc=mu, scale=sigma)
    return tfd.Normal(loc=mu, scale=tf.eye(self.size)*(2/3.)), self.fcnet.get_variables()


def toy_example():
    with tf.GradientTape() as tape:

        train_xs, valid_xs, test_xs = utils.load_toy_data()

        # set up your prior dist, proposal and likelihood networks
        p_z = ToyPrior(mu = 3, size = 1, name="toy_prior")

        p_x_given_z = ToyMean(name="likelihood")

        # with tf.name_scope('proposal') as scope:
        q_z = ToyNormalproposal(
            size=FLAGS.latent_dim,
            hidden_layer_sizes=1,
            initializers=None,
            use_bias=True,
            name="proposal")

        # returns the Normal dist proposal, and also the params (fixed to optimal A and b)
        proposal, inference_network_params = q_z(train_xs, stop_gradient=False)
        print("params: ", inference_network_params)
        z = proposal.sample(sample_shape=[FLAGS.num_samples])

        likelihood = p_x_given_z(z)
        prior = p_z()
        log_p_z = tf.reduce_sum(prior.log_prob(z), axis=-1)
        log_q_z = tf.reduce_sum(proposal.log_prob(z), axis=-1)
        log_p_x_given_z = tf.reduce_sum(likelihood.log_prob(train_xs), axis=-1)
        log_weights = log_p_z + log_p_x_given_z - log_q_z
        log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)  # this converts back to IWAE estimator (log of the sum)
        log_avg_weight = log_sum_weight - tf.log(tf.to_float(FLAGS.num_samples))

        # Build the evidence lower bound (ELBO) or the negative loss
        kl = tf.reduce_mean(tfd.kl_divergence(proposal, prior), axis=-1)  # analytic KL
        log_sum_ll = tf.reduce_logsumexp(log_p_x_given_z, axis=0)  # this converts back to IWAE estimator (log of the sum)
        expected_log_likelihood = log_sum_ll - tf.log(tf.to_float(FLAGS.num_samples))
        KL_elbo = tf.reduce_mean(expected_log_likelihood - kl)

        model_loss = -tf.reduce_mean(log_avg_weight)
        inference_loss = -tf.reduce_mean(log_avg_weight)
        log_p_hat_mean = tf.reduce_mean(log_avg_weight)

        # Set the network parameters to "close" to their optimal
        # noiseA = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim, 1)).astype(np.float32)
        # noiseb = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim,)).astype(np.float32)


        # inference_network_params = proposal.fcnet.get_variables()

        # opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # inference_grads = opt.compute_gradients(inference_loss, var_list=inference_network_params)
        # print("Grads, = ", inference_grads)

        inference_grads = tape.gradient(inference_loss, inference_network_params)
        vectorized_grads = tf.concat([tf.reshape(g, [-1]) for g in inference_grads if g is not None], axis=0)

        first_moments = np.average(vectorized_grads)
        second_moments = np.average(tf.square(vectorized_grads))
        variances = second_moments - tf.square(first_moments)
        inference_grad_snr_sq = tf.reduce_mean(tf.square(first_moments)) / tf.reduce_mean(variances)

        print("grads = ", vectorized_grads)
        print("1st moments  = ", first_moments)
        print("2nd moments  = ", second_moments)
        print("var ", variances)
        print("SNR ", inference_grad_snr_sq)
    # with tf.Session() as sess:
    #     grads_ = sess.run(inference_grads)
    #     print(grads_)



if __name__ == "__main__":

    ## FORCE TO USE THE CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    toy_example()