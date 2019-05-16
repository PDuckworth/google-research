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

import GPy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import model as model
import utils as utils
from bayesquad.batch_selection import select_batch, KRIGING_BELIEVER
from bayesquad.gp_prior_means import NegativeQuadratic
from bayesquad.gps import VanillaGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel
from tensorflow.python.training import summary_io
import warnings

warnings.filterwarnings("ignore")

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_enum("estimator", "iwae", [
    "iwae", "rws", "stl", "dreg", "dreg-cv", "rws-dreg", "rws-dreg-norm",
    "dreg-norm", "jk", "jk-dreg", "dreg-alpha", "bq"
], "Estimator type to use.")
flags.DEFINE_float("alpha", 0.9, "Weighting for DReG(alpha)")
flags.DEFINE_integer("batch_size", 256, "The batch size.")
flags.DEFINE_integer("num_samples", 64, "The numer of K samples to use.")
flags.DEFINE_integer("latent_dim", 1, "The dimension of the VAE latent space.")
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate for ADAM.")
flags.DEFINE_integer("max_steps", int(1e5), "The number of steps to train for.")
flags.DEFINE_integer("summarize_every", 10,
                     "Number of steps between summaries.")
flags.DEFINE_string("logdir", "/home/paul/Software/DREG-data/Toy/",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_string("subfolder", "",
                    "Folder name to put summaries and checkpoints in (under logdir).")
flags.DEFINE_bool("bias_check", False,
                  "Run a bias check instead of training the model.")
flags.DEFINE_string(
    "initial_checkpoint_dir", None,
    ("Initial checkpoint directory to start from. This also disables model ",
     "training. Only the inference network is trained.")
)
flags.DEFINE_integer(
    "run", 0,
    ("A number to distinguish which run this is. This allows us to run ",
     "multiple trials with the same params.")
)
flags.DEFINE_string(
    "var_calc", None,
    ("Comma separated list of estimators to calculate the variance of on this",
     "trajectory")
)
flags.DEFINE_enum("dataset", "toy", [
    "mnist",
    "struct_mnist",
    "omniglot",
    "toy"
], "Dataset to use.")
flags.DEFINE_bool("image_summary", True, "Create visualizations")

FLAGS = flags.FLAGS


def create_logging_hook(metrics):

  def summary_formatter(d):
    return ", ".join(
        ["%s: %g" % (key, float(value)) for key, value in sorted(d.items())])

  logging_hook = tf.train.LoggingTensorHook(
      metrics, formatter=summary_formatter, every_n_iter=FLAGS.summarize_every)
  return logging_hook

def main(unused_argv):
  # proposal_hidden_dims = [20]
  # likelihood_hidden_dims = [0]
  with tf.Graph().as_default():
    if FLAGS.dataset in ["mnist", "struct_mnist"]:
      train_xs, valid_xs, test_xs = utils.load_mnist()
    elif FLAGS.dataset == "omniglot":
      train_xs, valid_xs, test_xs = utils.load_omniglot()
    elif FLAGS.dataset == "toy":
      train_xs, valid_xs, test_xs = utils.load_toy_data()

    print("dataset = ", train_xs.shape)

    # Placeholder for input mnist digits.
    # observations_ph = tf.placeholder("float32", [None, 2])
    observations_ph = tf.placeholder("float32", [None, FLAGS.latent_dim])  # This model requires latent_dim == input data shape

    # set up your prior dist, proposal and likelihood networks
    (prior, likelihood, proposal) = model.get_toy_models(train_xs, which_example="toy1D")

    # Compute the lower bound and the loss
    estimators = model.iwae(
        prior,
        likelihood,
        proposal,
        observations_ph,
        FLAGS.num_samples, [], # [alpha, beta, gamma, delta],
        contexts=None,
        debug=False)

    # actual_proposal = proposal(observations_ph)

    print("VARS: ", proposal.fcnet.get_variables())
    log_p_hat, neg_model_loss, neg_inference_loss = estimators[FLAGS.estimator]

    model_loss = -tf.reduce_mean(neg_model_loss)
    log_p_hat_mean = tf.reduce_mean(log_p_hat)

    if FLAGS.estimator == "bq":
        _, _, inference_loss = estimators['bq']
        tf.summary.scalar("ELBOs/bq_train", inference_loss)
        tf.summary.scalar("ELBOs/iwae", tf.reduce_mean(estimators["iwae"][0]))
    else:
        inference_loss = -tf.reduce_mean(neg_inference_loss)

    model_params = prior.get_parameter_mu()
    inference_params = proposal.fcnet.get_variables()

    # Compute and apply the gradients, summarizing the gradient variance.
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    cv_grads = []

    model_grads = opt.compute_gradients(model_loss, var_list=(model_params))
    # inference model (encoder) params are just A and b. (Ax+b)
    inference_grads = opt.compute_gradients(inference_loss, var_list=inference_params)

    grads = model_grads + inference_grads #+ cv_grads

    model_ema_op, model_grad_variance, _ = (utils.summarize_grads(model_grads))
    inference_ema_op, inference_grad_variance, inference_grad_snr_sq = (utils.summarize_grads(inference_grads))

    ema_ops = [model_ema_op, inference_ema_op]

    # this ensures you evaluate ema_ops before the apply_gradient function :)
    with tf.control_dependencies(ema_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)


    tf.summary.scalar("params/b", tf.reshape(inference_params[1][0], ()))
    tf.summary.scalar("params/A", tf.reshape(inference_params[0][0][0], ()))
    tf.summary.scalar("params/mu", model_params[0])
    #
    # tf.summary.scalar("gradients/A", tf.reshape(inference_grads[0][0], ()))
    # tf.summary.scalar("gradients/b", tf.reshape(inference_grads[1][0], ()))
    # tf.summary.scalar("gradients/mu", model_grads[0][0])

    tf.summary.scalar("grad_variance/phi", inference_grad_variance)
    tf.summary.scalar("grad_variance/model", model_grad_variance)

    tf.summary.scalar("log_p_hat/train", log_p_hat_mean)
    tf.summary.scalar("log_p_hat/ana_KL", estimators["elbo"])

    exp_name = "%s.lr-%g.n_samples-%d.batch_size-%d.alpha-%g.dataset-%s.run-%d" % (
        FLAGS.estimator, FLAGS.learning_rate, FLAGS.num_samples, FLAGS.batch_size, FLAGS.alpha,
        FLAGS.dataset, FLAGS.run)
    checkpoint_dir = os.path.join(FLAGS.logdir, FLAGS.subfolder, exp_name)
    print("Checkpoints: : ", checkpoint_dir)

    if FLAGS.initial_checkpoint_dir and not tf.gfile.Exists(checkpoint_dir):
      tf.gfile.MakeDirs(checkpoint_dir)
      f = "checkpoint"
      tf.gfile.Copy(
          os.path.join(FLAGS.initial_checkpoint_dir, f),
          os.path.join(checkpoint_dir, f))

    with tf.train.MonitoredTrainingSession(
        is_chief=True,
        hooks=[
            create_logging_hook({
                "Step": global_step,
                "log_p_hat": log_p_hat_mean,
                # "model_grads": model_grad,
                # "model_grad_variance": model_grad_variance,
                "infer_grad_varaince": inference_grad_variance,
                "infer_grad_snr_sq": inference_grad_snr_sq,
            })
        ],
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=10,
        save_summaries_steps=FLAGS.summarize_every,
        # log_step_count_steps=FLAGS.summarize_every * 10
        log_step_count_steps=0,  # disable logging of steps/s to avoid TF warning in validation sets
        # config=tf.ConfigProto(log_device_placement=True) # spits out the location of each computation (CPU, GPU etc.)
        ) as sess:

      writer = summary_io.SummaryWriterCache.get(checkpoint_dir)
      t_stats = []
      cur_step = -1
      indices = list(range(train_xs.shape[0]))
      n_epoch = 0

      def run_eval(cur_step, split="valid", eval_batch_size=256):
        """Run evaluation on a datasplit."""
        if split == "valid":
          eval_dataset = valid_xs
        elif split == "test":
          eval_dataset = test_xs

        log_p_hat_vals = []
        for i in range(0, eval_dataset.shape[0], eval_batch_size):
          # batch_xs = utils.binarize_batch_xs(eval_dataset[i:(i + eval_batch_size)])
          batch_xs = eval_dataset[i:(i + eval_batch_size)]
          log_p_hat_vals.append(sess.run(log_p_hat_mean, feed_dict={observations_ph: batch_xs}))

        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="log_p_hat/%s" % split,
                simple_value=np.mean(log_p_hat_vals))
        ])
        writer.add_summary(summary, cur_step)
        print("curr_step: %g, log_p_hat/%s: %g" % (cur_step, split, np.mean(log_p_hat_vals)))

      while cur_step < FLAGS.max_steps and not sess.should_stop():
        n_epoch += 1
        print(">>>>", n_epoch)

        random.shuffle(indices)

        for cnt, i in enumerate(range(0, train_xs.shape[0], FLAGS.batch_size)):
          if sess.should_stop() or cur_step > FLAGS.max_steps:
            break

          print("epoch: ", n_epoch, " . batch no:", cnt)
          # Get a batch, then dynamically binarize
          ns = indices[i:i + FLAGS.batch_size]
          # batch_xs = utils.binarize_batch_xs(train_xs[ns])
          batch_xs = train_xs[ns]

          _, cur_step, grads_ = sess.run([train_op, global_step, grads], feed_dict={observations_ph: batch_xs})
          # grads_ = sess.run([train_op, global_step, model_params, grads], feed_dict={observations_ph: batch_xs})

        if n_epoch % 10 == 0:
          print("epoch #", n_epoch)
          run_eval(cur_step, "test", FLAGS.batch_size)
          # run_eval(cur_step, "valid", FLAGS.batch_size)

          var_names = ["theta", "A    ", "b    "]
          # var_names = ["A    ", "b    "]
          for m, (i,j) in enumerate(grads_):
            print(var_names[m],": grad, val: ", i, j)

if __name__ == "__main__":
    ## FORCE TO USE THE CPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""

    tf.app.run(main)
