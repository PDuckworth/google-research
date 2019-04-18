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

import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

import model as model
import utils as utils
from bayesquad.batch_selection import select_batch, KRIGING_BELIEVER
from bayesquad.gp_prior_means import NegativeQuadratic
from bayesquad.gps import VanillaGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel
from tensorflow.python.training import summary_io

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_enum("estimator", "iwae", [
    "iwae", "rws", "stl", "dreg", "dreg-cv", "rws-dreg", "rws-dreg-norm",
    "dreg-norm", "jk", "jk-dreg", "dreg-alpha"
], "Estimator type to use.")
flags.DEFINE_integer("num_samples", 1, "The numer of K samples to use.")
flags.DEFINE_integer("latent_dim", 1, "The dimension of the VAE latent space.")
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate for ADAM.")
flags.DEFINE_integer("batch_size", 1, "The batch size.")

tf.enable_eager_execution()

FLAGS = flags.FLAGS


class ToyConditionalNormalLikelihood(object):
  def __init__(self, size =1, name="toy_mean"):
    self.size = size
    self.name = name

  def __call__(self, *args, **kwargs):
    return tfd.Normal(loc=args, scale=tf.eye(self.size))

class ToyPrior(object):
  def __init__(self, mu_inital_value = 2., size = 1, name="toy_prior"):
    self.size = size
    self.name = name
    self.mu_inital_value = mu_inital_value
    self.mu = tf.Variable(name="mu", initial_value= self.mu_inital_value)

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution"""
    return tfd.Normal(loc=self.mu, scale=tf.eye(self.size)), self.mu


DEFAULT_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer(),
    "b": tf.zeros_initializer()
}

class ToyConditionalNormal(object):

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

    # # In EAGER Mode, tensor_list is now an ndarray :(
    if isinstance(tensor_list, np.ndarray):
        tensor_list = [t.item() for t in tensor_list]
        input_dim = self.size
        raw_input_shape = tf.shape(tensor_list)
        fcnet_input_shape = [tf.reduce_prod(raw_input_shape), input_dim]
        fcnet_inputs = tf.reshape(tensor_list, fcnet_input_shape)
        print("fcnet_inputs shape:", fcnet_inputs.shape)
        out = self.fcnet(fcnet_inputs)
    else:
        print("CONDITION ON AN NDARRAY")
        # helpful commands if tensor_list is again a tensor
        # tensor_list = [t[0].item() for t in tensor_list]
        # # Reshape outputs to the original shape.
        # output_size = tf.concat([raw_input_shape, [1]], axis=0)
        # out = tf.reshape(outs, output_size)

    # print("1. ", len(tensor_list))
    # print("3. ", input_dim)
    # print("4. ", raw_input_shape)
    # print("5. ", fcnet_input_shape)
    # print("6. ", fcnet_inputs.shape)
    # print("input ", fcnet_inputs)
    # print("7. ", out)

    return out

  def initialise_and_fix_network(self, some_data):

      mu = self.condition(some_data)
      # Set the network parameters to "close" to their optimal
      noiseA = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim, 1)).astype(np.float32)
      noiseb = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim,)).astype(np.float32)

      A = np.ones(FLAGS.latent_dim) / 2. + noiseA
      b = np.eye(FLAGS.latent_dim) * (2 / 2.) + noiseb
      tf_A = tf.Variable(A.astype(np.float32).reshape(FLAGS.latent_dim, FLAGS.latent_dim))
      tf_b = tf.Variable(b.astype(np.float32).reshape(FLAGS.latent_dim, ))

      self.fcnet.get_variables()[0].assign(tf_A)
      self.fcnet.get_variables()[1].assign(tf_b)

      print(self.fcnet.get_variables())

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""

    mu = self.condition(args[0])
    # mu = tf.Variable(name="mu", initial_value=) # This breaks the gradients :S

    if kwargs.get("stop_gradient", False):
        print("STOP GRADIENTS")
        mu = tf.stop_gradient(mu)
    return tfd.Normal(loc=mu, scale=tf.eye(self.size)*(2/3.)), self.fcnet.get_variables()


def toy_example():
    with tf.GradientTape() as tape:

        train_xs, valid_xs, test_xs = utils.load_toy_data()
        batch_xs = train_xs[0:FLAGS.batch_size]
        print("batch shape: ", batch_xs.shape)

        # set up your prior model, proposal and likelihood networks
        p_z = ToyPrior(mu_inital_value = 2., size=FLAGS.latent_dim, name="toy_prior")

        # returns a callable Normal distribution
        p_x_given_z = ToyConditionalNormalLikelihood()

        # with tf.name_scope('proposal') as scope:
        q_z = ToyConditionalNormal(
            size=FLAGS.latent_dim,
            hidden_layer_sizes=1,
            initializers=None,
            use_bias=True,
            name="proposal")

        # initialise the network parameters to near optimal
        q_z.initialise_and_fix_network(batch_xs)

        # returns the Normal dist proposal, and the parameters (fixed to optimal A and b)
        proposal, inference_network_params = q_z(batch_xs, stop_gradient=False)
        z = proposal.sample(sample_shape=[FLAGS.num_samples])
        # [num_samples, batch_size, latent_dim]
        print("samples ", z.shape)

        # returns a Normal dist conditioned on z
        likelihood = p_x_given_z(z)

        # returns the Prior normal (p_z), and the parameter mu
        prior, mu = p_z()

        # merge the parameters to compute gradients wrt
        parameters = (inference_network_params[0], inference_network_params[1], mu)

        log_p_z = tf.reduce_sum(prior.log_prob(z), axis=-1)
        log_q_z = tf.reduce_sum(proposal.log_prob(z), axis=-1)
        log_p_x_given_z = tf.reduce_sum(likelihood.log_prob(batch_xs), axis=-1)
        log_weights = log_p_z + log_p_x_given_z - log_q_z
        log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)  # this converts back to IWAE estimator (log of the sum)
        log_avg_weight = log_sum_weight - tf.log(tf.to_float(FLAGS.num_samples))

        # Build the evidence lower bound (ELBO) or the negative loss
        # kl = tf.reduce_mean(tfd.kl_divergence(proposal, prior), axis=-1)  # analytic KL
        # log_sum_ll = tf.reduce_logsumexp(log_p_x_given_z, axis=0)  # this converts back to IWAE estimator (log of the sum)
        # expected_log_likelihood = log_sum_ll - tf.log(tf.to_float(FLAGS.num_samples))
        # KL_elbo = tf.reduce_mean(expected_log_likelihood - kl)




        def get_log_joint(z):
            return np.reshape(p_x_given_z(z).log_prob(batch_xs).numpy() + prior.log_prob(z).numpy(), (-1, 1))

        kernel = GPy.kern.RBF(1, variance=2, lengthscale=2)
        kernel.variance.constrain_bounded(1e-5, 1e5)
        bq_likelihood = GPy.likelihoods.Gaussian(variance=1e-1)

        bq_prior = Gaussian(mean=proposal._loc.numpy().squeeze(), covariance=proposal._scale.numpy().item())

        initial_x = bq_prior.sample(5)
        initial_y = []
        for point in initial_x:
            initial_y.append(get_log_joint(np.atleast_2d(point)))
        initial_y = np.concatenate(initial_y)
        mean_function = NegativeQuadratic(1)
        gpy_gp = GPy.core.GP(initial_x, initial_y, kernel=kernel, likelihood=bq_likelihood, mean_function=mean_function)
        warped_gp = VanillaGP(gpy_gp)
        bq_model = IntegrandModel(warped_gp, bq_prior)

        for i in range(10):
            if i % 5 == 0:
                gpy_gp.optimize_restarts(num_restarts=5)
            failed = True
            while failed:
                try:
                    batch = select_batch(bq_model, 1, KRIGING_BELIEVER)
                    failed = False
                except FloatingPointError:
                    gpy_gp.optimize_restarts(num_restarts=5)

            X = np.array(batch)
            Y = get_log_joint(X)

            bq_model.update(batch, Y)

        gpy_gp.optimize_restarts(num_restarts=5)

        bq_elbo = bq_model.integral_mean()

        import scipy.integrate
        def integrand(z):
            return get_log_joint(z) * np.exp(bq_prior.logpdf(np.atleast_2d(z)))
        brute_force_elbo = scipy.integrate.quad(integrand, -10, 10)

        # Set the network parameters to "close" to their optimal
        # noiseA = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim, 1)).astype(np.float32)
        # noiseb = np.random.normal(loc=0, scale=0.01, size=(FLAGS.latent_dim,)).astype(np.float32)

        # inference_network_params = proposal.fcnet.get_variables()

        # opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # inference_grads = opt.compute_gradients(inference_loss, var_list=inference_network_params)
        # print("Grads, = ", inference_grads)

        inference_loss = -tf.reduce_mean(log_avg_weight)

        # log_p_hat_mean = tf.reduce_mean(log_avg_weight)
        print("inference_loss ", inference_loss)

        print("BQ ", bq_elbo)
        print("ELBO ", brute_force_elbo)

        grads = tape.gradient(inference_loss, parameters)
        vectorized_grads = [np.array(g).reshape(1, ) for g in grads]
        print("GRADIENTS = ", vectorized_grads)

        # first_moments = np.average(vectorized_grads)
        # print("first_moments  = ", first_moments )
        # second_moments = np.average(tf.square(vectorized_grads))
        # print("second_moments  = ", second_moments)
        #
        # variances = second_moments - tf.square(first_moments)
        # inference_grad_snr_sq = tf.reduce_mean(tf.square(first_moments)) / tf.reduce_mean(variances)
        # snr = tf.reduce_mean(tf.square(first_moments)) / tf.reduce_mean(variances)
        #
        # print("SNR = ", snr)


if __name__ == "__main__":

    ## FORCE TO USE THE CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    toy_example()
