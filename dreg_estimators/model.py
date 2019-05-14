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

"""Basic IWAE setup with DReGs estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import GPy
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from bayesquad.batch_selection import select_batch, KRIGING_BELIEVER
from bayesquad.gp_prior_means import NegativeQuadratic
from bayesquad.gps import VanillaGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel


tfd = tfp.distributions
FLAGS = tf.flags.FLAGS

DEFAULT_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer(),
    "b": tf.zeros_initializer()
}


class ConditionalBernoulli(object):
  """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

  def __init__(self,
               size,
               hidden_layer_sizes,
               hidden_activation_fn=tf.nn.tanh,
               initializers=None,
               bias_init=0.0,
               name="conditional_bernoulli"):
    """Creates a conditional Bernoulli distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intiializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must be a
        dictionary mapping the keys 'w' and 'b' to the initializers for the
        weights and biases. Defaults to xavier for the weights and zeros for the
        biases when initializers is None.
      bias_init: A scalar or vector Tensor that is added to the output of the
        fully-connected network that parameterizes the mean of this
        distribution.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.bias_init = bias_init
    self.size = size
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the p parameter of the Bernoulli distribution."""
    # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    concatted_inputs = tf.concat(tensor_list, axis=-1)
    input_dim = concatted_inputs.get_shape().as_list()[-1]
    raw_input_shape = tf.shape(concatted_inputs)[:-1]
    fcnet_input_shape = [tf.reduce_prod(raw_input_shape), input_dim]
    fcnet_inputs = tf.reshape(concatted_inputs, fcnet_input_shape)
    outs = self.fcnet(fcnet_inputs) + self.bias_init
    # Reshape outputs to the original shape.
    output_size = tf.concat([raw_input_shape, [self.size]], axis=0)
    return tf.reshape(outs, output_size)

  def __call__(self, *args, **kwargs):
    p = self.condition(args)
    if kwargs.get("stop_gradient", False):
      p = tf.stop_gradient(p)
    return tfd.Bernoulli(logits=p)  # logits = log-odds of a 1 occurring


class ConditionalNormal(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self,
               size,
               hidden_layer_sizes,
               mean_center=None,
               sigma_min=0.0,
               raw_sigma_bias=0.25,
               hidden_activation_fn=tf.nn.tanh,
               initializers=None,
               name="conditional_normal"):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      mean_center: Optionally, mean center the data using this Tensor as the
        mean.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intitializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must be a
        dictionary mapping the keys 'w' and 'b' to the initializers for the
        weights and biases. Defaults to xavier for the weights and zeros for the
        biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    self.mean_center = mean_center
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [2 * size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the parameters of a normal distribution based on the inputs."""
    # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    inputs = tf.concat(tensor_list, axis=1)
    if self.mean_center is not None:
      inputs -= self.mean_center
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(
        tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args)

    # Optional stop_gradient argument stops the parameters of the distribution.
    # TODO(gjt): This only works for 1 latent layer networks.
    if kwargs.get("stop_gradient", False):
      mu = tf.stop_gradient(mu)
      sigma = tf.stop_gradient(sigma)
    return tfd.Normal(loc=mu, scale=sigma)


class ToyConditionalNormalLikelihood(object):
  def __init__(self, size =1, name="toy_likelihood"):
    """Here, the input data dimenion, must be the same as the requested latent_dimension, since there is no function, """
    self.size = size
    self.name = name

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution (conditioned?) on the inputs."""
    return tfd.Normal(loc=args[0], scale=tf.ones(self.size))


class ToyPrior(object):
  def __init__(self, mu_inital_value = 2., size = 1, name="toy_prior"):
    self.size = size
    self.name = name

    self.prior_loc = tf.Variable(name="mu", initial_value=  tf.ones([FLAGS.latent_dim], dtype=tf.float32)*mu_inital_value)
    self.prior_scale = tf.ones([FLAGS.latent_dim], dtype=tf.float32)

    # self.mu = tf.Variable(name="mu", initial_value= self.mu_inital_value)
    # print(self.mu)

  def get_parameter_mu(self):
    return self.prior_loc

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution"""
    return tfd.Normal(loc=self.prior_loc, scale=self.prior_scale)

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
    self.hidden_layer_sizes = hidden_layer_sizes
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

    # print("fcnet_inputs shape:", fcnet_inputs.shape, type(fcnet_inputs))
    out = self.fcnet(fcnet_inputs)

    # Reshape outputs to the original shape.
    print(">>>why is this wrong?", raw_input_shape)

    output_size = tf.concat([raw_input_shape, [1]], axis=0)
    # out = tf.reshape(out, output_size)
    return out
    # mu, sigma = tf.split(outs, 2, axis=1)
    # return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    # mu, sigma = self.condition(args)
    mu = self.condition(args)
    self.mu = mu

    if kwargs.get("stop_gradient", False):
      mu = tf.stop_gradient(mu)
      # sigma = tf.stop_gradient(sigma)
    # return tfd.Normal(loc=mu, scale=sigma)
    print("<<< MU SHAPE<", mu.shape)

    # a = np.ones([self.hidden_layer_sizes, self.hidden_layer_sizes], dtype=np.float32) * 2 / 3
    # block_scale = np.kron(np.eye(FLAGS.batch_size, dtype=np.float32), a)

    scale = tf.ones_like(mu) * 2/3. #(FLAGS.batch_size, dtype=np.float32)*2/3
    return tfd.Normal(loc=mu, scale=scale)


class Toy20DNormal(object):

  def __init__(self,
               size,
               hidden_layer_sizes,
               mean_center=None,
               sigma_min=0.0,
               raw_sigma_bias=0.25,
               hidden_activation_fn=None,
               initializers=None,
               name="toy_normal"):

    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    self.mean_center = mean_center
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [2 * size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the parameters of a normal distribution based on the inputs."""
    # Remove None's from tensor_list
    tensor_list = [t for t in tensor_list if t is not None]
    inputs = tf.concat(tensor_list, axis=1)
    if self.mean_center is not None:
      inputs -= self.mean_center
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(
        tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args)

    # Optional stop_gradient argument stops the parameters of the distribution.
    # TODO(gjt): This only works for 1 latent layer networks.
    if kwargs.get("stop_gradient", False):
      mu = tf.stop_gradient(mu)
      sigma = tf.stop_gradient(sigma)
    return tfd.Normal(loc=mu, scale=sigma)


def get_toy_models(train_xs, which_example="toy1D"):
  input_dim = train_xs.shape[1]

  if FLAGS.dataset == "toy":

    z_prior = ToyPrior(mu_inital_value = 2., size = FLAGS.latent_dim, name="toy_prior")

    likelihood = ToyConditionalNormalLikelihood(size = input_dim)

    # with tf.name_scope('proposal') as scope:
    proposal = ToyConditionalNormal(
      size=FLAGS.latent_dim,
      hidden_layer_sizes= FLAGS.latent_dim,
      use_bias=True,
      name="proposal")

  return z_prior, likelihood, proposal


def iwae(p_z,           # prior
         p_x_given_z,   # likelihood
         q_z,           # proposal
         observations,
         num_samples,
         cvs = [],
         contexts=None,
         antithetic=False,
         debug = False):
  """Computes a gradient of the IWAE estimator.

  Args:
    p_z: The prior. Should be a callable that optionally accepts a conditioning
      context and returns a tfp.distributions.Distribution which has the
      log_prob and sample methods implemented. The distribution should be over a
      [batch_size, latent_dim] space.
    p_x_given_z: The likelihood. Should be a callable that accepts as input a
      tensor of shape [num_samples, batch_size, latent_size + context_size] and
      returns a tfd.Distribution over a [num_samples, batch_size, data_dim]
      space.
    q_z: The proposal, should be a callable which accepts a batch of
      observations of shape [batch_size, data_dim] and returns a distribution
      over [batch_size, latent_dim].
    observations: A float Tensor of shape [batch_size, data_dim] containing the
      observations.
    num_samples: The number of samples for the IWAE estimator.
    cvs: Control variate variables.
    contexts: A float Tensor of shape [batch_size, context_dim] containing the
      contexts. (Optionally, none)
    antithetic: Whether to use antithetic sampling.

  Returns:
    estimators: Dictionary of tuples (objective, neg_model_loss,
      neg_inference_network_loss).
  """
  # alpha, beta, gamma, delta = cvs
  batch_size = tf.shape(observations)[0]
  proposal = q_z(observations, contexts, stop_gradient=False)

  # [num_samples, batch_size, latent_size]
  z = proposal.sample(sample_shape=[FLAGS.num_samples])

  likelihood = p_x_given_z(z)
  prior = p_z(contexts)
  log_p_z = tf.reduce_sum(prior.log_prob(z), axis=-1)  # [num_samples, batch_size]

  # Before reduce_sum is [num_samples, batch_size, latent_dim].
  # Sum over the latent dim.

  # log_q_z = tf.reduce_sum(tf.diag_part(tf.squeeze(proposal.log_prob(z))), axis=-1)  # [num_samples, batch_size]
  log_q_z = tf.reduce_sum(proposal.log_prob(z), axis=-1)  # [num_samples, batch_size]

  # Before reduce_sum is [num_samples, batch_size, latent_dim].
  log_prob_of_obs = likelihood.log_prob(observations) # [num_samples, batch_size, input_dim]
  log_p_x_given_z = tf.reduce_sum(log_prob_of_obs, axis=-1)  # [num_samples, batch_size]

  log_weights = log_p_z + log_p_x_given_z - log_q_z    # [num_samples, batch_size]
  if debug:
      print_op = tf.print("Z>>>", z.shape, z)
      print_op0 = tf.print("LOG WEIGHTS>>>", log_weights.shape, log_weights)
      print_op1 = tf.print("LOG P(Z)>>>", log_p_z.shape, log_p_z)
      print_op2 = tf.print("LOG P(X|Z)>>>", log_p_x_given_z.shape, log_p_x_given_z)
      print_op3 = tf.print("LOG Q(Z)>>>", log_q_z.shape, log_q_z)

      with tf.control_dependencies([print_op, print_op0, print_op1, print_op2, print_op3]):
        log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)   # sum over samples before log converts back to IWAE estimator (log of the sum)
  else:
    log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)  # sum over samples before log converts back to IWAE estimator (log of the sum)

  log_avg_weight = log_sum_weight - tf.log(tf.to_float(num_samples))
  # print("shapes", log_p_z.shape, log_p_x_given_z.shape, log_q_z.shape, log_weights.shape, log_sum_weight.shape)

  normalized_weights = tf.stop_gradient(tf.nn.softmax(log_weights, axis=0))

  # Compute gradient estimators
  model_loss = log_avg_weight
  estimators = {}

  if FLAGS.estimator == "bq":

      def get_log_joint(z, observations):
          sess = tf.Session()

          with sess.as_default():

              float_z = tf.transpose(tf.cast(z, tf.float32))
              if debug: print("z", float_z)

              likelihood = p_x_given_z(float_z)  # condition the likelihood on the z
              prior = ToyPrior(mu_inital_value = 2., size = FLAGS.latent_dim, name="toy_prior")
              # init = tf.initialize_all_variables()
              init = tf.global_variables_initializer()
              sess.run(init)

              prior = prior()
              # log_prob = likelihood.log_prob(observations.reshape(1,2)).eval()
              if debug: print("like", likelihood._loc, likelihood._scale)

              log_prob = likelihood.log_prob(observations)

              if debug: print("logprob", log_prob)
              log_prior = prior.log_prob(float_z)

              if debug: print("prior", log_prior)
              return log_prob.eval() + log_prior.eval()

      kernel = GPy.kern.RBF(FLAGS.latent_dim, variance=2, lengthscale=2)
      bq_likelihood = GPy.likelihoods.Gaussian(variance=1e-5)

      def get_bq_estimate(loc, scale, observations):

          # print("proposal / bq prior>>", loc, scale)
          bq_prior = Gaussian(mean=loc.squeeze(), covariance=scale.squeeze()**2)
          initial_x = bq_prior.sample(1)  # 1 sample from each proposal distribution in the batch

          # initial_y = []
          # for point in initial_x:
          #     initial_y.append(get_log_joint(np.atleast_2d(point), observations))
          # initial_y = np.concatenate(initial_y)

          # print("obs", observations)

          initial_y = get_log_joint(np.atleast_2d(initial_x), observations)
          # print("initial_y", initial_y.shape, initial_y)

          mean_function = NegativeQuadratic(FLAGS.latent_dim)
          initial_x = np.expand_dims(initial_x, 1)

          # print("x", type(initial_x), initial_x.shape)
          # print("y", type(initial_y), initial_y.shape)
          gpy_gp = GPy.core.GP(initial_x, initial_y, kernel=kernel, likelihood=bq_likelihood, mean_function=mean_function)
          warped_gp = VanillaGP(gpy_gp)
          # bq_model = IntegrandModel(warped_gp, bq_prior)

          # for i in range(10):
          #     if i % 5 == 0:
          #         gpy_gp.optimize_restarts(num_restarts=5)
          #     failed = True
          #     while failed:
          #         try:
          #             batch = select_batch(bq_model, 1, KRIGING_BELIEVER)
          #             failed = False
          #         except FloatingPointError:
          #             gpy_gp.optimize_restarts(num_restarts=5)
          #
          #     X = np.array(batch)
          #     Y = get_log_joint(X)
          #
          #     bq_model.update(batch, Y)

          gpy_gp.optimize_restarts(num_restarts=5, verbose=False)

          return gpy_gp.mean_function.mu, gpy_gp.mean_function.m_0, gpy_gp.mean_function.omega, gpy_gp.kern.lengthscale.values[0], gpy_gp.kern.variance.values[0], gpy_gp.X, warped_gp.underlying_gp.K_inv_Y


      mu, m_0, omega, kernel_lengthscale, kernel_variance, gp_X, K_inv_Y = tf.py_func(get_bq_estimate, [proposal._loc, proposal._scale, observations], [tf.float64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64])
      # mu, m_0, omega, kernel_lengthscale, kernel_variance, gp_X, K_inv_Y = get_bq_estimate(proposal._loc, proposal._scale)

      # latent_dim = 1

      # Need to set shapes explicitly because these variables come out of py_func, and TF can't infer their shapes
      omega.set_shape((FLAGS.latent_dim))
      mu.set_shape((FLAGS.latent_dim))
      m_0.set_shape(())

      dimensions = FLAGS.latent_dim  # which dimensions?? latent dimensions?

      # mu is latent_dim-dimensional, proposal._loc is (latent_dim x batch_size)-dimensional
      mu_diff = proposal._loc - tf.expand_dims(tf.cast(mu, tf.float32), 1)

      Lambda = tf.diag(omega ** -2)  # omega is latent_dim-dimensional
      Lambda = tf.cast(Lambda, tf.float32)  # latent_dim x latent_dim

      print("mu diff", mu_diff)  # batch_size x latent_dim
      print("scale", proposal._scale)  # (batch_size.latent_dim x batch_size.latent_dim) block-diagonal matrix...
      # should be batch_size x latent_dim x latent_dim
      # (batch_size blocks of size latent_dim x latent_dim)


      # hacked_proposal_scale = tf.reshape(tf.diag_part(proposal._scale), (-1, 1, 1))  # this does the right thing iff latent_dim is 1...
      # hacked_proposal_scale = tf.reshape(proposal._scale, (batch_size, FLAGS.latent_dim))  # this is [batch_size, latent_dim]
      # inner thing wants to be batch_size x latent_dim x latent_dim, result wants to be (batch_size,)
      # inner_qfe1 = tf.tensordot(hacked_proposal_scale**2, Lambda, [[2],[0]])

      quadratic_form_expectation1 = tf.reshape(tf.matmul(tf.reshape(tf.cast(omega, tf.float32), (1, FLAGS.latent_dim)), tf.transpose(proposal._scale)), (batch_size,) )
      # quadratic_form_expectation1 = tf.trace(inner_qfe1)  # shape: [batch_size,]

      quadratic_form_expectation2_1 = tf.tensordot(mu_diff, Lambda, [[1],[0]])  # (batch_size x latent_dim)
      quadratic_form_expectation2 = tf.einsum('ij,ij->i', quadratic_form_expectation2_1, mu_diff)  # (batch_size)

      quadratic_form_expectation = quadratic_form_expectation1 + quadratic_form_expectation2

      prior_mean_integral = (tf.cast(m_0, tf.float32) - 0.5 * quadratic_form_expectation)  # (batch_size)

      kernel_normalising_constant = (2 * np.pi * tf.cast(kernel_lengthscale, tf.float32)) ** dimensions / 2

      cov_matrix = tf.expand_dims(tf.cast(kernel_lengthscale, tf.float32) * tf.eye(dimensions), 0) + hacked_proposal_scale**2  # (batch_size x latent_dim x latent_dim)

      multivariate_normal = tfp.distributions.MultivariateNormalFullCovariance(loc=proposal._loc,
                                                                               covariance_matrix=cov_matrix)

      # gp_X is num_gp_points x latent_dim
      # TODO Make this its own function
      int_k_pi = tf.cast(kernel_variance, tf.float32) \
                 * kernel_normalising_constant \
                 * multivariate_normal.prob(tf.cast(tf.reshape(gp_X, (-1, 1, 1)), tf.float32))

      int_k_pi = tf.transpose(int_k_pi)
      #  ... we have batch_size many normals, and batch_size many points in latent_dim space...
      # for each normal, we want the integral considering all points...
      # int_k_pi should be batch_size x num_gp_points, and K_inv_Y should be (num_gp_points)

      integral_mean = tf.transpose(int_k_pi @ tf.cast(K_inv_Y, tf.float32))

      bq_elbo = integral_mean + prior_mean_integral + tf.diag_part(proposal.entropy())
      bq_elbo = tf.reduce_mean(bq_elbo)
      bq_loss = bq_elbo

      estimators['bq'] = (log_avg_weight, log_avg_weight, bq_loss)

  # Build the evidence lower bound (ELBO) or the negative loss
  kl = tf.reduce_mean(tfd.kl_divergence(proposal, prior), axis=-1)  # analytic KL
  log_sum_ll = tf.reduce_logsumexp(log_p_x_given_z, axis=0)  # this converts back to IWAE estimator (log of the sum)
  expected_log_likelihood = log_sum_ll - tf.log(tf.to_float(num_samples))
  elbo = tf.reduce_mean(expected_log_likelihood - kl)
  estimators["elbo"] = elbo

  # things we are interested in: (log_p_hat, neg_model_loss, neg_inference_loss)
  estimators["iwae"] = (log_avg_weight, log_avg_weight, log_avg_weight)

  stopped_z_log_q_z = tf.reduce_sum(proposal.log_prob(tf.stop_gradient(z)), axis=-1)
  estimators["rws"] = (log_avg_weight, model_loss,
                       tf.reduce_sum(
                           normalized_weights * stopped_z_log_q_z, axis=0))

  # # Doubly reparameterized
  stopped_proposal = q_z(observations, contexts, stop_gradient=True)
  stopped_log_q_z = tf.reduce_sum(stopped_proposal.log_prob(z), axis=-1)
  stopped_log_weights = log_p_z + log_p_x_given_z - stopped_log_q_z
  sq_normalized_weights = tf.square(normalized_weights)

  estimators["stl"] = (log_avg_weight, model_loss,
                       tf.reduce_sum(normalized_weights * stopped_log_weights, axis=0))
  estimators["dreg"] = (log_avg_weight, model_loss,
                        tf.reduce_sum(sq_normalized_weights * stopped_log_weights, axis=0))

  estimators["rws-dreg"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(
          (normalized_weights - sq_normalized_weights) * stopped_log_weights,
          axis=0))

  # Add normed versions
  normalized_sq_normalized_weights = (
      sq_normalized_weights / tf.reduce_sum(
          sq_normalized_weights, axis=0, keepdims=True))
  estimators["dreg-norm"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(
          normalized_sq_normalized_weights * stopped_log_weights, axis=0))

  rws_dregs_weights = normalized_weights - sq_normalized_weights
  normalized_rws_dregs_weights = rws_dregs_weights / tf.reduce_sum(
      rws_dregs_weights, axis=0, keepdims=True)
  estimators["rws-dreg-norm"] = (
      log_avg_weight, model_loss,
      tf.reduce_sum(normalized_rws_dregs_weights * stopped_log_weights, axis=0))

  estimators["dreg-alpha"] = (log_avg_weight, model_loss,
                              (1 - FLAGS.alpha) * estimators["dreg"][-1] +
                              FLAGS.alpha * estimators["rws-dreg"][-1])

  # Jackknife
  loo_log_weights = tf.tile(
      tf.expand_dims(tf.transpose(log_weights), -1), [1, 1, num_samples])
  loo_log_weights = tf.matrix_set_diag(
      loo_log_weights, -np.inf * tf.ones([batch_size, num_samples]))
  loo_log_avg_weight = tf.reduce_mean(
      tf.reduce_logsumexp(loo_log_weights, axis=1) - tf.log(
          tf.to_float(num_samples - 1)),
      axis=-1)
  jk_model_loss = num_samples * log_avg_weight - (
      num_samples - 1) * loo_log_avg_weight

  estimators["jk"] = (jk_model_loss, jk_model_loss, jk_model_loss)

  # Compute JK w/ DReG for the inference network
  loo_normalized_weights = tf.reduce_mean(
      tf.square(tf.stop_gradient(tf.nn.softmax(loo_log_weights, axis=1))),
      axis=-1)
  estimators["jk-dreg"] = (
      jk_model_loss, jk_model_loss, num_samples * tf.reduce_sum(
          sq_normalized_weights * stopped_log_weights, axis=0) -
      (num_samples - 1) * tf.reduce_sum(
          tf.transpose(loo_normalized_weights) * stopped_log_weights, axis=0))

  # # Compute control variates
  # loo_baseline = tf.expand_dims(tf.transpose(log_weights), -1)
  # loo_baseline = tf.tile(loo_baseline, [1, 1, num_samples])
  # loo_baseline = tf.matrix_set_diag(
  #     loo_baseline, -np.inf * tf.ones_like(tf.transpose(log_weights)))
  # loo_baseline = tf.reduce_logsumexp(loo_baseline, axis=1)
  # loo_baseline = tf.transpose(loo_baseline)
  #
  # learning_signal = tf.stop_gradient(tf.expand_dims(
  #     log_avg_weight, 0)) - (1 - gamma) * tf.stop_gradient(loo_baseline)
  # vimco = tf.reduce_sum(learning_signal * stopped_z_log_q_z, axis=0)
  #
  # first_part = alpha * vimco + (1 - alpha) * tf.reduce_sum(
  #     normalized_weights * stopped_log_weights, axis=0)
  # second_part = ((1 - beta) * (tf.reduce_sum(
  #     ((1 - delta) / tf.to_float(num_samples) - normalized_weights) *
  #     stopped_z_log_q_z,
  #     axis=0)) + beta * tf.reduce_sum(
  #         (sq_normalized_weights - normalized_weights) * stopped_log_weights,
  #         axis=0))
  # estimators["dreg-cv"] = (log_avg_weight, model_loss, first_part + second_part)

  return estimators
