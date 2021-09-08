# Copyright (c) 2015, J.M. Hernandez-Lobato


# License: copied from https://github.com/HIPS/Probabilistic-Backpropagation

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the project nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import math

import numpy as np
import theano
import theano.tensor as T
import pickle
import gzip



class Prior:

    def __init__(self, layer_sizes, var_targets):

        # We refine the factor for the prior variance on the weights

        n_samples = 3.0
        v_observed = 1.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed

        # We refine the factor for the prior variance on the noise

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        # We refine the gaussian prior on the weights

        self.rnd_m_w = []
        self.m_w_hat_nat = []
        self.v_w_hat_nat = []
        self.a_w_hat_nat = []
        self.b_w_hat_nat = []
        for size_out, size_in in zip(layer_sizes[ 1 : ], layer_sizes[ : -1 ]):
            self.rnd_m_w.append(1.0 / np.sqrt(size_in + 1) *
                np.random.randn(size_out, size_in + 1))
            self.m_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.v_w_hat_nat.append((self.a_w - 1) / self.b_w * \
                np.ones((size_out, size_in + 1)))
            self.a_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.b_w_hat_nat.append(np.zeros((size_out, size_in + 1)))

    def get_initial_params(self):

        m_w = []
        v_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.rnd_m_w[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a_sigma_hat_nat + 1,
            'b': -self.b_sigma_hat_nat }

    def get_params(self):

        m_w = []
        v_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.m_w_hat_nat[ i ] / self.v_w_hat_nat[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a_sigma_hat_nat + 1,
            'b': -self.b_sigma_hat_nat }

    def refine_prior(self, params):

        for i in range(len(params[ 'm_w' ])):
            for j in range(params[ 'm_w' ][ i ].shape[ 0 ]):
                for k in range(params[ 'm_w' ][ i ].shape[ 1 ]):

                    # We obtain the parameters of the cavity distribution

                    v_w_nat = 1.0 / params[ 'v_w' ][ i ][ j, k ]
                    m_w_nat = params[ 'm_w' ][ i ][ j, k ] / \
                        params[ 'v_w' ][ i ][ j, k ]
                    v_w_cav_nat = v_w_nat - self.v_w_hat_nat[ i ][ j, k ]
                    m_w_cav_nat = m_w_nat - self.m_w_hat_nat[ i ][ j, k ]
                    v_w_cav = 1.0 / v_w_cav_nat
                    m_w_cav = m_w_cav_nat / v_w_cav_nat
                    a_w_nat = self.a_w - 1
                    b_w_nat = -self.b_w
                    a_w_cav_nat = a_w_nat - self.a_w_hat_nat[ i ][ j, k ]
                    b_w_cav_nat = b_w_nat - self.b_w_hat_nat[ i ][ j, k ]
                    a_w_cav = a_w_cav_nat + 1
                    b_w_cav = -b_w_cav_nat

                    if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
                        v_w_cav < 1e6:

                        # We obtain the values of the new parameters of the
                        # posterior approximation

                        v = v_w_cav + b_w_cav / (a_w_cav - 1)
                        v1  = v_w_cav + b_w_cav / a_w_cav
                        v2  = v_w_cav + b_w_cav / (a_w_cav + 1)
                        logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
                        logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
                        logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
                        d_logZ_d_m_w_cav = -m_w_cav / v
                        d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav**2 / v**2
                        m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav
                        v_w_new = v_w_cav - v_w_cav**2 * \
                            (d_logZ_d_m_w_cav**2 - 2 * d_logZ_d_v_w_cav)
                        a_w_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
                            (a_w_cav + 1) / a_w_cav - 1.0)
                        b_w_new = 1.0 / (np.exp(logZ2 - logZ1) * \
                            (a_w_cav + 1) / (b_w_cav) - np.exp(logZ1 - \
                            logZ) * a_w_cav / b_w_cav)
                        v_w_new_nat = 1.0 / v_w_new
                        m_w_new_nat = m_w_new / v_w_new
                        a_w_new_nat = a_w_new - 1
                        b_w_new_nat = -b_w_new

                        # We update the parameters of the approximate factor,
                        # whih is given by the ratio of the new posterior
                        # approximation and the cavity distribution

                        self.m_w_hat_nat[ i ][ j, k ] = m_w_new_nat - \
                            m_w_cav_nat
                        self.v_w_hat_nat[ i ][ j, k ] = v_w_new_nat - \
                            v_w_cav_nat
                        self.a_w_hat_nat[ i ][ j, k ] = a_w_new_nat - \
                            a_w_cav_nat
                        self.b_w_hat_nat[ i ][ j, k ] = b_w_new_nat - \
                            b_w_cav_nat

                        # We update the posterior approximation

                        params[ 'm_w' ][ i ][ j, k ] = m_w_new
                        params[ 'v_w' ][ i ][ j, k ] = v_w_new

                        self.a_w = a_w_new
                        self.b_w = b_w_new

        return params

class Network_layer:

    def __init__(self, m_w_init, v_w_init, non_linear = True):

        # We create the theano variables for the means and variances

        self.m_w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='m_w', borrow = True)
        self.v_w = theano.shared(value = v_w_init.astype(theano.config.floatX),
            name='v_w', borrow = True)
        self.w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='w', borrow = True)

        # We store the type of activation function

        self.non_linear = non_linear

        # We store the number of inputs

        self.n_inputs = theano.shared(float(m_w_init.shape[ 1 ]))

    @staticmethod
    def n_pdf(x):

        return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

    @staticmethod
    def n_cdf(x):

        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    @staticmethod
    def gamma(x):

        return Network_layer.n_pdf(x) / Network_layer.n_cdf(-x)

    @staticmethod
    def beta(x):

        return Network_layer.gamma(x) * (Network_layer.gamma(x) - x)

    def output_probabilistic(self, m_w_previous, v_w_previous):

        # We add an additional deterministic input with mean 1 and variance 0

        m_w_previous_with_bias = \
            T.concatenate([ m_w_previous, T.alloc(1, 1) ], 0)
        v_w_previous_with_bias = \
            T.concatenate([ v_w_previous, T.alloc(0, 1) ], 0)

        # We compute the mean and variance after the linear operation

        m_linear = T.dot(self.m_w, m_w_previous_with_bias) / T.sqrt(self.n_inputs)
        v_linear = (T.dot(self.v_w, v_w_previous_with_bias) + \
            T.dot(self.m_w**2, v_w_previous_with_bias) + \
            T.dot(self.v_w, m_w_previous_with_bias**2)) / self.n_inputs

        if (self.non_linear):

            # We compute the mean and variance after the ReLU activation

            alpha = m_linear / T.sqrt(v_linear)
            gamma = Network_layer.gamma(-alpha)
            gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
            gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma, gamma_robust)

            v_aux = m_linear + T.sqrt(v_linear) * gamma_final

            m_a = Network_layer.n_cdf(alpha) * v_aux
            v_a = m_a * v_aux * Network_layer.n_cdf(-alpha) + \
                Network_layer.n_cdf(alpha) * v_linear * \
                (1 - gamma_final * (gamma_final + alpha))

            return (m_a, v_a)

        else:

            return (m_linear, v_linear)

    def output_deterministic(self, output_previous):

        # We add an additional input with value 1

        output_previous_with_bias = \
            T.concatenate([ output_previous, T.alloc(1, 1) ], 0) / \
            T.sqrt(self.n_inputs)

        # We compute the mean and variance after the linear operation

        a = T.dot(self.w, output_previous_with_bias)

        if (self.non_linear):

            # We compute the ReLU activation

            a = T.switch(T.lt(a, T.fill(a, 0)), T.fill(a, 0), a)

        return a

class Network:

    def __init__(self, m_w_init, v_w_init, a_init, b_init):

        # We create the different layers

        self.layers = []

        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[ : -1 ], v_w_init[ : -1 ]):
                self.layers.append(Network_layer(m_w, v_w, True))

        self.layers.append(Network_layer(m_w_init[ -1 ],
            v_w_init[ -1 ], False))

        # We create mean and variance parameters from all layers

        self.params_m_w = []
        self.params_v_w = []
        self.params_w = []
        for layer in self.layers:
            self.params_m_w.append(layer.m_w)
            self.params_v_w.append(layer.v_w)
            self.params_w.append(layer.w)

        # We create the theano variables for a and b

        self.a = theano.shared(float(a_init))
        self.b = theano.shared(float(b_init))

    def output_deterministic(self, x):

        # Recursively compute output

        for layer in self.layers:
            x = layer.output_deterministic(x)

        return x

    def output_probabilistic(self, m):

        v = T.zeros_like(m)

        # Recursively compute output

        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)

        return (m[ 0 ], v[ 0 ])

    def logZ_Z1_Z2(self, x, y):

        m, v = self.output_probabilistic(x)

        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)

        logZ = -0.5 * (T.log(v_final) + (y - m)**2 / v_final)
        logZ1 = -0.5 * (T.log(v_final1) + (y - m)**2 / v_final1)
        logZ2 = -0.5 * (T.log(v_final2) + (y - m)**2 / v_final2)

        return (logZ, logZ1, logZ2)

    def generate_updates(self, logZ, logZ1, logZ2):

        updates = []
        for i in range(len(self.params_m_w)):
            updates.append((self.params_m_w[ i ], self.params_m_w[ i ] + \
                self.params_v_w[ i ] * T.grad(logZ, self.params_m_w[ i ])))
            updates.append((self.params_v_w[ i ], self.params_v_w[ i ] - \
               self.params_v_w[ i ]**2 * \
                (T.grad(logZ, self.params_m_w[ i ])**2 - 2 * \
                T.grad(logZ, self.params_v_w[ i ]))))

        updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * \
            (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / \
            (self.b) - T.exp(logZ1 - logZ) * self.a / self.b)))

        return updates

    def get_params(self):

        m_w = []
        v_w = []
        for layer in self.layers:
            m_w.append(layer.m_w.get_value())
            v_w.append(layer.v_w.get_value())

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a.get_value(),
            'b': self.b.get_value() }

    def set_params(self, params):

        for i in range(len(self.layers)):
            self.layers[ i ].m_w.set_value(params[ 'm_w' ][ i ])
            self.layers[ i ].v_w.set_value(params[ 'v_w' ][ i ])

        self.a.set_value(params[ 'a' ])
        self.b.set_value(params[ 'b' ])

    def remove_invalid_updates(self, new_params, old_params):

        m_w_new = new_params[ 'm_w' ]
        v_w_new = new_params[ 'v_w' ]
        m_w_old = old_params[ 'm_w' ]
        v_w_old = old_params[ 'v_w' ]

        for i in range(len(self.layers)):
            index1 = np.where(v_w_new[ i ] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(m_w_new[ i ]),
                np.isnan(v_w_new[ i ])))

            index = [ np.concatenate((index1[ 0 ], index2[ 0 ])),
                np.concatenate((index1[ 1 ], index2[ 1 ])) ]

            if len(index[ 0 ]) > 0:
                # CHANGED: index -> tuple(index)
                # FutureWarning: Using a non-tuple sequence for multidimensional
                # indexing is deprecated; use `arr[tuple(seq)]` instead of
                # `arr[seq]`. In the future this will be interpreted as an array
                # index, `arr[np.array(seq)]`, which will result either in an
                # error or a different result.
                m_w_new[ i ][ tuple(index) ] = m_w_old[ i ][ tuple(index) ]
                v_w_new[ i ][ tuple(index) ] = v_w_old[ i ][ tuple(index) ]

    def sample_w(self):

        w = []
        for i in range(len(self.layers)):
            w.append(self.params_m_w[ i ].get_value() + \
                np.random.randn(self.params_m_w[ i ].get_value().shape[ 0 ], \
                self.params_m_w[ i ].get_value().shape[ 1 ]) * \
                np.sqrt(self.params_v_w[ i ].get_value()))

        for i in range(len(self.layers)):
            self.params_w[ i ].set_value(w[ i ])

class PBP:

    def __init__(self, layer_sizes, mean_y_train, std_y_train):

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = Prior(layer_sizes, var_targets)

        # We create the network

        params = self.prior.get_initial_params()
        self.network = Network(params[ 'm_w' ], params[ 'v_w' ],
            params[ 'a' ], params[ 'b' ])

        # We create the input and output variables in theano

        self.x = T.vector('x')
        self.y = T.scalar('y')

        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2 = \
            self.network.logZ_Z1_Z2(self.x, self.y)

        # We create a theano function for updating the posterior

        self.adf_update = theano.function([ self.x, self.y ], self.logZ,
            updates = self.network.generate_updates(self.logZ, self.logZ1,
            self.logZ2))

        # We greate a theano function for the network predictive distribution

        self.predict_probabilistic = theano.function([ self.x ],
            self.network.output_probabilistic(self.x))

        self.predict_deterministic = theano.function([ self.x ],
            self.network.output_deterministic(self.x))

    def do_pbp(self, X_train, y_train, n_iterations):

        if n_iterations > 0:

            # We first do a single pass

            self.do_first_pass(X_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            # sys.stdout.write('{}\n'.format(0))
            # sys.stdout.flush()

            for i in range(int(n_iterations) - 1):

                # We do one more pass

                params = self.prior.get_params()
                self.do_first_pass(X_train, y_train)

                # We refine the prior

                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

                # sys.stdout.write('{}\n'.format(i + 1))
                # sys.stdout.flush()

    def get_deterministic_output(self, X_test):

        output = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            output[ i ] = self.predict_deterministic(X_test[ i, : ])
            output[ i ] = output[ i ] * self.std_y_train + self.mean_y_train

        return output

    def get_predictive_mean_and_variance(self, X_test):

        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            m, v = self.predict_probabilistic(X_test[ i, : ])
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            mean[ i ] = m
            variance[ i ] = v

        v_noise = self.network.b.get_value() / \
            (self.network.a.get_value() - 1) * self.std_y_train**2

        return mean, variance, v_noise

    def do_first_pass(self, X, y):

        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ],
            replace = False)

        counter = 0
        for i in permutation:

            old_params = self.network.get_params()
            logZ = self.adf_update(X[ i, : ], y[ i ])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            # if counter % 1000 == 0:
            #     sys.stdout.write('.')
            #     sys.stdout.flush()

            counter += 1

        # sys.stdout.write('\n')
        # sys.stdout.flush()

    def sample_w(self):

        self.network.sample_w()

class PBP_net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        # We construct the network

        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], n_hidden, [ 1 ]))
        self.pbp_instance = \
            PBP(n_units_per_layer, self.mean_y_train, self.std_y_train)

        # We iterate the learning process

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

        # We are done!

    def re_train(self, X_train, y_train, n_epochs):

        """
            Function that re-trains the network on some data.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network.
        """

        # We normalize the training data

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

    def predict(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """

        X_test = np.array(X_test, ndmin = 2)

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(X_test)

        # We are done!

        return m, v, v_noise

    def predict_deterministic(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data


            @return o       The predictive value for the test target variables.
        """

        X_test = np.array(X_test, ndmin = 2)

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        o = self.pbp_instance.get_deterministic_output(X_test)

        # We are done!

        return o

    def sample_weights(self):

        """
            Function that draws a sample from the posterior approximation
            to the weights distribution.
        """

        self.pbp_instance.sample_w()

    def save_to_file(self, filename):

        """
            Function that stores the network in a file.
            @param filename   The name of the file.

        """

        # We save the network to a file using pickle

        def save_object(obj, filename):

            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
            dest.close()

        save_object(self, filename)

def load_PBP_net_from_file(filename):

    """
        Function that load a network from a file.
        @param filename   The name of the file.

    """

    def load_object(filename):

        with gzip.GzipFile(filename, 'rb') as \
            source: result = source.read()
        ret = pickle.loads(result)
        source.close()

        return ret

    # We load the dictionary with the network parameters

    PBP_network = load_object(filename)

    return PBP_network
