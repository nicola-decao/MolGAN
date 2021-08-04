import numpy as np
import tensorflow as tf


class GraphVAEOptimizer(object):

    def __init__(self, model, learning_rate=1e-3):

        self.kl_weight = tf.placeholder_with_default(1., shape=())
        self.la = tf.placeholder_with_default(1., shape=())

        edges_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.edges_labels,
                                                            logits=model.edges_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.edges_loss = tf.reduce_sum(edges_loss, [-2, -1])

        nodes_loss = tf.losses.sparse_softmax_cross_entropy(labels=model.nodes_labels,
                                                            logits=model.nodes_logits,
                                                            reduction=tf.losses.Reduction.NONE)
        self.nodes_loss = tf.reduce_sum(nodes_loss, -1)

        self.loss_ = self.edges_loss + self.nodes_loss
        self.reconstruction_loss = tf.reduce_mean(self.loss_)

        self.p_z = tf.distributions.Normal(tf.zeros_like(model.embeddings_mean),
                                           tf.ones_like(model.embeddings_std))
        self.kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(model.q_z, self.p_z), axis=-1))

        self.ELBO = - self.reconstruction_loss - self.kl

        self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + (model.value_logits_fake - model.rewardF) ** 2

        self.loss_RL = - model.value_logits_fake

        self.loss_VAE = tf.cond(model.variational,
                                lambda: self.reconstruction_loss + self.kl_weight * self.kl,
                                lambda: self.reconstruction_loss)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        self.loss_RL *= tf.abs(tf.stop_gradient(self.loss_VAE / self.loss_RL))

        self.VAE_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_VAE = self.VAE_optim.minimize(
            loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_VAE, lambda: 0.) + tf.cond(
                tf.less(self.la, 1), lambda: (1 - self.la) * self.loss_RL, lambda: 0.),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder'))

        self.V_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step_V = self.V_optim.minimize(
            loss=self.loss_V,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))

        self.log_likelihood = self.__log_likelihood
        self.model = model

    def __log_likelihood(self, n):

        z = self.model.q_z.sample(n)

        log_p_z = self.p_z.log_prob(z)
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

        log_p_x_z = -self.loss_

        log_q_z_x = self.model.q_z.log_prob(z)
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

        print([a.shape for a in (log_p_z, log_p_x_z, log_q_z_x)])

        return tf.reduce_mean(tf.reduce_logsumexp(
            tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))
