import numpy as np
import tensorflow as tf

from models import postprocess_logits
from utils.layers import multi_dense_layers


class GraphVAEModel:

    def __init__(self, vertexes, edges, nodes, features, embedding_dim, encoder_units, decoder_units, variational,
                 encoder, decoder, soft_gumbel_softmax=False, hard_gumbel_softmax=False, with_features=True):
        self.vertexes, self.nodes, self.edges, self.embedding_dim, self.encoder, self.decoder = \
            vertexes, nodes, edges, embedding_dim, encoder, decoder

        self.training = tf.placeholder_with_default(False, shape=())
        self.variational = tf.placeholder_with_default(variational, shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder_with_default(1., shape=())

        self.edges_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes, vertexes))
        self.nodes_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes))
        self.node_features = tf.placeholder(dtype=tf.float32, shape=(None, vertexes, features))

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.adjacency_tensor = tf.one_hot(self.edges_labels, depth=edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(self.nodes_labels, depth=nodes, dtype=tf.float32)

        with tf.variable_scope('encoder'):
            outputs = self.encoder(
                (self.adjacency_tensor, self.node_features if with_features else None, self.node_tensor),
                units=encoder_units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=encoder_units[-1], activation=tf.nn.tanh,
                                         training=self.training, dropout_rate=0.)

            self.embeddings_mean = tf.layers.dense(outputs, embedding_dim, activation=None)
            self.embeddings_std = tf.layers.dense(outputs, embedding_dim, activation=tf.nn.softplus)
            self.q_z = tf.distributions.Normal(self.embeddings_mean, self.embeddings_std)

            self.embeddings = tf.cond(self.variational,
                                      lambda: self.q_z.sample(),
                                      lambda: self.embeddings_mean)

        with tf.variable_scope('decoder'):
            self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                                                                training=self.training, dropout_rate=0.)

        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
            (self.edges_argmax, self.nodes_argmax), \
            (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
            (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
            (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                (self.edges_logits, self.nodes_logits), temperature=self.temperature)

            self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                                     default=lambda: self.edges_softmax,
                                     exclusive=True)

            self.nodes_hat = tf.case({self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                          self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                                     default=lambda: self.nodes_softmax,
                                     exclusive=True)

        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.V_x((self.adjacency_tensor, None, self.node_tensor), units=encoder_units)
        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.V_x((self.edges_hat, None, self.nodes_hat), units=encoder_units)

    def V_x(self, inputs, units):
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.encoder(inputs, units=units[:-1], training=self.training, dropout_rate=0.)

            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=0.)

            outputs = tf.layers.dense(outputs, units=1, activation=tf.nn.sigmoid)

        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))
