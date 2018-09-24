import tensorflow as tf
from utils.layers import multi_graph_convolution_layers, graph_aggregation_layer, multi_dense_layers


def encoder_rgcn(inputs, units, training, dropout_rate=0.):
    graph_convolution_units, auxiliary_units = units

    with tf.variable_scope('graph_convolutions'):
        output = multi_graph_convolution_layers(inputs, graph_convolution_units, activation=tf.nn.tanh,
                                                dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('graph_aggregation'):
        _, hidden_tensor, node_tensor = inputs
        annotations = tf.concat(
            (output, hidden_tensor, node_tensor) if hidden_tensor is not None else (output, node_tensor), -1)

        output = graph_aggregation_layer(annotations, auxiliary_units, activation=tf.nn.tanh,
                                         dropout_rate=dropout_rate, training=training)

    return output


def decoder_adj(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units, activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(tf.layers.dense(inputs=output, units=edges * vertexes * vertexes,
                                                  activation=None), (-1, edges, vertexes, vertexes))
        edges_logits = tf.transpose((edges_logits + tf.matrix_transpose(edges_logits)) / 2, (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = tf.layers.dense(inputs=output, units=vertexes * nodes, activation=None)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_dot(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(tf.layers.dense(inputs=output, units=edges * vertexes * units[-1],
                                                  activation=None), (-1, edges, vertexes, units[-1]))
        edges_logits = tf.transpose(tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = tf.layers.dense(inputs=output, units=vertexes * nodes, activation=None)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_rnn(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)

        edges_logits = tf.layers.dense(edges_logits, edges * units[-1])
        edges_logits = tf.transpose(tf.reshape(edges_logits, (-1, vertexes, edges, units[-1])), (0, 2, 1, 3))
        edges_logits = tf.transpose(tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)
        nodes_logits = tf.layers.dense(nodes_logits, nodes)
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def postprocess_logits(inputs, temperature=1.):

    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    softmax = [tf.nn.softmax(e_logits / temperature)
               for e_logits in listify(inputs)]
    argmax = [tf.one_hot(tf.argmax(e_logits, axis=-1), depth=e_logits.shape[-1], dtype=e_logits.dtype)
              for e_logits in listify(inputs)]
    gumbel_logits = [e_logits - tf.log(- tf.log(tf.random_uniform(tf.shape(e_logits), dtype=e_logits.dtype)))
                     for e_logits in listify(inputs)]
    gumbel_softmax = [tf.nn.softmax(e_gumbel_logits / temperature)
                      for e_gumbel_logits in gumbel_logits]
    gumbel_argmax = [
        tf.one_hot(tf.argmax(e_gumbel_logits, axis=-1), depth=e_gumbel_logits.shape[-1], dtype=e_gumbel_logits.dtype)
        for e_gumbel_logits in gumbel_logits]

    return [delistify(e) for e in (softmax, argmax, gumbel_logits, gumbel_softmax, gumbel_argmax)]
