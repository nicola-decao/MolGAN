import tensorflow as tf


def graph_convolution_layer(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))

    annotations = tf.concat((hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor

    output = tf.stack([tf.layers.dense(inputs=annotations, units=units) for _ in range(adj.shape[1])], 1)

    output = tf.matmul(adj, output)
    output = tf.reduce_sum(output, 1) + tf.layers.dense(inputs=annotations, units=units)
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def graph_aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid)
    j = tf.layers.dense(inputs, units=units, activation=activation)
    output = tf.reduce_sum(i * j, 1)
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs
    for u in units:
        hidden_tensor = tf.layers.dense(hidden_tensor, units=u, activation=activation)
        hidden_tensor = tf.layers.dropout(hidden_tensor, dropout_rate, training=training)

    return hidden_tensor


def multi_graph_convolution_layers(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = graph_convolution_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                                units=u, activation=activation, dropout_rate=dropout_rate,
                                                training=training)

    return hidden_tensor
