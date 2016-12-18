# Copyright 2016 Goekcen Eraslan
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
# ==============================================================================

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def embedding(inputs, size, dropout_rate=0.5, is_training=True, reuse=None, scope=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.005),
                        activation_fn=tf.nn.relu):

        # Embedding layer, can be shared depending on the reuse parameter
        net = slim.fully_connected(inputs,
                                   size,
                                   scope=scope,
                                   reuse=reuse)
        net = slim.dropout(net, dropout_rate,
                is_training=is_training, scope='dropout')
        tf.summary.histogram('activations/auxnet/embedding', net)
    return net


def auxnet(embedding, size, dropout_rate=0.5, is_training=True, scope='auxnet'):
    # MLP in auxiliary networks
    with tf.variable_scope(scope, 'AuxNet'):
        with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(0.005),
                activation_fn=tf.nn.relu):

            net = slim.fully_connected(embedding, size, scope='hidden')
            net = slim.dropout(net, dropout_rate,
                    is_training=is_training, scope='dropout')
            net = slim.fully_connected(net, size, scope='output',
                    activation_fn=None)

    tf.summary.histogram('activations/auxnet/%s' % scope, net)
    return net


def diet(inputs, outputs, xt,
         batch_size=64,
         hidden_size=100,
         embedding_size=100,
         dropout_rate=0.5,
         is_training=True,
         use_aux=True,
         autoencoder=True,
         gamma=1,
         share_embedding=True,
         scope=None):

    with slim.arg_scope([slim.fully_connected],
                      activation_fn=tf.nn.relu):

        input_size = inputs.get_shape().as_list()[1]
        output_size = outputs.get_shape().as_list()[1]

        # use placeholder_with_default hack to be able to run valid set
        # see: https://github.com/tensorflow/tensorflow/issues/2514
        inputs  = tf.placeholder_with_default(inputs,  [None,  input_size], name='inputs')
        outputs = tf.placeholder_with_default(outputs, [None, output_size], name='outputs')

        if use_aux:
            embed = embedding(xt, embedding_size, dropout_rate=dropout_rate,
                              is_training=is_training, scope='auxembed')
            We = auxnet(embed, hidden_size, dropout_rate=dropout_rate,
                        is_training=is_training, scope='aux_We')

        else:
            We = slim.model_variable('We',
                                     shape=(input_size, hidden_size),
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     regularizer=slim.l2_regularizer(0.0005))

        We = tf.clip_by_norm(We, clip_norm=1)

        tf.summary.histogram('weights/dietnet/We', We)

        output_h1 = tf.matmul(inputs, We)
        output_h1 = slim.bias_add(output_h1, activation_fn=tf.nn.relu)
        tf.summary.histogram('activations/dietnet/output_h1', output_h1)
        output_h1 = slim.dropout(output_h1, dropout_rate, is_training=is_training)

        with tf.variable_scope(scope, 'DietNet'):
            class_predictions = slim.fully_connected(output_h1, output_size,
                                                     activation_fn=None,
                                                     scope='output')

        entropy_loss = slim.losses.softmax_cross_entropy(class_predictions, outputs)
        tf.summary.scalar('loss/crossent_loss', entropy_loss)
        tf.summary.scalar('accuracy',
                slim.metrics.accuracy(tf.argmax(class_predictions, 1),
                                      tf.argmax(outputs, 1)))

        if autoencoder:
            if use_aux:
                if share_embedding:
                    Wd = auxnet(embed, hidden_size, dropout_rate=dropout_rate,
                                is_training=is_training, scope='aux_Wd')
                else:
                    embed2 = embedding(xt, embedding_size, scope='auxembed2')
                    Wd = auxnet(embed2, hidden_size,
                            dropout_rate=dropout_rate,
                            is_training=is_training, scope='aux_Wd')
                Wd = tf.transpose(Wd)
            else:
                Wd = slim.model_variable('Wd',
                                         shape=(hidden_size, input_size),
                                         initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         regularizer=slim.l2_regularizer(0.0005))

            Wd = tf.clip_by_norm(Wd, clip_norm=1)
            tf.summary.histogram('weights/dietnet/Wd', Wd)

            xhat = tf.matmul(output_h1, Wd)
            xhat = slim.bias_add(xhat, activation_fn=tf.nn.relu)
            mean_squared_loss = slim.losses.mean_squared_error(xhat,
                                                               inputs,
                                                               weight=gamma)
            tf.summary.scalar('loss/autoencoder_mse_loss', mean_squared_loss)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss/total_loss', total_loss)

    return total_loss

