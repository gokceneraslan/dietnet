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
from keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer


class Bias(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.bias_add(x, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


#non-parametric dense layer
NPDense = lambda name: Lambda(lambda x: K.dot(x[0], x[1]), name=name)


# embedding closure for shared parameters
def Embedding(size, activation='elu', dropout_rate=0.):
    dense = Dense(size)
    bn    = BatchNormalization()
    act   = Activation(activation)
    drop  = Dropout(dropout_rate)
    def emb(x):
        return drop(act(bn(dense(x))))
    return emb


def MLP(inp, size=100, activation='linear', dropout_rate=0.):
    ret = Dense(size)(inp)
    ret = BatchNormalization()(ret)
    ret = Activation(activation)(ret)
    if dropout_rate > 0.:
        ret = Dropout(dropout_rate)(ret)
    return ret


def diet(input_size, output_size, xt_size=None,
         batch_size=64,
         hidden_size=100,
         embedding_size=100,
         dropout_rate=0.,
         use_aux=True,
         autoencoder=True,
         gamma=1.,
         share_embedding=True,
         activation='elu'):

    X = Input(shape=(input_size,), name='x')
    inputs = [X]

    if use_aux:
        assert xt_size, "use_aux requires x_transpose matrix"
        Xt = Input(shape=(xt_size,), name='x_t')
        inputs += [Xt]

        # might be shared
        embed = Embedding(embedding_size, activation, dropout_rate=dropout_rate)
        embed_xt = embed(Xt)

        We = MLP(embed_xt, hidden_size, 'linear', dropout_rate)

        hidden0 = NPDense(name='fat_hidden')([X, We])
        # add bias
        hidden0 = Bias(hidden_size, name='fat_hidden_bias')(hidden0)
        hidden0 = Activation(activation, name='fat_hidden_act')(hidden0)
    else:
        hidden0 = Dense(hidden_size, activation=activation, name='fat_hidden')(X)

    hidden1 = MLP(hidden0, hidden_size, activation, dropout_rate)
    predictions = MLP(hidden1, size=output_size, activation='softmax')
    outputs = [predictions]
    loss = ['categorical_crossentropy']
    loss_weights = [1.]

    if autoencoder:
        if use_aux:
            if not share_embedding:
                #create new embedding
                embed = Embedding(embedding_size, dropout_rate=dropout_rate)
                embed_xt = embed(Xt)

            Wd = MLP(embed_xt, hidden_size, 'linear', dropout_rate)
            Wd = Lambda(lambda x: K.transpose(x), name='transpose')(Wd)

            xhat = NPDense(name='fat_hidden_ae')([hidden1, Wd])
            # add bias
            xhat = Bias(hidden_size, name='fat_hidden_bias_ae')(xhat)
        else:
            xhat = Dense(hidden_size, name='fat_hidden_ae')(hidden1)

        outputs += [xhat]
        loss += ['mse']
        loss_weights += [gamma] #coefficient of AE loss

    model = Model(inputs=inputs, outputs=outputs)

    return model, loss, loss_weights

