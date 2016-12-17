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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .network import embedding, auxnet, diet


def read_input(prefix, batch_size, num_classes, filename):
    xt = np.load(prefix + '_x_transpose.npy')
    x_batch, y_batch = io.read_input(prefix, filename, batch_size )
    y_batch = slim.one_hot_encoding(y_batch, num_classes)
    assert(x_batch.get_shape()[0] == y_batch.get_shape()[0])

    return (tf.cast(x_batch, tf.float32),
            tf.convert_to_tensor(xt, tf.float32),
            tf.cast(y_batch, tf.float32))


def train(args):
    x, xt, y = read_input(args.prefix, args.batchsize, args.numclasses)
    net = diet(x, y, xt=xt,
               batch_size=args.batchsize,
               hidden_size=args.hiddensize,
               embedding_size=args.embeddingsize,
               dropout_rate=1-args.dropoutrate, #switch to dropout keep prob.
               is_training=args.training,
               use_aux=args.aux,
               gamma=args.gamma,
               autoencoder=args.autoencoder,
               share_embedding=args.shareembedding)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss/total_loss', total_loss)
    optimizer = tf.train.RMSPropOptimizer(args.learningrate)
    train_op = slim.learning.create_train_op(total_loss, optimizer,
                                             summarize_gradients=True,
                                             clip_gradient_norm=10)
    summary_ops = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners()
        #TODO: Handle checkpoint saver/restore
        swriter = tf.summary.FileWriter(args.logdir, sess.graph)

        for step in range(args.numsteps):
            loss, summaries = sess.run([train_op, summary_ops])
            swriter.add_summary(summaries)

        swriter.close()


