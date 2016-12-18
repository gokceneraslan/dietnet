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

import time

from .network import embedding, auxnet, diet
from . import io

import tensorflow as tf
slim = tf.contrib.slim


def train(args):
    meta = io.read_metadata(args.prefix)
    folds = io.read_batch_from_fold(args.prefix,
                                    args.batchsize,
                                    fold=args.fold,
                                    sets=('train', 'valid'))
    x_t = io.read_transpose(args.prefix)

    for fold_i, (train, valid) in enumerate(folds):
        loss = diet(train['genotype'], train['label'], x_t,
                    batch_size=args.batchsize,
                    hidden_size=args.hiddensize,
                    embedding_size=args.embeddingsize,
                    dropout_rate=1-args.dropoutrate, #switch to dropout keep prob.
                    is_training=True,
                    use_aux=args.useaux,
                    gamma=args.gamma,
                    autoencoder=args.autoencoder,
                    share_embedding=args.shareembedding)

        optimizer = tf.train.RMSPropOptimizer(args.learningrate)
        train_op = slim.learning.create_train_op(loss, optimizer,
                                                 summarize_gradients=True,
                                                 clip_gradient_norm=10)
        summary_ops = tf.summary.merge_all()

        # Create coordinator
        coord = tf.train.Coordinator()
        fold_logdir = '%s/fold%s' % (args.logdir, fold_i)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            #TODO: Handle checkpoint restore
            # Saver for storing checkpoints of the model
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            swriter = tf.summary.FileWriter(fold_logdir, sess.graph)
            step = 0

            try:
                while True:
                    start_time = time.time()
                    trainloss, summaries = sess.run([train_op, summary_ops])
                    swriter.add_summary(summaries)

                    valloss = sess.run(loss,
                            feed_dict={'inputs:0': valid['genotype'].eval(),
                                       'outputs:0': valid['label'].eval()})

                    duration = time.time() - start_time
                    print('step {:d} - loss = {:.3f} vallos = {:.3f}, ({:.3f} sec/step)'
                          .format(step, trainloss, valloss, duration))

                    if step % args.checkpoint_every == 0:
                        saver.save(sess, fold_logdir, global_step=step)
                        last_saved_step = step

                    step += 1

            except KeyboardInterrupt as e:
                coord.request_stop()
                break
            finally:
                coord.request_stop()
                coord.join(threads)
                swriter.close()



