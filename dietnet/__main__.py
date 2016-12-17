#!/usr/bin/env python

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

import os, sys, argparse
import numpy as np
import six
import tensorflow as tf
slim = tf.contrib.slim

from . import io
from .network import embedding, auxnet, diet


def parse_args():
    parser = argparse.ArgumentParser(description='Diet network')
    subparsers = parser.add_subparsers(title='subcommands',
            help='sub-command help', description='valid subcommands', dest='cmd')
    subparsers.required = True

    # Preprocess subparser
    parser_preprocess = subparsers.add_parser('preprocess',
            help='Convert plink format to required TFRecords and split CV folds')
    parser_preprocess.set_defaults(func=preprocess)

    parser_preprocess.add_argument('-f', '--prefix', type=str, help='PLINK prefix',
            required=True)
    parser_preprocess.add_argument('-p', '--pheno', type=str,
            help='TSV/CSV formatted phenotype file', required=True)
    parser_preprocess.add_argument('-k', '--kfold', type=int, default=5,
            help='Number of folds in cross-validation (default=5)')
    parser_preprocess.add_argument('-i', '--phenoidcol', type=int,
            help='0-based column index of sample ids in phenotype file (default=0)',
            default=0)
    parser_preprocess.add_argument('-j', '--phenocol', type=int,
            help='0-based column index of phenotypes in phenotype file (default=1)',
            default=1)
    parser_preprocess.add_argument('-c', '--categorical', type=bool,
            help='Phenotype is categorical (default=True)', default=True)

    # train subparser
    parser_train = subparsers.add_parser('train',
            help='Start training using given training set.')
    parser_train.set_defaults(func=train)

    parser_train.add_argument('-f', '--prefix', type=str, default='genotypes',
            help="Prefix of inputs e.g. genotypes if files "
            "are named genotypes_x.npy")
    parser_train.add_argument('-d', '--logdir', type=str, default='logs',
            help="The directory where training logs are written to")
    parser_train.add_argument('-b', '--batchsize', type=int, default=128,
            help="Batch size")
    parser_train.add_argument('--dropoutrate', type=float, default=0.0,
            help="Dropout rate")
    parser_train.add_argument('--numsteps', type=int, default=5000,
            help="The max number of gradient steps to take during training")
    parser_train.add_argument('--hiddensize', type=int, default=100,
            help="Size of hidden layers")
    parser_train.add_argument('--embeddingsize', type=int, default=100,
            help="Size of embedding layers")
    parser_train.add_argument('-l', '--learningrate', type=float, default=1e-4,
            help="Learning rate")
    parser_train.add_argument('--gamma', type=float, default=1,
            help="Loss weight of autoencoder")
    parser_train.add_argument('-c', '--numclasses', type=int, required=True,
            help="Total number of classes")
    parser_train.add_argument('--aux', action='store_true',
            help="Use auxiliary networks to reduce number of parameters.")
    parser_train.add_argument('--autoencoder', action='store_true',
            help="Enable autoencoder")
    parser_train.add_argument('embeddingtype', choices=['rawend2end'],
            help="Type of embedding: Only raw_end2end supported.")
    parser_train.add_argument('--shareembedding', action='store_true',
            help="Share embeddings of auxiliary nets")

    return parser.parse_args()


def preprocess(args):
    io.run_args(args)


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


def main():
    args = parse_args()
    args.func(args)
