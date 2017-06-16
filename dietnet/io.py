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

import sys, csv, argparse, glob, json, os

from plinkio import plinkfile #install via pip: pip install plinkio
import numpy as np
import pandas as pd
import tensorflow as tf
slim = tf.contrib.slim
from sklearn.model_selection import KFold, train_test_split

_int_feature = lambda v: tf.train.Int64List(value=v)
_templ = {
        'dir':        '{pref}.diet',
        'metadata':   '{pref}.diet/metadata.json',
        'phenomap':   '{pref}.diet/phenomap.tsv',
        'plinktrans': '{pref}.diet/transpose',
        'x_t':        '{pref}.diet/x_transpose.npy',
        'npy':        '{pref}.diet/data.npy',
        'npy_fold':   '{pref}.diet/data_fold{k}_{set}.npy',
        'y':          '{pref}.diet/y.npy',
        'data':       '{pref}.diet/data.tfrecords',
        'fold':       '{pref}.diet/data_fold{k}_{set}.tfrecords',
}


def create_diet_dir(prefix):
    dirname = _templ['dir'].format(pref=prefix)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def write_records(prefix, phenotype_file,
                  nfolds=5,
                  phenotype_idcol=0,
                  phenotype_col=1,
                  phenotype_categorical=True,
                  save_tfrecords=True,
                  save_npy=False,
                  num_class = None):

    assert save_tfrecords or save_npy, 'Either TFRecords or NPY must be specified'

    create_diet_dir(prefix)

    # Read plink files
    Xt_plink = plinkfile.open(prefix)
    num_snps = len(Xt_plink.get_loci())
    num_ind = len(Xt_plink.get_samples())

    # Read sample ids from the .fam file
    fam_ids = np.array([s.iid for s in Xt_plink.get_samples()])
    pheno = pd.read_csv(phenotype_file, sep=None, engine='python')
    assert len(fam_ids) == pheno.shape[0], "Number of records in .fam file "\
                                           "and phenotype file do not match."

    assert np.all(fam_ids ==
            np.array(pheno.iloc[:,phenotype_idcol].as_matrix())),\
           "IDs of .fam file and phenotype file do not match"

    pheno_list = pheno.iloc[:, phenotype_col]

    if phenotype_categorical:
        pheno_list_cat = pheno_list.astype('category').cat
        pheno_list_values = pheno_list_cat.categories.values
        pheno_map = pd.DataFrame({'Phenotype': pheno_list_values,
                                  'Codes': range(len(pheno_list_values))},
                                  columns=('Phenotype', 'Codes'))

        pheno_map.to_csv(_templ['phenomap'].format(pref=prefix),
                         sep='\t', index=False)

        labels = pheno_list_cat.codes.astype(np.uint8)
        num_class = num_class or len(set(labels))
    else:
        # TODO: Test that
        labels = pheno_list.as_matrix()

    # Prepare indices for k-fold cv and train/valid/test split
    cv_indices = []
    for cv_trainval, cv_test in KFold(nfolds, True, 42).split(range(num_ind)):
        cv_train, cv_val = train_test_split(cv_trainval, test_size=1/(nfolds-1))
        cv_indices.append((cv_train, cv_val, cv_test))

    # Save metadata as json
    with open(_templ['metadata'].format(pref=prefix), 'w') as f:
        json.dump({'num_snp': num_snps,
                   'num_ind': num_ind,
                   'phenotype_categorical': phenotype_categorical,
                   'nfolds': nfolds,
                   'num_ind_per_fold': [(len(x),len(y),len(z)) for x,y,z in cv_indices],
                   'num_class': num_class
        }, f)

    # Transpose bed file to get X matrix
    trans_filename = _templ['plinktrans'].format(pref=prefix)
    # Produces transposed BED file
    print('Transposing plink file...')
    assert Xt_plink.transpose(trans_filename), 'Transpose failed'

    # Open transposed file and iterate over records
    X_plink = plinkfile.open(trans_filename)
    assert not X_plink.one_locus_per_row(), 'PLINK file should be transposed'
    assert len(labels) == num_ind, 'Number of labels is not equal to num individuals'

    if save_tfrecords:
        wr = lambda i, t: tf.python_io.TFRecordWriter(_templ['fold'].format(pref=prefix,
                                                                            k=i,
                                                                            set=t))
        tf_writers = [{
            'train': wr(i+1, 'train'),
            'valid': wr(i+1, 'valid'),
            'test':  wr(i+1, 'test')} for i in range(nfolds)]

        tf_writer_all = tf.python_io.TFRecordWriter(_templ['data'].format(pref=prefix))

    if save_npy:
        X = np.zeros((num_ind, num_snps), np.int8)

    # Write k-fold train/valid/test splits
    for i, (row, label) in enumerate(zip(X_plink, labels)): #iterates over individuals

        if save_tfrecords:
            # Save TFRecords
            example = tf.train.Example(features=tf.train.Features(feature={
                'genotype': tf.train.Feature(int64_list=_int_feature(list(row))),
                'label':    tf.train.Feature(int64_list=_int_feature([int(label)]))}))

            for fold, (train_idx, valid_idx, test_idx) in zip(range(nfolds), cv_indices):
                serialized_example = example.SerializeToString()
                if i in train_idx:
                    tf_writers[fold]['train'].write(serialized_example)
                elif i in valid_idx:
                    tf_writers[fold]['valid'].write(serialized_example)
                elif i in test_idx:
                    tf_writers[fold]['test'].write(serialized_example)
                else:
                    raise 'Not valid index'
            tf_writer_all.write(serialized_example)

        if save_npy:
            X[i, :] = list(row)

        if i % 100 == 0:
            print('Writing genotypes... {:.2f}% completed'.format((i/num_ind)*100), end='\r')
            sys.stdout.flush()

    # Save fold as npy if requested
    if save_npy:
        for i, (train_idx, valid_idx, test_idx) in zip(range(nfolds), cv_indices):
            fold_filename = _templ['npy_fold'].format(pref=prefix, k=i+1, set='train')
            np.save(fold_filename, X[train_idx,])
            fold_filename = _templ['npy_fold'].format(pref=prefix, k=i+1, set='valid')
            np.save(fold_filename, X[valid_idx,])
            fold_filename = _templ['npy_fold'].format(pref=prefix, k=i+1, set='test')
            np.save(fold_filename, X[test_idx,])

        np.save(_templ['npy'].format(pref=prefix), X)

    print('\nDone')

    if save_tfrecords:
        for fold in range(nfolds):
            tf_writers[fold]['train'].close()
            tf_writers[fold]['valid'].close()
            tf_writers[fold]['test'].close()
        tf_writer_all.close()

    Xt = np.zeros([num_snps, num_ind], np.int8)
    for i, row in enumerate(Xt_plink): #iterates over snps
        Xt[i,:] = row
        if i % 1000 == 0:
            print('Writing X transpose matrix... {:.2f}% completed'.format((i/num_snps)*100), end='\r')
            sys.stdout.flush()
    print('\nDone')

    # Save X^T as numpy arrays
    np.save(_templ['x_t'].format(pref=prefix), Xt)


def read_metadata(prefix):
    meta = json.load(open(_templ['metadata'].format(pref=prefix)))
    return meta


def get_fold_files(prefix, fold=None, sets=('train', 'valid', 'test'),
                   file_format='tfrecords'):
    meta = read_metadata(prefix)
    nfolds = meta['nfolds']
    if file_format == 'tfrecords':
        pattern = _templ['fold']
    elif file_format == 'npy':
        pattern = _templ['npy_fold']
    else:
        raise Exception('Unknown input format')

    if fold is not None:
        yield [pattern.format(pref=prefix, k=fold, set=s) for s in sets]
    else:
        for f in range(1, nfolds+1):
            yield [pattern.format(pref=prefix, k=f, set=s) for s in sets]


def read_batch_from_file(prefix, filename, batch_size, file_format='tfrecords'):
    meta = read_metadata(prefix)
    num_snps = meta['num_snp']
    num_class = meta['num_class']

    if file_format == 'tfrecords':
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([filename])

        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'genotype': tf.FixedLenFeature([num_snps], tf.int64),
                'label':    tf.FixedLenFeature([1], tf.int64)
            })

        outputs = tf.train.batch(features,
                                 batch_size=batch_size,
                                 capacity=batch_size*50)
    elif file_format == 'npy':
        x = np.load(filename)
        y = np.load(_templ['y'].format(pref=prefix))
        x_batch, y_batch = tf.train.shuffle_batch([x, y],
                                                  batch_size,
                                                  enqueue_many=True,
                                                  capacity=batch_size*100,
                                                  min_after_dequeue=batch_size*30)
        y_batch = slim.one_hot_encoding(y_batch, num_classes)
        assert(x_batch.get_shape()[0] == y_batch.get_shape()[0])

        return (tf.cast(x_batch, tf.float32),
                tf.convert_to_tensor(xt, tf.float32),
                tf.cast(y_batch, tf.float32))

    else:
        raise Exception('Invalid input file format')

    outputs['label'] = slim.one_hot_encoding(outputs['label'], num_class)

    # squeeze to remove singletons
    outputs = {'genotype': tf.cast(tf.squeeze(outputs['genotype']), tf.float32),
               'label':    tf.cast(tf.squeeze(outputs['label']), tf.float32)}

    return outputs


def read_batch_from_fold(prefix, batch_size, fold=None,
                         sets=('train', 'valid', 'test')):
    filenames = get_fold_files(prefix, fold=fold, sets=sets)
    for fold_file in filenames:
        yield [read_batch_from_file(prefix, f, batch_size)
               for f in fold_file]


def read_transpose(prefix):
    xt = np.load(_templ['x_t'].format(pref=prefix))
    return tf.convert_to_tensor(xt, tf.float32)


def preprocess(args):
    write_records(args.prefix, args.pheno,
            nfolds=args.kfold,
            phenotype_idcol=args.phenoidcol,
            phenotype_col=args.phenocol,
            phenotype_categorical=args.categorical,
            save_tfrecords=args.savequeue,
            save_npy=args.savenpy,
            num_class=args.numclasses)
