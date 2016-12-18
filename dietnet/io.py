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
from sklearn.model_selection import KFold, train_test_split

_int_feature = lambda v: tf.train.Int64List(value=v)
_templ = {
        'dir':        '{pref}.diet',
        'metadata':   '{pref}.diet/{pref}.metadata',
        'phenomap':   '{pref}.diet/{pref}.phenomap',
        'plinktrans': '{pref}.diet/{pref}_transpose',
        'x_t':        '{pref}.diet/{pref}_transpose.npy',
        'data':       '{pref}.diet/{pref}.tfrecords',
        'fold':       '{pref}.diet/{pref}_fold{k}_{set}.tfrecords',
}


def create_diet_dir(prefix):
    dirname = _templ['dir'].format(pref=prefix)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def write_records(prefix, phenotype_file,
                  nfolds=5,
                  phenotype_idcol=0,
                  phenotype_col=1,
                  phenotype_categorical=True):

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
    else:
        # TODO: Test that
        labels = pheno_list.as_matrix()

    # Transpose bed file to get X matrix
    trans_filename = _templ['plinktrans'].format(pref=prefix)
    # Produces transposed BED file
    print('Transposing plink file...')
    assert Xt_plink.transpose(trans_filename), 'Transpose failed'

    # Open transposed file and iterate over records
    X_plink = plinkfile.open(trans_filename)

    assert len(labels) == num_ind, 'Number of labels is not equal to num individuals'
    wr = lambda i, t: tf.python_io.TFRecordWriter(_templ['fold'].format(pref=prefix,
                                                                        k=i,
                                                                        set=t))
    tf_writers = [{
        'train': wr(i+1, 'train'),
        'valid': wr(i+1, 'valid'),
        'test':  wr(i+1, 'test')} for i in range(nfolds)]

    tf_writer_all = tf.python_io.TFRecordWriter(_templ['data'].format(pref=prefix))

    # Prepare indices for k-fold cv and train/valid/test split
    cv_indices = []
    for cv_trainval, cv_test in KFold(nfolds).split(range(num_ind)):
        cv_train, cv_val = train_test_split(cv_trainval, test_size=1/(nfolds-1))
        cv_indices.append((cv_train, cv_val, cv_test))

    # Save metadata as json
    with open(_templ['metadata'].format(pref=prefix), 'w') as f:
        json.dump({'num_snp': num_snps,
                   'num_ind': num_ind,
                   'phenotype_categorical': phenotype_categorical,
                   'nfolds': nfolds,
                   'num_ind_per_fold': [(len(x),len(y),len(z)) for x,y,z in cv_indices]}, f)

    # Write k-fold train/valid/test splits
    for i, (row, label) in enumerate(zip(X_plink, labels)): #iterates over individuals
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

        if i % 100 == 0:
            print('Writing genotypes... {:.2f}% completed'.format((i/num_ind)*100), end='\r')
            sys.stdout.flush()
    print('\nDone')

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


def get_fold_files(prefix, fold=None, sets=('train', 'valid', 'test')):
    meta = read_metadata(prefix)
    nfolds = meta['nfolds']
    pattern = _templ['fold']

    if fold is not None:
        yield [pattern.format(pref=prefix, k=fold+1, set=s) for s in sets]
    else:
        for f in range(nfolds):
            yield [pattern.format(pref=prefix, k=f+1, set=s) for s in sets]


def read_batch_from_file(prefix, filename, batch_size, num_classes):
    meta = read_metadata(prefix)
    num_snps = int(meta['num_snp'])

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

    outputs['label'] = slim.one_hot_encoding(outputs['label'], num_classes)

    # squeeze to remove singletons
    outputs = {'genotype': tf.cast(tf.squeeze(outputs['genotype']), tf.float32),
               'label':    tf.cast(tf.squeeze(outputs['label']), tf.float32)}

    return outputs


def read_batch_from_fold(prefix, batch_size, num_classes, fold=None,
                         sets=('train', 'valid', 'test')):
    filenames = get_fold_files(prefix, fold=fold, sets=sets)
    for fold_file in filenames:
        yield [read_batch_from_file(prefix, f, batch_size, num_classes)
               for f in fold_file]


def read_transpose(prefix):
    xt = np.load(_templ['x_t'].format(pref=prefix))
    return tf.convert_to_tensor(xt, tf.float32),


def preprocess(args):
    write_records(args.prefix, args.pheno,
            nfolds=args.kfold,
            phenotype_idcol=args.phenoidcol,
            phenotype_col=args.phenocol,
            phenotype_categorical=args.categorical)
