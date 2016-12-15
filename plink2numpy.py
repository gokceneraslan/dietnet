#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, csv, argparse, glob, json

from plinkio import plinkfile #install via pip: pip install plinkio
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

_int_feature = lambda v: tf.train.Int64List(value=v)

def write_records(prefix, phenotype_file,
                  nfolds=5,
                  phenotype_idcol=0,
                  phenotype_col=1,
                  phenotype_categorical=True):

    # Read plink files
    Xt_plink = plinkfile.open(prefix)
    num_snps = len(Xt_plink.get_loci())
    num_ind = len(Xt_plink.get_samples())

    with open('%s.dietmetadata' % prefix, 'w') as f:
        json.dump({'num_snp': num_snps,
                   'num_ind': num_ind,
                   'phenotype_categorical': phenotype_categorical,
                   'nfolds': nfolds}, f)

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

        pheno_map.to_csv('{}.phenomap'.format(prefix), sep='\t', index=False)

        labels = pheno_list_cat.codes.astype(np.uint8)
    else:
        # TODO: Test that
        labels = pheno_list.as_matrix()

    # Transpose bed file to get X matrix
    trans_filename = '%s_transpose' % prefix
    # Produces transposed BED file
    print('Transposing plink file...')
    assert Xt_plink.transpose(trans_filename), 'Transpose failed'

    # Open transposed file and iterate over records
    X_plink = plinkfile.open(trans_filename)

    assert len(labels) == num_ind, 'Number of labels is not equal to num individuals'

    tf_writers = [{
        'train': tf.python_io.TFRecordWriter('%s_fold%i_train.tfrecords' % (prefix, i+1)),
        'valid': tf.python_io.TFRecordWriter('%s_fold%i_valid.tfrecords' % (prefix, i+1)),
        'test':  tf.python_io.TFRecordWriter('%s_fold%i_test.tfrecords'  % (prefix, i+1))}
        for i in range(nfolds)]
    tf_writer_all = tf.python_io.TFRecordWriter('%s.tfrecords' % prefix)

    # Prepare indices for k-fold cv and train/valid/test split
    cv_indices = []
    for cv_trainval, cv_test in KFold(nfolds).split(range(num_ind)):
        cv_train, cv_val = train_test_split(cv_trainval, test_size=1/(nfolds-1))
        cv_indices.append((cv_train, cv_val, cv_test))

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
    print('Done')

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
    print('Done')

    # Save X and X^T as numpy arrays
    np.save('{}_x_transpose.npy'.format(prefix), Xt)


def get_fold_files(prefix):
    meta = json.load(open('%s.dietmetadata' % prefix))
    nfolds = int(meta['nfolds'])
    pattern = '%s_fold%i_%s.tfrecords'
    fold_files = {
            'train': [pattern % (prefix, i+1, 'train') for i in range(nfolds)],
            'valid': [pattern % (prefix, i+1, 'valid') for i in range(nfolds)],
            'test':  [pattern % (prefix, i+1, 'test')  for i in range(nfolds)]}

    return fold_files


def read_input(prefix, filename, batch_size):
    meta = json.load(open('%s.dietmetadata' % prefix))
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

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert plink files to '
                                                  'numpy arrays.')
    parser.add_argument('-f', '--prefix', type=str, help='PLINK prefix',
                        required=True)
    parser.add_argument('-p', '--pheno', type=str,
            help='TSV/CSV formatted phenotype file', required=True)
    parser.add_argument('-k', '--kfold', type=int, default=5,
            help='Number of folds in cross-validation (default=5)')
    parser.add_argument('-i', '--phenoidcol', type=int,
            help='0-based column index of sample ids in phenotype file (default=0)',
            default=0)
    parser.add_argument('-j', '--phenocol', type=int,
            help='0-based column index of phenotypes in phenotype file (default=1)',
            default=1)
    parser.add_argument('-c', '--categorical', type=bool,
            help='Phenotype is categorical (default=True)', default=True)

    args = parser.parse_args()
    write_records(args.prefix, args.pheno,
            nfolds=args.kfold,
            phenotype_idcol=args.phenoidcol,
            phenotype_col=args.phenocol,
            phenotype_categorical=args.categorical)
