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
import bcolz as bc
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .sequence import BcolzSequence, OrderedEnqueuer


class Config:
    def __init__(self, prefix):
        self.prefix = prefix
        self._files = {
                'dir':        '{pref}.diet',
                'metadata':   '{pref}.diet/metadata.json',
                'pheno':      '{pref}.diet/phenomap.bcz',
                'plinktrans': '{pref}.diet/transpose', #prefix. will be created by plinkio.
                'x_t':        '{pref}.diet/x_t.bcz',
                'x':          '{pref}.diet/x.bcz',
                'y':          '{pref}.diet/y.bcz',
                'x_fold':     '{pref}.diet/x_fold{{k}}_{{set}}.bcz',
                'y_fold':     '{pref}.diet/y_fold{{k}}_{{set}}.bcz'
                }

        if not os.path.exists(prefix):
            os.mkdir(prefix)

    def __getattr__(self, name):
        return self._files[name].format(pref=self.prefix)

config = None


class Dataset(object):

    def __init__(self, prefix):
        self.prefix = prefix
        self.config = Config(prefix)
        self.meta = None

    def from_plink(self,
             plink_prefix,
             phenotype_file,
             nfolds=5,
             phenotype_idcol=0,
             phenotype_col=1,
             phenotype_categorical=True,
             num_class = None,
             disk = False):

        bc.cparams.setdefaults(shuffle=0)

        # Read plink files
        Xt_plink = plinkfile.open(plink_prefix)
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

            pheno_df = bc.ctable.fromdataframe(pheno_map, rootdir=self.config.pheno,
                                               mode='w')
            pheno_df.flush()
            labels = pheno_list_cat.codes.astype(np.uint8)
            num_class = num_class or len(set(labels))
        else:
            # TODO: Test that
            labels = pheno_list.as_matrix()

        # Prepare indices for k-fold cv and train/valid/test split
        cv_indices = []
        for cv_trainval, cv_test in KFold(nfolds, True, 42).split(range(num_ind)):
            cv_train, cv_val = train_test_split(cv_trainval, test_size=1/(nfolds-1))
            cv_indices.append((list(cv_train), list(cv_val), list(cv_test)))

        # Save metadata as json
        self.meta = {'num_snp': num_snps,
                'num_ind': num_ind,
                'phenotype_categorical': phenotype_categorical,
                'nfolds': nfolds,
                'num_ind_per_fold': [(len(x),len(y),len(z)) for x,y,z in cv_indices],
                'num_class': num_class
                }
        with open(self.config.metadata, 'w') as f:
            json.dump(self.meta, f)

        # Transpose bed file to get X matrix
        print('Transposing plink file...')
        # Produces transposed BED file
        assert Xt_plink.transpose(self.config.plinktrans), 'Transpose failed'

        # Open transposed file and iterate over records
        X_plink = plinkfile.open(self.config.plinktrans)
        assert not X_plink.one_locus_per_row(), 'PLINK file should be transposed'
        assert len(labels) == num_ind, 'Number of labels is not equal to num individuals'

        if not disk:
            X = np.zeros((num_ind, num_snps), np.int8)
            Y = np.zeros((num_ind, ))
        else:
            X = bc.zeros((num_ind, num_snps), np.int8, rootdir=self.config.x, mode='w')
            Y = bc.zeros((num_ind, ), rootdir=self.config.y, mode='w')

        folds = {'train': [], 'valid': [], 'test': [],
                 'ytrain': [], 'yvalid': [], 'ytest': []}

        for fold, (train_idx, valid_idx, test_idx) in zip(range(nfolds), cv_indices):
            folds['train'].append(bc.zeros((len(train_idx), num_snps), np.int8, mode='w',
                                  rootdir=self.config.x_fold.format(k=fold+1, set='train')))
            folds['valid'].append(bc.zeros((len(valid_idx), num_snps), np.int8, mode='w',
                                  rootdir=self.config.x_fold.format(k=fold+1, set='valid')))
            folds['test'].append(bc.zeros((len(test_idx), num_snps), np.int8, mode='w',
                                  rootdir=self.config.x_fold.format(k=fold+1, set='test')))

            folds['ytrain'].append(bc.zeros((len(train_idx),), mode='w',
                                  rootdir=self.config.y_fold.format(k=fold+1, set='train')))
            folds['yvalid'].append(bc.zeros((len(valid_idx),), mode='w',
                                  rootdir=self.config.y_fold.format(k=fold+1, set='valid')))
            folds['ytest'].append(bc.zeros((len(test_idx),), mode='w',
                                  rootdir=self.config.y_fold.format(k=fold+1, set='test')))

        # Write k-fold train/valid/test splits
        for i, (row, label) in enumerate(zip(X_plink, labels)): #iterates over individuals
            for fold, (train_idx, valid_idx, test_idx) in zip(range(nfolds), cv_indices):

                if disk:
                    if i in train_idx:
                        folds['train'][fold][train_idx.index(i)] = list(row)
                        folds['ytrain'][fold][train_idx.index(i)] = label
                    elif i in valid_idx:
                        folds['valid'][fold][valid_idx.index(i)] = list(row)
                        folds['yvalid'][fold][valid_idx.index(i)] = label
                    elif i in test_idx:
                        folds['test'][fold][test_idx.index(i)] = list(row)
                        folds['ytest'][fold][test_idx.index(i)] = label
                    else:
                        raise 'Not valid index'

            X[i, :] = list(row)
            Y[i] = label

            if i % 100 == 0:
                print('Writing genotypes... {:.2f}% completed'.format((i/num_ind)*100), end='\r')
                sys.stdout.flush()

        if not disk:
            X = bc.carray(X, rootdir=self.config.x, mode='w')
            Y = bc.carray(Y, rootdir=self.config.y, mode='w')

            for fold, (train_idx, valid_idx, test_idx) in zip(range(nfolds), cv_indices):
                folds['train'][fold][:]  = X[train_idx]
                folds['ytrain'][fold][:] = Y[train_idx]
                folds['valid'][fold][:]  = X[valid_idx]
                folds['yvalid'][fold][:] = Y[valid_idx]
                folds['test'][fold][:]   = X[test_idx]
                folds['ytest'][fold][:]  = Y[test_idx]

        X.flush()
        for fold in range(nfolds):
            folds['train'][fold].flush()
            folds['ytrain'][fold].flush()
            folds['valid'][fold].flush()
            folds['yvalid'][fold].flush()
            folds['test'][fold].flush()
            folds['ytest'][fold].flush()
        print('\nDone')
        del(X)
        del(folds)

        if not disk:
            Xt = np.zeros([num_snps, num_ind], np.int8)
        else:
            Xt = bc.zeros([num_snps, num_ind], np.int8, rootdir=self.config.x_t, mode='w')

        for i, row in enumerate(Xt_plink): #iterates over snps
            Xt[i,:] = row
            if i % 1000 == 0:
                print('Writing X transpose matrix... {:.2f}% completed'.format((i/num_snps)*100), end='\r')
                sys.stdout.flush()

        if not disk:
            Xt = bc.carray(Xt, rootdir=self.config.x_t, mode='w')

        print('\nDone')
        Xt.flush()


    def read_metadata(self):
        self.meta = json.load(open(self.config.metadata))
        return self.meta


    def get_fold_file(self, fold, set):

        assert set in ('train', 'valid', 'test'), 'Invalid set'

        meta = self.read_metadata()
        nfolds = meta['nfolds']
        xf = self.config.x_fold
        yf = self.config.y_fold

        return xf.format(k=fold, set=set), yf.format(k=fold, set=set)


    def get_generator_from_fold(self, fold, set, batch_size, num_process=4):

        xfile, yfile = self.get_fold_file(fold, set)
        seq = BcolzSequence(xfile, yfile, batch_size)
        q = OrderedEnqueuer(seq, use_multiprocessing = True, scheduling = 'shuffle')
        q.start(num_process)
        return q.get()


    def read_transpose(self):
        xt = bc.open(self.config.x_t)
        return xt


def preprocess(args):
    dt = Dataset(args.prefix)

    dt.from_plink(args.prefix, args.pheno,
            nfolds=args.kfold,
            phenotype_idcol=args.phenoidcol,
            phenotype_col=args.phenocol,
            phenotype_categorical=args.categorical,
            num_class=args.numclasses,
            disk=args.disk)
