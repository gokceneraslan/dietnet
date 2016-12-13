#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, csv, argparse

from plinkio import plinkfile #install via pip: pip install plinkio
import numpy as np
import pandas as pd

def plink2numpy(prefix, phenotype_file,
                phenotype_idcol=0,
                phenotype_col=1,
                phenotype_categorical=True):

    # Read plink files
    p = plinkfile.open(prefix)
    num_snps = len(p.get_loci())
    num_ind = len(p.get_samples())

    Xt = np.zeros([num_snps, num_ind], np.int8)
    for i, row in enumerate(p):
        Xt[i,:] = row
        if i % 1500 == 0:
            print('{:.2f}% complete'.format((i/num_snps)*100))

    # Save X and X^T as numpy arrays
    np.save('{}_x_transpose.npy'.format(prefix), Xt)
    np.save('{}_x.npy'.format(prefix), np.transpose(Xt))

    # Read sample ids from the .fam file
    fam_ids = np.array([s.iid for s in p.get_samples()])
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

        # Save class labels
        np.save('{}_y.npy'.format(prefix), pheno_list_cat.codes.astype(np.uint8))
    else:
        np.save('{}_y.npy'.format(prefix), pheno_list.as_matrix())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert plink files to '
                                                  'numpy arrays.')
    parser.add_argument('-f', '--prefix', type=str, help='PLINK prefix',
                        required=True)
    parser.add_argument('-p', '--pheno', type=str,
            help='TSV/CSV formatted phenotype file', required=True)
    parser.add_argument('-i', '--phenoidcol', type=int,
            help='0-based column index of sample ids in phenotype file (default=0)',
            default=0)
    parser.add_argument('-j', '--phenocol', type=int,
            help='0-based column index of phenotypes in phenotype file (default=1)',
            default=1)
    parser.add_argument('-c', '--categorical', type=bool,
            help='Phenotype is categorical (default=True)', default=True)

    args = parser.parse_args()
    plink2numpy(args.prefix, args.pheno,
            phenotype_idcol=args.phenoidcol,
            phenotype_col=args.phenocol,
            phenotype_categorical=args.categorical)
