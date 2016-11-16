#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
from collections import OrderedDict

from plinkio import plinkfile #install via pip: pip install plinkio
import numpy as np

prefix = sys.argv[1] if len(sys.argv) > 1 else 'genotypes'

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
np.save('genotypes_x_transpose.npy', Xt)
np.save('genotypes_x.npy', np.transpose(Xt))

# Read sample ids from the .fam file
sample_family_list = OrderedDict.fromkeys([s.iid for s in p.get_samples()])

# Read ethnicity labels from the panel file
affy_samples = {row[0]: row[1] for i, row in
        enumerate(csv.reader(open('affy_samples.20141118.panel'), delimiter='\t')) if i!=0}

assert(len(sample_family_list) == len(affy_samples))
assert(np.all(np.array(sample_family_list.keys()) == np.array(affy_samples.keys())))

for k, v in affy_samples.items():
    sample_family_list[k] = v

# Write population to integer mapping
population_map = sorted(set(sample_family_list.values()))
population_map = OrderedDict([(k,i) for i,k in enumerate(population_map)])

with open('genotypes.population.map', 'w') as f:
   wr = csv.writer(f, delimiter='\t', lineterminator='\n')
   for k, v in population_map.items():
       wr.writerow([k, v])

# Save class labels
np.save('genotypes_y.npy', np.array([population_map[k] for k in sample_family_list.values()]))
