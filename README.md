# Diet Networks: Thin Parameters for Fat Genomics

Unofficial implementation of [diet networks](http://openreview.net/forum?id=Sk-oDY9ge) in TensorFlow.

![tb](tensorboard.png)

## Requirements:

- Python 2.7 or 3.5
- [TensorFlow](https://www.tensorflow.org) >= 1.1.0. See [installation instructions](https://www.tensorflow.org/install/install_linux#InstallingNativePip).
- numpy
- [plinkio](https://pypi.python.org/pypi/plinkio) (only for preprocessing) Install via pip e.g. `pip install plinkio`
- [plink2](https://www.cog-genomics.org/plink2) (only for preprocessing) This can be easily installed via [Bioconda](http://bioconda.github.io) e.g. `conda install -c bioconda plink2`
- [pandas](http://pandas.pydata.org/) (only for preprocessing)


## Usage:

- Install the requirements above. Use plink2numpy to preprocess PLINK files and run dietnet script with `--prefix` option.
- To reproduce 1000G results, install dietnet e.g. `python setup.py install` and run `make` command in `1000G` folder. (~700MB file will be downloaded) Finally, run `./dietnet train 1000G/genotypes`

## TODO:

- ~~Dropout~~
- ~~Bias terms for We and Wd~~
- ~~Summary ops and tensorboard screenshots. also misclass. err~~
- K-fold CV
- Make train and predict subcommands e.g. add placeholders
- Other embeddings: random projection, histogram
- SNP2vec
