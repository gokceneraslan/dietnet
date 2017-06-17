from setuptools import setup
from subprocess import check_output, CalledProcessError

try:
    num_gpus = len(check_output(['nvidia-smi', '--query-gpu=gpu_name',
                                 '--format=csv']).decode().strip().split('\n'))
    tf = 'tensorflow-gpu>1.1' if num_gpus > 1 else 'tensorflow'
except (CalledProcessError, FileNotFoundError):
    tf = 'tensorflow>=1.1'


setup(
    name='dietnet',
    version='0.1',
    description='A program to fit MLPs to high dimensional data',
    author='Gokcen Eraslan',
    author_email="goekcen.eraslan@helmholtz-muenchen.de",
    packages=['dietnet'],
    install_requires=[tf,
                      'numpy>=1.7',
                      'six>=1.10.0',
                      'bcolz',
                      'plinkio', #for preprocessing
                      'pandas', #for preprocessing
                      'scikit-learn'], #only for k-fold and train/valid/test split
    url='https://github.com/gokceneraslan/dietnet',
    entry_points={
        'console_scripts': [
            'dietnet = dietnet.__main__:main'
    ]},
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Bio-Informatics',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5'],
)
