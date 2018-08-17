import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='CANTRIP',
    version='0.1.1',
    author='Travis R. Goodwin',
    author_email='travis.goodwin@nih.gov',
    description='TensorFlow implementation of reCurrent Additive Network for Temporal RIsk Prediction',
    license='MIT',
    keywords='tensorflow deep-learning machine-learning predictive-model risk-prediction medical-informatics',
    url='https://h4ste.github.io/src/',
    download_url='https://github.com/h4ste/src/tarball/master',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    install_requires=open(os.path.join(os.path.dirname(__file__), 'requirements.txt')).read(),
    extras_require={
        'tqdm': ['tqdm'],
        'tabulate': ['tabulate'],
        'tf': ['tensorflow>=1.9.0'],
        'tf_gpu': ['tensorflow-gpu>=1.9.0']
    },
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'src = src.scripts.src',
            'baselines = src.scripts.baselines',
            'svm = src.scripts.svm',
        ]
    }
)
