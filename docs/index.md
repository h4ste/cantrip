---
layout: default
---

# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

# Dependencies
- Python >= 3.6
- TensorFlow >= 1.9
- SciPy
- tqdm
- tabulate

# Installation
To install, run 
  $ python setup.py
  
# Documentation
[Sphinx](http://www.sphinx-doc.org/en/master/)-based Python documentation is available [here](https://h4ste.github.io/cantrip/sphinx/html/).

# Structure
- [src/data](sphinx/html/src.data.html) Classes and utilities for loading clinical chronologies (and observation vocabularies from the disk)
- [src/models](sphinx/html/src.models.html) TensorFlow implemenation of CANTRIP, including:
    - [src/models/encoder](sphinx/html/src.models.encoder.html) TensorFlow implementation of clinical snapshot encoders
    - [src/models/rnn_cell](sphinx/html/src.models.rnn_cell.html) TensorFlow implementation of [Recurrent Additive Networks (RANs)](https://arxiv.org/abs/1705.07393) and Batch-normalized Gated Recurrent Units
- [src/scripts](sphinx/html/src.scripts.html) Executable scripts for running and evaluating CANTRIP as well as SVM and other baseline systems on pneumonia risk predcition
