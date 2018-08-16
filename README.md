# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

Please check the [website](https://h4ste.github.io/cantrip) for details.

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
Documentation on CANTRIP is provided at [here](https://h4ste.github.io/cantrip). 
[Sphinx](http://www.sphinx-doc.org/en/master/)-based Python documentation is available [here](https://h4ste.github.io/cantrip/sphinx/html/).

# Structure
- [src/data](src/data) Classes and utilities for loading clinical chronologies (and observation vocabularies from the disk); Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.data.html)
- [src/models](src/models) TensorFlow implemenation of CANTRIP; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.html)
    - [src/models/encoder](src/models/encoder/snapshot_encoder.py) TensorFlow implementation of clinical snapshot encoders; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.encoder.html)
    - [src/models/rnn_cell](src/models/encoder/rnn_cell.py) TensorFlow implementation of [Recurrent Additive Networks (RANs)](https://arxiv.org/abs/1705.07393) and Batch-normalized Gated Recurrent Units; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.rnn_cell.html)
- [src/scripts](src/models/scripts.py) Executable scripts for running and evaluating CANTRIP as well as SVM and other baseline systems on pneumonia risk predcition; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.scripts.html)
