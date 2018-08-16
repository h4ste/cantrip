---
layout: default
---

# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

# Dependencies
- Python >= 3.6
- TensorFlow >= 1.9

# Installation
To install, run
```bash
$ python setup.py
```

# Usage
CANTRIP is evoked at the module level, with scripts for training and evaluating CANTRIP itself, or SVM/baseline models.

### Training and Evaluating CANTRIP
 To train and evaluate CANTRIP, you need to pass `cantrip` a path to a chronology file and a vocabulary file.
```bash
$ python -m src.scripts.cantrip [-h] --chronology-path CHRONOLOGY_PATH --vocabulary-path VOCABULARY_PATH 
                  [--max-chron-len L] [--max-snapshot-size N]
                  [--vocabulary-size V] [--discrete-deltas]
                  [--dropout DROPOUT]
                  [--observation-embedding-size OBSERVATION_EMBEDDING_SIZE]
                  [--snapshot-embedding-size SNAPSHOT_EMBEDDING_SIZE]
                  [--snapshot-encoder {RNN,CNN,BAG,DAN,DENSE}]
                  [--snapshot-rnn-num-hidden SNAPSHOT_RNN_NUM_HIDDEN [SNAPSHOT_RNN_NUM_HIDDEN ...]]
                  [--snapshot-rnn-cell-type {LSTM,LSTM-LN,GRU,GRU-LN,RAN,RAN-LN} [{LSTM,LSTM-LN,GRU,GRU-LN,RAN,RAN-LN} ...]]
                  [--snapshot-cnn-windows [SNAPSHOT_CNN_WINDOWS]]
                  [--snapshot-cnn-kernels SNAPSHOT_CNN_KERNELS]
                  [--snapshot-dan-num-hidden-avg SNAPSHOT_DAN_NUM_HIDDEN_AVG [SNAPSHOT_DAN_NUM_HIDDEN_AVG ...]]
                  [--snapshot-dan-num-hidden-obs SNAPSHOT_DAN_NUM_HIDDEN_OBS [SNAPSHOT_DAN_NUM_HIDDEN_OBS ...]]
                  [--rnn-num-hidden RNN_NUM_HIDDEN [RNN_NUM_HIDDEN ...]]
                  [--rnn-cell-type {SRAN,RAN,RAN-LN,LSTM,LSTM-LN,GRU,GRU-LN}]
                  [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS]
                  [--tdt-ratio TDT_RATIO] [--early-term]
                  [--summary-dir SUMMARY_DIR]
                  [--checkpoint-dir CHECKPOINT_DIR] [--clear] [--debug DEBUG]
                  [--print-performance] [--print-latex-results]
```
with optional arguments:
```bash
  -h, --help            show this help message and exit
  --chronology-path CHRONOLOGY_PATH
                        path to cohort chronologies
  --vocabulary-path VOCABULARY_PATH
                        path to cohort vocabulary
  --max-chron-len L     maximum number of snapshots per chronology
  --max-snapshot-size N
                        maximum number of observations to consider per
                        snapshot
  --vocabulary-size V   maximum vocabulary size, only the top V occurring
                        terms will be used
  --discrete-deltas     rather than encoding deltas as tanh(log(delta)),
                        discretize them into buckets: > 1 day, > 2 days, > 1
                        week, etc.(we don't have enough data for this be
                        useful)
  --dropout DROPOUT     dropout used for all dropout layers (including the
                        vocabulary)
  --observation-embedding-size OBSERVATION_EMBEDDING_SIZE
                        dimensions of observation embedding vectors
  --snapshot-embedding-size SNAPSHOT_EMBEDDING_SIZE
                        dimensions of clinical snapshot encoding vectors
  --snapshot-encoder {RNN,CNN,BAG,DAN,DENSE}
                        type of clinical snapshot encoder to use
  --rnn-num-hidden RNN_NUM_HIDDEN [RNN_NUM_HIDDEN ...]
                        size of hidden layer(s) used for inferring the
                        clinical picture; multiple arguments result in
                        multiple hidden layers
  --rnn-cell-type {SRAN,RAN,RAN-LN,LSTM,LSTM-LN,GRU,GRU-LN}
                        type of RNN cell to use for inferring the clinical
                        picture
  --batch-size BATCH_SIZE
                        batch size
  --num-epochs NUM_EPOCHS
                        number of training epochs
  --tdt-ratio TDT_RATIO
                        training:development:testing ratio
  --early-term          stop when F1 on dev set decreases; this is pretty much
                        always a bad idea
  --summary-dir SUMMARY_DIR
  --checkpoint-dir CHECKPOINT_DIR
  --clear               remove previous summary/checkpoints before starting
                        this run
  --debug DEBUG         hostname:port of TensorBoard debug server
  --print-performance
  --print-latex-results
```
And optional snapshot encoder arguments:
```
Snapshot Encoder: RNN Flags:
  --snapshot-rnn-num-hidden SNAPSHOT_RNN_NUM_HIDDEN [SNAPSHOT_RNN_NUM_HIDDEN ...]
                        size of hidden layer(s) used for combining clinical
                        obserations to produce the clinical snapshot encoding;
                        multiple arguments result in multiple hidden layers
  --snapshot-rnn-cell-type {LSTM,LSTM-LN,GRU,GRU-LN,RAN,RAN-LN} [{LSTM,LSTM-LN,GRU,GRU-LN,RAN,RAN-LN} ...]
                        size of hidden layer(s) used for combining clinical
                        observations to produce the clinical snapshot
                        encoding; multiple arguments result in multiple hidden
                        layers

Snapshot Encoder: CNN Flags:
  --snapshot-cnn-windows [SNAPSHOT_CNN_WINDOWS]
                        length of convolution window(s) for CNN-based snapshot
                        encoder; multiple arguments results in multiple
                        convolution windows
  --snapshot-cnn-kernels SNAPSHOT_CNN_KERNELS
                        number of filters used in CNN
  --snapshot-dan-num-hidden-avg SNAPSHOT_DAN_NUM_HIDDEN_AVG [SNAPSHOT_DAN_NUM_HIDDEN_AVG ...]
                        number of hidden units to use when refining the DAN
                        average layer; multiple arguments results in multiple
                        dense layers
  --snapshot-dan-num-hidden-obs SNAPSHOT_DAN_NUM_HIDDEN_OBS [SNAPSHOT_DAN_NUM_HIDDEN_OBS ...]
                        number of hidden units to use when refining clinical
                        observation embeddings; multiple arguments results in
                        multiple dense layers
```

### Training and Evaluating Support Vector Machines
Training and evaluating SVM baselines can be accomplished by:
```bash
$ python -m src.scripts.svm [-h] --chronology-path CHRONOLOGY_PATH --vocabulary-path VOCABULARY_PATH 
              [--tdt-ratio TDT_RATIO] [--vocabulary-size V]
              [--final-only] [--discrete-deltas] [--kernel KERNEL]
```
with optional arguments
```
  -h, --help            show this help message and exit
  --chronology-path CHRONOLOGY_PATH
                        path to cohort chronologies
  --vocabulary-path VOCABULARY_PATH
                        path to cohort vocabulary
  --tdt-ratio TDT_RATIO
                        training:development:testing ratio
  --vocabulary-size V   maximum vocabulary size, only the top V occurring
                        terms will be used
  --final-only          only consider the final prediction in each chronology
  --discrete-deltas     rather than encoding deltas as tanh(log(delta)),
                        discretize them into buckets: > 1 day, > 2 days, > 1
                        week, etc.
  --kernel KERNEL       SVM kernel to evaluate
```

### Training and Evaluating Misc. Baselines
Training and evaluating miscellanous SciKit: Learn baselines is done through:
```bash
python -m src.scripts.baselines.py --chronology-path CHRONOLOGY_PATH --vocabulary-path VOCABULARY_PATH 
                    [--tdt-ratio TDT_RATIO]
                    [--vocabulary-size V] [--final-only] [--discrete-deltas]
```
with the following optional parameters:
```
  -h, --help            show this help message and exit
  --chronology-path CHRONOLOGY_PATH
                        path to cohort chronologies
  --vocabulary-path VOCABULARY_PATH
                        path to cohort vocabulary
  --tdt-ratio TDT_RATIO
                        training:development:testing ratio
  --vocabulary-size V   maximum vocabulary size, only the top V occurring
                        terms will be used
  --final-only          only consider the final prediction in each chronology
  --discrete-deltas     rather than encoding deltas as tanh(log(delta)),
                        discretize them into buckets: > 1 day, > 2 days, > 1
                        week, etc.
```
 
# Documentation
[Sphinx](http://www.sphinx-doc.org/en/master/)-based Python documentation is available [here](https://h4ste.github.io/cantrip/sphinx/html/).

# Structure
- [src/data](sphinx/html/src.data.html) Classes and utilities for loading clinical chronologies (and observation vocabularies from the disk)
- [src/models](sphinx/html/src.models.html) TensorFlow implementation of CANTRIP, including:
    - [src/models/encoder](sphinx/html/src.models.encoder.html) TensorFlow implementation of clinical snapshot encoders
    - [src/models/rnn_cell](sphinx/html/src.models.rnn_cell.html) TensorFlow implementation of [Recurrent Additive Networks (RANs)](https://arxiv.org/abs/1705.07393) and Batch-normalized Gated Recurrent Units
- [src/scripts](sphinx/html/src.scripts.html) Executable scripts for running and evaluating CANTRIP as well as SVM and other baseline systems on pneumonia risk predcition
