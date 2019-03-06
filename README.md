# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

Please check the [website](https://h4ste.github.io/cantrip) for details.

# Dependencies
- Python >= 3.6
- TensorFlow >= 1.9
- numpy >= 1.13.3
- scikit-learn >= 0.19.0
- scipy >= 0.13.3

# Installation
First, install the required dependencies:
```bash
$ pip -r requirements.txt
```
Then, install TensorFlow with or without gpu support:

| CPU only | GPU Enabled |
|---------:|-----------:|
| `$ pip install tensorflow>=1.9.0` | `$ pip install tensorflow-gpu>=1.9.0` |

Optionally install any of the below optional dependencies:

| Dependency | Purpose |
|-----------:|--------:|
| tqdm       | pretty console progress logging |
| tabulate   | printing LaTeX style results tables |

---

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
                        week, etc.
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
```bash
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
```bash
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
Training and evaluating miscellaneous SciKit: Learn baselines is done through:
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

---

## Data format
The `cantrip` script load chronology and vocabulary files. Chronology and vocabulary files are assumed to follow specific formats.

### Chronology format
The format of this chronology file is assumed to be as follows:

    [external_patient_id]\t[chronology]

where each ``[chronology]`` is encoded as as sequence of snapshots, separated by tabs:

    [[snapshot]\t...]

and each ``[snapshot]`` is encoded as:

    [delta] [label] [observation IDs..]

Note that snapshots are delimited by *spaces*, label must be 'true' or 'false', delta is represented in seconds
since previous chronology, and observation IDs should be the IDs associated with the observation in the given
vocabulary file.

For example, the line:

    11100004a   0 false 1104 1105 2300 25001    86400 false 1104 2300   172800 true 1104 2300 3500

would indicate that patient with external ID '11100004a' had a chronology including 3 snapshots
where:

* the first snapshot was negative for pneumonia, had a delta of 0, and contained only three clinical
  observations: those associated with vocabulary terms 1104, 1105, 2300, and 25001;
* the second snapshot was negative for pneumonia, had a delta of 86400 seconds (1 day), and included only
  two clinical observations: 1104 and 2300
* the third snapshot was positive for pneumonia, had a delta of 172800 seconds (2 days), and included only
  three clinical observations: 1104, 2300, and 3500
  
### Vocabulary format
The vocabulary file is assumed to be formatted as follows:

    [observation]\t[frequency]

where the line number indicates the ID of the observation in the chronology (e.g., 1104), ``[observation]`` is a
human-readable string describing the observation, and ``[frequency]`` is the frequency of that observation in
the dataset (this value is only important if specifying a max_vocabulary_size as terms will be sorted in
descending frequency before the cut-off is made)

Note: as described in the AMIA paper , chronologies are truncated to terminate at the first positive label.
Chronologies in which the first snapshot is positive or in which no snapshot is positive are discarded.

---

# Python Documentation
Documentation on CANTRIP is provided at [here](https://h4ste.github.io/cantrip). 
[Sphinx](http://www.sphinx-doc.org/en/master/)-based Python documentation is available [here](https://h4ste.github.io/cantrip/sphinx/html/).

## Structure
- [src/data](src/data) Classes and utilities for loading clinical chronologies (and observation vocabularies from the disk); Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.data.html)
- [src/models](src/models) TensorFlow implementation of CANTRIP; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.html)
    - [src/models/encoder](snapshot_encoding.py) TensorFlow implementation of clinical snapshot encoders; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.encoder.html)
    - [src/models/rnn_cell](rnn_cell) TensorFlow implementation of [Recurrent Additive Networks (RANs)](https://arxiv.org/abs/1705.07393) and Batch-normalized Gated Recurrent Units; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.models.rnn_cell.html)
- [src/scripts](src/scripts) Executable scripts for running and evaluating CANTRIP as well as SVM and other baseline systems on pneumonia risk prediction; Python documentation is provided [here](https://h4ste.github.io/cantrip/sphinx/html/src.scripts.html)
