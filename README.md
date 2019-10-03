# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

# Dependencies
- python >= 3.6
- tensorflow >= 1.14
- absl-py >= 0.7.0
- python-dateutil >= 2.6.1
- numpy >= 1.16.0
- scikit-learn >= 0.19.0
- scipy >= 0.13.3

# Installation
First, install the required dependencies:
```bash
$ pip -r requirements.txt
```
Then, install TensorFlow with or without gpu support:

| CPU only | GPU Enabled |
|:---------|:------------|
| `$ pip install tensorflow>=1.14.0` | `$ pip install tensorflow-gpu>=1.14.0` |

Optionally install any of the below optional dependencies:

| Dependency | Purpose |
|-----------:|:--------|
| tqdm       | pretty console progress logging |
| tabulate   | printing LaTeX style results tables |

---

## Usage
```bash
       USAGE: run_experiment.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR [flags]

required flags:
  --data_dir: Location of folder containing features, labels, & start times for training, development, and testing cohorts
  --output_dir: The output directory where model checkpoints and summaries will be written.
optional flags:
  --augment_negatives: Augment negative examples by randomly truncating the given percent of positive examples to end early
    (default: '0.0')
    (a number in the range [0.0, 1.0])
  --batch_size: The batch size.
    (default: '40')
    (a positive integer)
  --[no]clear_prev: Whether to remove previous summary/checkpoints before starting this run.
    (default: 'false')
  --correct_imbalance: <none|weighted|downsample|upsample>: How to correct class imbalance in the training set.
    (default: 'weighted')
  --debug: The hostname:port of TensorBoard debug server; debugging will be enabled if this flag is specified.
  --delta_combine: <concat|add>: How to combine deltas and observation embeddings
    (default: 'concat')
  --delta_enc: <logsig|logtanh|discrete|raw|sinusoid>: Method for encoding elapsed time.
    (default: 'logsig')
  --[no]do_predict: Whether to run predictions on test data.
    (default: 'false')
  --[no]do_test: Whether to evaluate on test data.
    (default: 'false')
  --[no]do_train: Whether to train on training data.
    (default: 'false')
  --dropout: Dropout used for all dropout layers (except vocabulary)
    (default: '0.0')
    (a non-negative number)
  --[no]embed_delta: Embed delta vectors to same size as observation embeddings (used with --delta_combine=add)
    (default: 'false')
  --learning_rate: The initial learning rate.
    (default: '0.0001')
    (number >= 1e-45)
  --max_chrono_length: The maximum number of snapshots per chronology (chronologies will be truncated from the end).
    (default: '7')
    (a positive integer)
  --max_pred_window: The maximum time (in hours) between the last snapshot and the label (chronologies whose last snapshot occurs more than this many hours before the label will be discarded)
    (default: '76')
    (a non-negative integer)
  --max_snapshot_delay: The maximum number of hours between admission and first snapshot(chronologies whose first snapshot occurs after this value will be discarded)
    (default: '96')
    (a non-negative integer)
  --max_snapshot_size: The maximum number of observations to consider per snapshot (observations will be truncated as read).
    (default: '500')
    (a positive integer)
  --max_to_keep: The number of model checkpoints to save.
    (default: '3')
    (an integer)
  --max_vocab_size: The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be used to encode clinical snapshots. Any remaining observations will be discarded.
    (default: '50000')
    (a positive integer)
  --min_chrono_length: The minimum number of snapshots per chronology (chronologies with fewer than the given length will be discarded).
    (default: '3')
    (integer >= 2)
  --min_pred_window: The minimum time (in hours) between the last snapshot and the label (chronologies will be truncated to end at least <min_pred_window> hours before the first label)
    (default: '24')
    (a non-negative integer)
  --min_snapshot_size: The minimum number of observations to consider per snapshot (snapshots with fewer observations will be discarded).
    (default: '10')
    (a non-negative integer)
  --min_start_window: The minimum length of time (in hours) between the start time and first label (chronologies for which the label occurs within this window will be discarded)
    (default: '48')
    (a non-negative integer)
  --model: <CANTRIP|LR|SVM>: model to train and/or evaluate
    (default: 'CANTRIP')
  --num_epochs: The number of training epochs.
    (default: '30')
    (a positive integer)
  --observation_embedding_size: The dimensions of observation embedding vectors.
    (default: '200')
    (a positive integer)
  --observational_dropout: Dropout used for vocabulary-level dropout (supersedes --dropout)
    (default: '0.0')
    (a non-negative number)
  -oan,--only_augmented_negatives: Use ignore negative examples in the train/dev/test data, and evaluate/train on only augmented negative examples (legacy behavior);
    repeat this option to specify a list of values
    (default: '[]')
  --[no]print_performance: Whether to print performance to the console.
    (default: 'false')
  --[no]restore_cohorts: Restore cohorts from .pickle files (if changing parameters for chronologies this should be set to False)
    (default: 'true')
  --rnn_cell_type: <RAN|GRU|LSTM|VHRAN|RHN>: The type of RNN cell to use for inferring the clinical picture.
    (default: 'RAN')
  --rnn_direction: <forward|bidirectional>: Direction for inferring the clinical picture with an RNN.
    (default: 'bidirectional')
  --rnn_highway_depth: Depth of residual connections in VHRAN/RHN.
    (default: '3')
    (an integer)
  --[no]rnn_layer_norm: Whether to use layer normalization in RNN used for inferring the clinical picture.
    (default: 'true')
  --rnn_num_hidden: The size of hidden layer(s) used for inferring the clinical picture; multiple arguments result in multiple hidden layers.;
    repeat this option to specify a list of values
    (default: '[100]')
    (a positive integer)
  --[no]save_latex_results: Whether to save performance in a LaTeX-friendly table.
    (default: 'false')
  --[no]save_tabbed_results: Whether to save performance in a tab-separated table.
    (default: 'false')
  --sinusoidal_embedding_size: The dimensions of sinusoidal delta encoding vectors.
    (default: '32')
    (a positive integer)
  --[no]snap_rnn_layer_norm: Enable layer normalization in the RNN used for snapshot encoding.
    (default: 'false')
  --snapshot_cnn_kernels: The number of filters used in CNN
    (default: '1000')
    (a positive integer)
  --snapshot_cnn_windows: The length of convolution window(s) for CNN-based snapshot encoder; multiple arguments results in multiple convolution windows.;
    repeat this option to specify a list of values
    (default: '[3, 4, 5]')
    (a positive integer)
  --snapshot_dan_activation: <tanh|gelu|relu|sigmoid>: The type of activation to use for DAN hidden layers
    (default: 'tanh')
  --snapshot_dan_num_hidden_avg: The number of hidden units to use when refining the DAN average layer; multiple arguments results in multiple dense layers.;
    repeat this option to specify a list of values
    (default: '[200, 200]')
    (a positive integer)
  --snapshot_dan_num_hidden_obs: The number of hidden units to use when refining clinical observation embeddings; multiple arguments results in multiple dense layers.;
    repeat this option to specify a list of values
    (default: '[200, 200]')
    (a positive integer)
  --snapshot_embedding_size: The dimensions of clinical snapshot encoding vectors.
    (default: '200')
    (a positive integer)
  --snapshot_encoder: <RNN|CNN|SPARSE|DAN|DENSE|RMLP|VHN>: The type of clinical snapshot encoder to use
    (default: 'DAN')
  --[no]snapshot_l1_reg: Use L1 regularization when encoding clinical snapshots
    (default: 'false')
  --[no]snapshot_l2_reg: Use L2 regularization when encoding clinical snapshots
    (default: 'false')
  --snapshot_rmlp_activation: <tanh|gelu|relu|sigmoid>: The type of activation to use for RMLP hidden layers
    (default: 'gelu')
  --snapshot_rmlp_layers: Number of hidden layers in snapshot RMLP.
    (default: '5')
    (an integer)
  --snapshot_rnn_cell_type: <LSTM|GRU|RAN>: The type of RNN cell to use when encoding snapshots
    (default: 'RAN')
  --snapshot_rnn_num_hidden: The size of hidden layer(s) used for combining clinical observations to produce the clinical snapshot encoding; multiple arguments result in multiple hidden layers;
    repeat this option to specify a list of values
    (default: '[200]')
    (a positive integer)
  --snapshot_vhn_activation: <tanh|gelu|relu|sigmoid>: The type of activation to use for VHN hidden layers
    (default: 'gelu')
  --snapshot_vhn_depth: Depth of residual connections (i.e., number of layers) in snapshot VHN.
    (default: '6')
    (an integer)
  --snapshot_vhn_layers: Number of hidden layers in snapshot VHN.
    (default: '10')
    (an integer)
  --snapshot_vhn_noise: Strength of variational noise.
    (default: '0.5')
    (a number)
  --time_repr: <prev|start|both>: Whether to encode elapsed times as time since previous snapshot, time since chronology start, or both.
    (default: 'prev')
  --[no]use_focal_loss: Use focal loss, so trivial cases have a smaller impact on the model's loss
    (default: 'false')
  --[no]use_l1_reg: Use l1 regularization
    (default: 'false')
  --[no]use_l2_reg: Use l2 regularization
    (default: 'false')
  --[no]use_weight_decay: Use weight decay
    (default: 'true')
```


### Training and Evaluating PRONTO
 To train and evaluate CANTRIP, you need to pass (1) a data directory containing train, development, and testing cohorts
 and (2) an output directory in which to save model checkpoints and/or test/development performance.

 ```bash
export $DATA_DIR=/path/to/inputs
export $OUTPUT_DIR=/path/to/output/directory

python run_experiment.py \
       --data_dir=$DATA_DIR \
       --output_dir=$OUTPUT_DIR \
       --do_train \
       --do_test \
       --do_predict \
       --print_performance \
       --time_repr both \
       --delta_enc sinusoid \
       --delta_combine add \
       --augment_negatives=0 \
       --dropout 0.3 \
       --observational_dropout 0.3 \
       --correct_imbalance weighted \
       --save_tabbed_results \
       --num_epochs 10 \
       --min_pred_window 24 \
       --max_pred_window 96
```

### Training and Evaluating Shallow Baselines
Training and evaluating SVMs can be accomplished by:
```bash
$ python run_experiment.py \                                                                                                                                                                                                                                                          (cantrip)
         --data_dir=$DATA_DIR \
         --output_dir=$OUTPUT_DIR \
         --do_train \
         --do_test \
         --do_predict \
         --print_performance \
         --time_repr both \
         --delta_enc sinusoid \
         --delta_combine add \
         --augment_negatives=0 \
         --correct_imbalance weighted \
         --save_tabbed_results \
         --min_pred_window 24 \
         --max_pred_window 96 \
         --model=SVM
```

Likewise to train and evaluate the logistic regression baseline:
```bash
$ python run_experiment.py \                                                                                                                                                                                                                                                          (cantrip)
         --data_dir=$DATA_DIR \
         --output_dir=$OUTPUT_DIR \
         --do_train \
         --do_test \
         --do_predict \
         --print_performance \
         --time_repr both \
         --delta_enc sinusoid \
         --delta_combine add \
         --augment_negatives=0 \
         --correct_imbalance weighted \
         --save_tabbed_results \
         --min_pred_window 24 \
         --max_pred_window 96 \
         --model=LR
```

## Training and Evaluating bi-LSTM
Training and evaluating the bi-LSTM baseline is accomplished by:
```bash
$ python run_experiment.py \                                                                                                                                                                                                                                                          (cantrip)
         --data_dir=$DATA_DIR \
         --output_dir=$OUTPUT_DIR \
         --do_train \
         --do_test \
         --do_predict \
         --print_performance \
         --time_repr prev \
         --delta_enc raw \
         --delta_combine concat \
         --augment_negatives=0 \
         --correct_imbalance weighted \
         --save_tabbed_results \
         --save_latex_results \
         --num_epochs 30 \
         --min_pred_window 24 \
         --max_pred_window 96 \
         --snapshot_encoder=DENSE \
         --rnn_cell_type=LSTM \
         --rnn_layer_norm=False \
         --rnn_direction=bidirectional
```

## Data format
The `run_experiment.py` script assumes the data directory contains the following CSV files:
- train.chronologies.csv
- train.admittimes.csv
- train.labels.csv
- devel.chronologies.csv
- devel.admittimes.csv
- devel.labels.csv
- test.chronologies.csv
- test.admittimes.csv
- test.labels.csv


### Chronology CSV format
The chronology CSV files is assumed to have and follow the header:

    [subject_id],[hadm_id],[timestamp],[observations]

where ``[observations]`` is encoded as a space separated list of observation IDs (e.g., UMLS CUIs)

### Admission CSV format:
The admission CSV files are assumed to have and follow the header:
    [subject_id],[hadm_id],[timestamp]
where ``[timestamp]`` is the admission time for the associated hospital admission

### Label CSV format:
The label CSV files are assumed to have and follow the header:
    [subject_id],[hadm_id],[timestamp],[label]
where ``[timestamp]`` is the timestamp of the label and ``[label]`` is a zero or one indicating the date-of-event for the disease


Note: the `run_experiment.py` script will handle truncating chronologies and discarding present-on-admission labels.
