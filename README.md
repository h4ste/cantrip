# reCurrent Additive Network for Temporal RIsk Prediction (CANTRIP)
A TensorFlow model for predicting temporal (disease) risk based on retrospective analysis of longitudinal clinical notes.

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
|:---------|:------------|
| `$ pip install tensorflow>=1.9.0` | `$ pip install tensorflow-gpu>=1.9.0` |

Optionally install any of the below optional dependencies:

| Dependency | Purpose |
|-----------:|:--------|
| tqdm       | pretty console progress logging |
| tabulate   | printing LaTeX style results tables |

---

### Training and Evaluating PRONTO
 To train and evaluate PRONTO, you need to pass `pronto` a path to a chronology file and a vocabulary file.
```bash
export CHRONOLOGY_FILE=/path/to/chronology.tsv
export VOCAB_FILE=/path/to/vocabulary.tsv
export OUTPUT_DIR=/path/to/output/directory

python run_cantrip.py \
      --chrono_file=$CHRONOLOGY_FILE  \
      --vocab_file=$VOCAB_FILE \
      --output_dir=$OUTPUT_DIR \
      --max_snapshot_size 256 \
      --batch_size=32 \
      --dropout=0.00 \
      --vocab_dropout=0.50 \
      --num_epochs 20 \
      --print_performance
  ```

Required command-line arguments:

| Flag | Description |
|-----:|------------:|
| --chrono_file | The chronology file for the cohort. |
| --output_dir | The output directory where model checkpoints and summaries will be written. |
| --vocab_file | The vocabulary file that chronologies were created with. |  |  |$ python run_pronto [-h] --chronology-path CHRONOLOGY_PATH --vocabulary-path VOCABULARY_PATH 
  
Additional command-line arguments:

| Flag | Description | Default | Range |
|:-----|:-----------|--------:|:-----|
| --batch_size | The batch size. | 40 | a positive integer |
| --tdt_ratio | The training:development:testing ratio. | 8:1:1 |  |
| --dropout | Dropout used for all dropout layers (except vocabulary) | 0.7 | a non-negative number |
| --vocab_dropout | Dropout used for vocabulary-level dropout (supersedes --dropout) | 0.7 | a non-negative number |
| --learning_rate | The initial learning rate. | 0.0001 | number >= 1e-45 |
| --max_chrono_length | The maximum number of snapshots per chronology. | 7 | a positive integer |
| --max_snapshot_size | The maximum number of observations to consider per snapshot. | 200 | a positive integer |
| --max_vocab_size | The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be used to encode clinical snapshots. Any remaining observations will be ignored. | 50000 | a positive integer |
| --num_epochs | The number of training epochs. | 30 | a positive integer |
| --observation_embedding_size | The dimensions of observation embedding vectors. | 200 | a positive integer |
| --rnn_num_hidden | The size of hidden layer(s) used for inferring the clinical picture; multiple arguments result in multiple hidden layers; repeat this option to specify a list of values | [100] | one or more positive integers |
| --[no]snap_rnn_layer_norm | Enable layer normalization in the RNN used for snapshot encoding. | false |  |
| --snapshot_cnn_kernels | The number of filters used in CNN | 1000 | a positive integer |
| --snapshot_cnn_windows | The length of convolution window(s) for CNN-based snapshot encoder; multiple arguments results in multiple convolution windows; repeat this option to specify a list of values | [3, 4, 5] | one or more positive integers |
| --snapshot_dan_num_hidden_avg | The number of hidden units to use when refining the DAN average layer; multiple arguments results in multiple dense layers; repeat this option to specify a list of values | [200, 200]  | one or more positive integers |    
| --snapshot_dan_num_hidden_obs | The number of hidden units to use when refining clinical observation embeddings; multiple arguments results in multiple dense layers; repeat this option to specify a list of values | [200, 200]  | one or more positive integers |
| --snapshot_embedding_size | The dimensions of clinical snapshot encoding vectors. | 200 | a positive integer |
| --snapshot_encoder | The type of clinical snapshot encoder to use | DAN | RNN, CNN, SPARSE, DAN, DENSE |
| --snapshot_rnn_cell_type | The type of RNN cell to use when encoding snapshots | RAN | LSTM, GRU, RAN |
| --snapshot_rnn_num_hidden | The size of hidden layer(s) used for combining clinical observations to produce the clinical snapshot encoding; multiple arguments result in multiple hidden layers; repeat this option to specify a list of values | [200]  | one or more positive integers |

Convenience options:

| Flag | Description |
|:-----|:-----------|
| --[no]early_term | Stop when F2 on dev set decreases; this is pretty much always a bad idea. |
| --debug: The hostname:port of TensorBoard debug server; debugging will be enabled if this flag is specified. |
| --[no]clear_prev | Whether to remove previous summary/checkpoints before starting this run. |
| --[no]print_performance | Whether to print performance to the console. |
| --[no]save_latex_results | Whether to save performance in a LaTeX-friendly table. |
| --[no]save_tabbed_results | Whether to save performance in a tab-separated table. |

Super secret options:

| Flag | Description | Default | Values |
|:-----|:------------|--------:|:-------|
| --optimizer | The type of optimizer to use when training PRONTO. | PRONTO | PRONTO, BERT | 
| --rnn_cell_type | The type of RNN cell to use for inferring the clinical picture. | RAN | RAN, LRAN, GRU, LSTM |
| --[no]rnn_layer_norm | Whether to use layer normalization in RNN used for inferring the clinical picture. | true | |
| --[no]use_discrete_deltas | Rather than encoding deltas as tanh(log(delta)), they will be discretized into buckets: > 1 day, > 2 days, > 1 week, etc. | false | |

### Training and Evaluating Support Vector Machines
Training and evaluating SVMs can be accomplished by:
```bash
$ python -m run_svm.py [-h] \
    --chronology-path $CHRONOLOGY_FILE \
    --vocabulary-path $VOCAB_FILE \
    --final-only 
```

Required command-line arguments:

| Flag | Description |
|-----:|------------:|
| --chronology-path | The chronology file for the cohort. |
| --vocabulary-path | The vocabulary file that chronologies were created with. | 


  
Additional command-line arguments:

| Flag | Description | Default | Range |
|:-----|:------------|--------:|:------|
| --tdt-ratio | The training:development:testing ratio. | 8:1:1 |  |
| --vocabulary-size | The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be used to encode clinical snapshots. Any remaining observations will be ignored. | 50000 | a positive integer |
| --final_only | If set, will train using only the final snapshot in each chronology | False | |
| --kernel | The type of kernel to evaluate | linear | linear, polynomial, rbf, sigmoid |
| --[no]use_discrete_deltas | Rather than encoding deltas as tanh(log(delta)), they will be discretized into buckets: > 1 day, > 2 days, > 1 week, etc. | false | |

### Training and Evaluating Support Vector Machines
Training and evaluating baselines can be accomplished by:
```bash
$ python -m run_baselines.py [-h] \
    --chronology-path $CHRONOLOGY_FILE \
    --vocabulary-path $VOCAB_FILE \
    --final-only 
```

Required command-line arguments:

| Flag | Description |
|-----:|------------:|
| --chronology-path | The chronology file for the cohort. |
| --vocabulary-path | The vocabulary file that chronologies were created with. | 

Additional command-line arguments:

| Flag | Description | Default | Range |
|:-----|:-----------|---------:|:------|
| --tdt-ratio | The training:development:testing ratio. | 8:1:1 |  |
| --vocabulary-size | The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be used to encode clinical snapshots. Any remaining observations will be ignored. | 50000 | a positive integer |
| --final_only | If set, will train using only the final snapshot in each chronology | False | |
| --[no]use_discrete_deltas | Rather than encoding deltas as tanh(log(delta)), they will be discretized into buckets: > 1 day, > 2 days, > 1 week, etc. | false | |

---

## Data format
The `pronto` script load chronology and vocabulary files. Chronology and vocabulary files are assumed to follow specific formats.

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
