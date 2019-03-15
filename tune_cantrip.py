import os

import numpy as np
import ray
import ray.tune as tune
import tensorflow as tf

import encoding
import modeling
import nio
import optimization
import preprocess
import run_cantrip
import summarization

np.random.seed(1337)
tf.set_random_seed(1337)

flags = tf.flags

FLAGS = flags.FLAGS


def run_tune(argv):
    tune.register_trainable('cantrip', TrainCANTRIP)
    cohort = preprocess.Cohort.from_disk(FLAGS.data_dir, FLAGS.vocab_file, FLAGS.max_vocab_size)
    tdt_cohorts = run_cantrip.make_train_devel_test_split(cohort.patients(), '8:1:1')

    ray.init()  # num_gpus=1)

    cohort_handle = tune.util.pin_in_object_store(cohort)
    tdt_cohorts_handle = tune.util.pin_in_object_store(tdt_cohorts)
    param_spec = {
        'run': 'cantrip',
        'stop': {
            'F2': 0.99,
            'time_total_s': 600,
            'training_iteration': 12,
        },
        'config': {
            'num_epochs': 12,
            'cohort_handle': cohort_handle,
            'tdt_cohorts_handle': tdt_cohorts_handle,
            'output_dir': FLAGS.output_dir,
            # 'learning_rate': tune.sample_from(lambda spec: 10 ** np.random.uniform(-5, -3)),
            'dropout': tune.sample_from(lambda spec: np.random.uniform(0.1, 1.0)),
            'vocab_dropout': tune.sample_from(lambda spec: np.random.uniform(0.1, 1.0)),
            'learning_rate': tune.grid_search([1e-3, 1e-4, 1e-5]),
            # 'dropout': tune.grid_search(np.arange(0.1, 1.0, .1).tolist()),
            # 'vocab_dropout': tune.grid_search(np.arange(0.1, 1.0, .1).tolist()),
            'observation_embedding_size': tune.sample_from(lambda spec: np.random.randint(1, 6) * 100),
            'snapshot_encoder': 'DAN',  # tune.grid_search(['RNN', 'CNN', 'DENSE', 'SPARSE', 'DAN']),
            'rnn_num_layers': tune.grid_search([1, 2, 3]),
            'rnn_num_hidden': tune.sample_from(lambda spec: np.random.randint(1, 6) * 100),
            'rnn_cell_type': 'RAN',  # tune.grid_search(['RAN', 'GRU', 'LSTM']),
            'rnn_layer_norm': tune.grid_search([True, False]),
            'optimizer': tune.grid_search(['BERT', 'CANTRIP']),
            'batch_size': 32,
            'max_snapshot_size': 256,
            'max_seq_len': 7,
        },
        'num_samples': 10,
    }

    # ray.init()  # num_gpus=1)

    hyperband = tune.schedulers.AsyncHyperBandScheduler(time_attr='iterations', reward_attr='F2', max_t=12)

    tune.run_experiments({'cantrip_hyperband': param_spec},
                         scheduler=hyperband
                         )


class TrainCANTRIP(tune.Trainable):

    def _setup(self, config):
        self.cohort = tune.util.get_pinned_object(config['cohort_handle'])
        self.train_cohort, self.devel_cohort, self.test_cohort = tune.util.get_pinned_object(
            config['tdt_cohorts_handle'])

        self.data = self.cohort.balance_chronologies()

        self.num_epochs = config['num_epochs']

        self.delta_encoder = preprocess.TanhLogDeltaEncoder()

        self.batch_size = config['batch_size']
        self.max_snapshot_size = config['max_snapshot_size']
        self.max_chrono_length = config['max_seq_len']

        snapshot_encoder = config['snapshot_encoder']
        if snapshot_encoder == 'RNN':
            snapshot_encoder = encoding.rnn_encoder([200])
        elif snapshot_encoder == 'CNN':
            snapshot_encoder = encoding.cnn_encoder([3, 4, 5], 1000, config['dropout'])
        elif snapshot_encoder == 'SPARSE':
            snapshot_encoder = encoding.bag_encoder
        elif snapshot_encoder == 'DENSE':
            snapshot_encoder = encoding.dense_encoder
        else:
            snapshot_encoder = encoding.dan_encoder([200, 200], [200, 200])

        vocabulary_size = len(self.cohort.vocabulary)

        self.model = modeling.CANTRIPModel(
            max_seq_len=config['max_seq_len'],
            max_snapshot_size=config['max_snapshot_size'],
            vocabulary_size=vocabulary_size,
            observation_embedding_size=config['observation_embedding_size'],
            delta_encoding_size=self.delta_encoder.size,
            num_hidden=[config['rnn_num_hidden']] * config['rnn_num_layers'],
            cell_type=config['rnn_cell_type'] + ('-LN' if config['rnn_layer_norm'] else ''),
            batch_size=config['batch_size'],
            dropout=config['dropout'],
            vocab_dropout=config['vocab_dropout'],
            num_classes=2,
            snapshot_encoder=snapshot_encoder)

        if config['optimizer'] == 'BERT':
            num_train_steps = len(self.data[self.train_cohort].make_epoch_batches(batch_size=self.batch_size,
                                                                                  max_snapshot_size=self.max_snapshot_size,
                                                                                  max_chrono_length=self.max_chrono_length,
                                                                                  delta_encoder=self.delta_encoder))

            self.optimizer = optimization.BERTOptimizer(self.model, num_train_steps * self.num_epochs, num_train_steps,
                                                        init_lr=config['learning_rate'])
        else:
            self.optimizer = optimization.CANTRIPOptimizer(self.model, learning_rate=config['learning_rate'],
                                                           sparse=True)

        self.summarizer = summarization.CANTRIPSummarizer(self.model, self.optimizer)

        # Save summaries and checkpoints into the directories passed to the script
        model_file = 'rnnln=%d_tanh_d=%.2f_vd=%.2f_lr=%g_oes=%d_rnnh=%dx%d' % (
            1 if config['rnn_layer_norm'] else 0,
            config['dropout'],
            config['vocab_dropout'],
            config['learning_rate'],
            config['observation_embedding_size'],
            config['rnn_num_hidden'],
            config['rnn_num_layers']
        )
        self.model_summaries_dir = os.path.join(config['output_dir'], config['optimizer'], config['rnn_cell_type'],
                                                config['snapshot_encoder'], model_file)
        self.model_checkpoint_dir = os.path.join(config['output_dir'], config['optimizer'], config['rnn_cell_type'],
                                                 config['snapshot_encoder'], model_file)

        # nio.delete_dir_quiet(self.model_summaries_dir)
        # nio.delete_dir_quiet(self.model_checkpoint_dir)

        # Make output directories so we don't blow up when saving
        nio.make_dirs_quiet(self.model_checkpoint_dir)

        self.saver = tf.train.Saver(tf.global_variables())

        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess.run(tf.global_variables_initializer())
        # Initialize local variables (these are just used for computing average metrics)
        self.sess.run(tf.local_variables_initializer())
        self.iterations = 0

        self.summary_writer = tf.summary.FileWriter(self.model_summaries_dir, self.sess.graph)

        # Save the training, development, and testing metrics for our best model (as measured by devel F1)
        # I'm lazy so I initialize best_devel_metrics with a zero F1 so I can compare the first iteration to it
        self.best_train_metrics, self.best_devel_metrics, self.best_test_metrics = {}, {'F2': 0}, {}

    def _train(self):
        # Get global step and reset training metrics
        global_step, _ = self.sess.run([self.optimizer.global_step, self.summarizer.train.reset_op])
        self.iterations += 1
        # Log our progress on the current epoch using tqdm cohort.make_epoch_batches shuffles the order of
        # chronologies and prepares them  into mini-batches with zero-padding if needed
        total_loss = 0.
        train_batches = self.data[self.train_cohort].make_epoch_batches(batch_size=self.batch_size,
                                                                        max_snapshot_size=self.max_snapshot_size,
                                                                        max_chrono_length=self.max_chrono_length,
                                                                        delta_encoder=self.delta_encoder)
        num_batches = len(train_batches)
        # Iterate over each batch
        for train_batch in train_batches:
            # We train the model by evaluating the optimizer's training op. At the same time we update the
            # training metrics and get metrics/summaries for the current batch and request the new global
            # step number (used by TensorBoard to coordinate metrics across different runs
            _, batch_summary, batch_metrics, global_step = self.sess.run(
                [[self.optimizer.train_op, self.summarizer.train.metric_ops],
                 # All fetches we aren't going to read
                 self.summarizer.batch_summary, self.summarizer.batch_metrics,
                 self.optimizer.global_step],
                train_batch.feed(self.model, training=True))

            # Save batch-level summaries
            self.summary_writer.add_summary(batch_summary, global_step=global_step)

            total_loss += batch_metrics['Loss']

        # Save epoch-level training metrics and summaries
        train_metrics, train_summary = self.sess.run(
            [self.summarizer.train.metrics, self.summarizer.train.summary])
        train_metrics['Loss'] = total_loss / num_batches
        self.summary_writer.add_summary(train_summary, global_step=global_step)

        # Evaluate development performance
        self.sess.run(self.summarizer.devel.reset_op)
        # Update local variables used to compute development metrics as we process each batch
        for devel_batch in self.data[self.devel_cohort].make_epoch_batches(
                batch_size=self.batch_size,
                max_snapshot_size=self.max_snapshot_size,
                max_chrono_length=self.max_chrono_length,
                delta_encoder=self.delta_encoder):
            self.sess.run([self.summarizer.devel.metric_ops], devel_batch.feed(self.model, training=False))
        # Compute the development metrics
        devel_metrics, devel_summary = self.sess.run(
            [self.summarizer.devel.metrics, self.summarizer.devel.summary])
        # Save TensorBoard summary
        self.summary_writer.add_summary(devel_summary, global_step=global_step)

        # Evaluate testing performance exactly as described above for development
        self.sess.run(self.summarizer.test.reset_op)
        for test_batch in self.data[self.test_cohort].make_epoch_batches(batch_size=self.batch_size,
                                                                         max_snapshot_size=self.max_snapshot_size,
                                                                         max_chrono_length=self.max_chrono_length,
                                                                         delta_encoder=self.delta_encoder):
            self.sess.run([self.summarizer.test.metrics, self.summarizer.test.metric_ops],
                          test_batch.feed(self.model, training=False))
        test_metrics, test_summary = self.sess.run([self.summarizer.test.metrics, self.summarizer.test.summary])
        self.summary_writer.add_summary(test_summary, global_step=global_step)

        # Re-sample chronologies in cohort
        self.data = self.cohort.balance_chronologies()

        # If this run did better on the dev set, save it as the new best model
        if devel_metrics['F2'] > self.best_devel_metrics['F2']:
            self.best_devel_metrics = devel_metrics
            self.best_train_metrics = train_metrics
            self.best_test_metrics = test_metrics

            # Save the model
            self.saver.save(self.sess, os.path.join(self.model_checkpoint_dir, 'best'), global_step=global_step)

            with open(os.path.join(self.model_checkpoint_dir, 'eval.txt'), 'w') as out:
                run_cantrip.print_table_results(self.best_train_metrics, self.best_devel_metrics,
                                                self.best_test_metrics,
                                                'simple', file=out)

        return devel_metrics

    def _save(self, checkpoint_dir):
        prefix = self.saver.save(self.sess, os.path.join(self.model_checkpoint_dir, checkpoint_dir, 'save'),
                                 global_step=self.iterations)
        return {'prefix': prefix}

    def _restore(self, ckpt_data):
        prefix = ckpt_data['prefix']
        return self.saver.restore(self.sess, prefix)


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run(run_tune)
