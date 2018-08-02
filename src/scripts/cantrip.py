import argparse
import os

import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import gfile

from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

from src.data import Cohort
from src.models import CANTRIPModel, CANTRIPOptimizer, CANTRIPSummarizer
# noinspection PyProtectedMember
from src.models.cantrip import _cell_types
from src.models.doc import rnn_encoder, cnn_encoder, bow_encoder, dan_encoder, dense_encoder
from src.models.util import make_dirs_quiet, delete_dir_quiet

np.random.seed(1337)

parser = argparse.ArgumentParser(description='invoke CANTRIP')
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--max-seq-len', type=int, default=7, help='maximum number of documents per chronology')
parser.add_argument('--max-doc-len', type=int, default=100, help='maximum number of terms to consider per document')
parser.add_argument('--vocabulary-size', type=int, default=50_000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--word-embedding-size', type=int, default=200, help="size of word embeddings")
parser.add_argument('--doc-embedding-size', type=int, default=200, help="size of document embeddings")
parser.add_argument('--num-hidden', type=int, nargs='+', default=[100], help="size of hidden layers in RNN")
parser.add_argument('--batch-size', type=int, default=40, help='batch size')
parser.add_argument('--cell-type', choices=_cell_types.keys(), default='LSTM', help='type of RNN cell to use')
parser.add_argument('--doc-encoder', choices=['RNN', 'CNN', 'BOW', 'DAN', 'DENSE'], default='RNN',
                    help='type of document encoder to use')

# doc_encoder = parser.add_mutually_exclusive_group()
doc_encoder_rnn = parser.add_argument_group(title='Document Encoder: RNN')
doc_encoder_rnn.add_argument('--doc-rnn-num-hidden', type=int, default=1,
                             help='number of layers in document encoder RNN')

doc_encoder_cnn = parser.add_argument_group(title='Document Encoder: CNN')
doc_encoder_cnn.add_argument('--doc-cnn-grams', type=int, nargs='?', default=[3, 4, 5], help='n-grams to use in CNN')
doc_encoder_cnn.add_argument('--doc-cnn-num-filters', type=int, default=1000, help='number of filters used in CNN')
doc_encoder_cnn.add_argument('--doc-cnn-dropout', type=float, default=0., help='dropout for CNN')
# doc_encoder_bow = doc_encoder.add_argument_group(title='Document Encoder: BOW')
# doc_encoder_bow.add_argument('--doc-bow', default=True)

doc_encoder_dan = parser.add_argument_group(title='Document Encoder: DAN')
doc_encoder_cnn.add_argument('--doc-dan-num-hidden-avg', type=int, nargs='+', default=[200, 200],
                             help='number of hidden units to use in document-level layers')
doc_encoder_cnn.add_argument('--doc-dan-num-hidden-word', type=int, nargs='+', default=[200, 200],
                             help='number of hidden units to use in word-level layers')

parser.add_argument('--summary-dir', default='data/working/summaries')
parser.add_argument('--checkpoint-dir', default='models/checkpoints')
parser.add_argument('--num-epochs', type=int, default='30', help='number of training epochs')
parser.add_argument('--validate-every', type=int, default='10',
                    help='how many training batches between validation steps')
parser.add_argument('--mode', choices=['TRAIN', 'TEST', 'INFER'], default='TRAIN')
parser.add_argument('--clear', default=False, action='store_true')
parser.add_argument('--debug', default=None, help='hostname:port of TensorBoard debug server')


def train_devel_test_split(data, ratio):
    train, devel, test = [int(x) for x in ratio.split(':')]

    total = train + devel + test
    train_devel_total = train + devel

    train_devel, _test = train_test_split(data, test_size=(test / total))
    _train, _devel = train_test_split(train_devel, test_size=(devel / train_devel_total))

    del train_devel
    return _train, _devel, _test


def train_model(model, args):
    cohort = Cohort.from_chronologies(args.chronology_path, args.vocabulary_path, args.vocabulary_size)

    train, devel, test = train_devel_test_split(cohort.patients(), args.tdt_ratio)

    def count_labels(split):
        from collections import Counter
        cnt = Counter()
        for visit in cohort[split].visits():
            for label in visit.labels:
                cnt[label] += 1
        return cnt

    print('Train:', count_labels(train))
    print('Devel:', count_labels(devel))
    print('Test:', count_labels(test))

    model_summaries_dir = os.path.join(args.summary_dir, args.cell_type, args.doc_encoder)
    model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.cell_type, args.doc_encoder)

    if args.clear:
        delete_dir_quiet(model_summaries_dir)
        delete_dir_quiet(model_checkpoint_dir)
        print('Deleted previous model summaries/checkpoints')

    make_dirs_quiet(model_checkpoint_dir)

    optimizer = CANTRIPOptimizer(model)
    summarizer = CANTRIPSummarizer(model, optimizer)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        if args.debug is not None:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, args.debug)

        summary_writer = tf.summary.FileWriter(model_summaries_dir, sess.graph)

        checkpoint = tf.train.get_checkpoint_state(model_checkpoint_dir)
        if checkpoint and gfile.Exists(checkpoint.model_checkpoint_path + '.index'):
            print("Reading model parameters from '%s'...", checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters...")
            sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())

        with trange(args.num_epochs, desc='Training') as train_log:
            for i in train_log:
                global_step, _ = sess.run([optimizer.global_step, summarizer.train.reset_op])
                with tqdm(cohort[train].make_epoch_batches(**vars(args)), desc='Epoch %d' % (i + 1)) as batch_log:
                    for j, batch in enumerate(batch_log):
                        # if np.random.random() < 0.25:
                        batch.deltas = np.zeros_like(batch.deltas)
                        _, batch_summary, batch_metrics, tensors, global_step = sess.run(
                            [[optimizer.train_op, summarizer.train.metric_ops],
                             summarizer.batch_summary, summarizer.batch_metrics,
                             {'SRNN.x': model.x,
                              'SRNN.final_out': model.seq_final_output,
                              'Model.out': model.output,
                              'Model.logits': model.logits},
                             optimizer.global_step],
                            batch.feed(model))
                        batch_log.set_postfix(batch_metrics)
                        summary_writer.add_summary(batch_summary, global_step=global_step)
                        # tqdm.write('Batch %d' % j)
                        # for name, tensor in tensors.items():
                        #     tqdm.write('%s: %s' % (name, str(tensor)))
                        # input('Press any key to continue...')

                        # input('Press any key to continue...')

                # if (j + 1) % args.validate_every == 0:
                sess.run(summarizer.devel.reset_op)
                for devel_batch in cohort[devel].make_epoch_batches(**vars(args)):
                    sess.run([summarizer.devel.metric_ops], devel_batch.feed(model))
                devel_metrics, devel_summary = sess.run([summarizer.devel.metrics, summarizer.devel.summary])
                train_log.set_postfix(devel_metrics)
                summary_writer.add_summary(devel_summary, global_step=global_step)

                summary_writer.add_summary(sess.run(summarizer.train.summary), global_step=global_step)

                saver.save(sess, model_checkpoint_dir, global_step=global_step)

                sess.run(summarizer.test.reset_op)
                for batch in cohort[test].make_epoch_batches(**vars(args)):
                    sess.run([summarizer.test.metrics, summarizer.test.metric_ops], batch.feed(model))
                summary_writer.add_summary(sess.run(summarizer.test.summary), global_step=global_step)


def main(argv):
    args = parser.parse_args(argv[1:])

    if args.doc_encoder == 'RNN':
        doc_encoder = rnn_encoder(args.doc_rnn_num_hidden)
    elif args.doc_encoder == 'CNN':
        doc_encoder = cnn_encoder(args.doc_cnn_grams, args.doc_cnn_num_filters, args.doc_cnn_dropout)
    elif args.doc_encoder == 'BOW':
        doc_encoder = bow_encoder
    elif args.doc_encoder == 'DENSE':
        doc_encoder = dense_encoder
    elif args.doc_encoder == 'DAN':
        doc_encoder = dan_encoder(args.doc_dan_num_hidden_word, args.doc_dan_num_hidden_avg)
    else:
        raise ValueError('Given illegal document encoder %s' % args.doc_encoder)

    model = CANTRIPModel(max_seq_len=args.max_seq_len,
                         max_doc_len=args.max_doc_len,
                         vocabulary_size=args.vocabulary_size,
                         embedding_size=args.word_embedding_size,
                         num_hidden=args.num_hidden,
                         cell_type=args.cell_type,
                         batch_size=args.batch_size,
                         doc_embedding=doc_encoder,
                         num_classes=2)

    if args.mode == 'TRAIN':
        train_model(model, args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run(main)
