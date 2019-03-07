import tensorflow as tf

from modeling import CANTRIPModel


class CANTRIPSummarizer(object):

    def __init__(self, model: CANTRIPModel, optimizer):
        self.model = model
        self.optimizer = optimizer

        # Batch-level confusion matrix
        self.tp = tf.count_nonzero(model.y * model.labels, dtype=tf.int32)
        self.tn = tf.count_nonzero((model.y - 1) * (model.labels - 1), dtype=tf.int32)
        self.fp = tf.count_nonzero(model.y * (model.labels - 1), dtype=tf.int32)
        self.fn = tf.count_nonzero((model.y - 1) * model.labels, dtype=tf.int32)

        # Batch-level binary classification metrics
        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.accuracy = (self.tp + self.tn) / model.batch_size
        self.specificity = self.tn / (self.tn + self.fp)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.f2 = 5 * self.precision * self.recall / (4 * self.precision + self.recall)

        # Dict of all metrics to make fetching more convenient
        self.batch_metrics = {
            'TP': self.tp,
            'TN': self.tn,
            'FP': self.fp,
            'FN': self.fn,
            'Precision': self.precision,
            'Recall': self.recall,
            'Accuracy': self.accuracy,
            'Specificity': self.specificity,
            'F1': self.f1,
            'F2': self.f2,
            'Loss': optimizer.loss
        }

        # Group all batch-level metrics in the same pane in TensorBoard using a name scope
        with tf.name_scope('Batch'):
            self.batch_summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', self.accuracy),
                tf.summary.scalar('Precision', self.precision),
                tf.summary.scalar('Recall', self.recall),
                tf.summary.scalar('F1', self.f1),
                tf.summary.scalar('F2', self.f2),
                tf.summary.scalar('Specificity', self.specificity),
                tf.summary.scalar('Loss', optimizer.loss)
            ])

        # Specific training/development/testing summarizers
        self.train = _CANTRIPModeSummarizer('train', model)
        self.devel = _CANTRIPModeSummarizer('devel', model)
        self.test = _CANTRIPModeSummarizer('test', model)


class _CANTRIPModeSummarizer(object):

    def __init__(self, mode: str, model: CANTRIPModel):
        self.mode = mode
        with tf.name_scope(mode) as scope:
            # Streaming, epoch-level metrics
            acc, acc_op = tf.metrics.accuracy(model.labels, model.y)
            auroc, auroc_op = tf.metrics.auc(model.labels, model.y, summation_method='careful_interpolation')
            auprc, auprc_op = tf.metrics.auc(model.labels, model.y, curve='PR',
                                             summation_method='careful_interpolation')
            p, p_op = tf.metrics.precision(model.labels, model.y)
            r, r_op = tf.metrics.recall(model.labels, model.y)
            f1 = 2 * p * r / (p + r)
            f2 = 5 * p * r / (4 * p + r)

            # Streaming, epoch-level confusion matrix information
            with tf.name_scope('confusion_matrix'):
                tp, tp_op = tf.metrics.true_positives(model.labels, model.y)
                tn, tn_op = tf.metrics.true_negatives(model.labels, model.y)
                fp, fp_op = tf.metrics.false_positives(model.labels, model.y)
                fn, fn_op = tf.metrics.false_negatives(model.labels, model.y)
                # Group these so they all show up together in TensorBoard
                confusion_matrix = tf.summary.merge([
                    tf.summary.scalar('TP', tp),
                    tf.summary.scalar('TN', tn),
                    tf.summary.scalar('FP', fp),
                    tf.summary.scalar('FN', fn),
                ])

            # Summary containing all epoch-level metrics computed for the current mode
            self.summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', acc),
                tf.summary.scalar('AUROC', auroc),
                tf.summary.scalar('AUPRC', auprc),
                tf.summary.scalar('Precision', p),
                tf.summary.scalar('Recall', r),
                tf.summary.scalar('F1', f1),
                tf.summary.scalar('F2', f2),
                confusion_matrix
            ])

            # Dictionary of epoch-level metrics for each fetching
            self.metrics = {
                'Accuracy': acc,
                'AUROC': auroc,
                'AUPRC': auprc,
                'Precision': p,
                'Recall': r,
                'F1': f1,
                'F2': f2,
            }

            # TensorFlow operations that need to be run to update the epoch-level metrics on each batch
            self.metric_ops = [acc_op, auroc_op, auprc_op, p_op, r_op,
                               [tp_op, tn_op, fp_op, fn_op]]

            # Operation to reset metrics after each epoch
            metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_op = tf.variables_initializer(var_list=metric_vars)