from collections import Iterable

import numpy as np

from tqdm import tqdm

_UNK = 'UNK'

_DELTA_BUCKETS = [1, 7, 30, 60, 90, 180, 360, 720]

class Vocabulary(object):
    def __init__(self, term_index=None, term_frequencies=None, terms=None, return_unk=True):
        if terms is None:
            self.terms = []
        else:
            self.terms = list(terms)

        if term_frequencies is None:
            self.term_frequencies = []
        elif isinstance(term_frequencies, dict):
            self.term_frequencies = np.zeros(len(terms), dtype=np.int32)
            for term, freq in term_frequencies.items():
                self.term_frequencies[term_index[term]] = freq
        else:
            self.term_frequencies = term_frequencies

        if term_index is None:
            self.term_index = {}
        else:
            self.term_index = term_index

        self.return_unk = return_unk

    @classmethod
    def from_scribe_file(cls, vocabulary_file, add_unk=True, return_unk=True, max_vocab_size=None):
        term_index = {}
        term_frequencies = []
        terms = []

        if add_unk:
            term_index[_UNK] = 0
            term_frequencies.append(0)
            terms.append(_UNK)

        with open(vocabulary_file, 'rt') as vocabulary:
            for i, line in enumerate(tqdm(vocabulary.readlines(), desc='Loading vocabulary')):
                if max_vocab_size is None or i < max_vocab_size:
                    term, frequency = line.split('\t')
                    term_index[term] = i
                    terms.append(term)
                    term_frequencies.append(int(frequency))
                else:
                    _, frequency = line.split
                    term_frequencies[0] += frequency

        return cls(term_index, term_frequencies, terms, return_unk=return_unk)

    @classmethod
    def from_terms(cls, terms, add_unk=True, return_unk=True, max_vocab_size=None):
        term_index = {}
        term_frequencies = []
        vocab_terms = []

        if add_unk:
            term_index[_UNK] = 0
            term_frequencies.append(0)
            vocab_terms.append(_UNK)

        for i, term in enumerate(terms):
            if max_vocab_size is None or i < max_vocab_size:
                if term in term_index:
                    term_frequencies[term_index[term]] += 1
                else:
                    term_index[term] = len(vocab_terms)
                    term_frequencies.append(1)
                    vocab_terms.append(term)

        return cls(term_index, term_frequencies, terms, return_unk=return_unk)

    def encode_term(self, term):
        if term in self.term_index:
            return term
        elif self.return_unk:
            return _UNK
        else:
            raise KeyError('Term \'' + term + '\' not found in vocabulary')

    def encode_term_id(self, term_id):
        if term_id < len(self.terms):
            return term_id
        elif self.return_unk:
            return 0
        else:
            raise KeyError('Term ID ' + term_id + ' not valid for vocabulary')

    def lookup_term_by_term_id(self, term_id):
        return self.terms[term_id]

    def lookup_term_id_by_term(self, term):
        if self.return_unk:
            return self.term_index.get(term, 0)
        else:
            return self.term_index[term]

    def add_term(self, term):
        if term in self.term_index:
            self.term_frequencies[self.term_index[term]] += 1
        else:
            self.term_index[term] = len(self.terms)
            self.terms.append(term)
            self.term_frequencies.append(1)
        return self.term_index[term]

    def __len__(self):
        return len(self.terms)


class Visit(object):

    def __init__(self, deltas, labels, docs):
        self.deltas = deltas
        self.labels = labels
        self.docs = docs

    def truncate(self, truncate_length):
        return Visit(self.deltas[:truncate_length],
                     self.labels[:truncate_length],
                     self.docs[:truncate_length])

    def __len__(self):
        return len(self.deltas)


# noinspection PyMissingConstructor
class Cohort(object):

    def __init__(self, patients=None, vocabulary=Vocabulary(), patient_vocabulary=Vocabulary()):
        self.vocabulary = vocabulary
        if patients is None:
            self.chronologies = np.empty(0, dtype=np.object)
        else:
            self.chronologies = np.asarray(patients, dtype=np.object)
        self.patient_vocabulary = patient_vocabulary

    @classmethod
    def from_dict(cls, cohort, vocabulary):
        patient_vocabulary = Vocabulary.from_terms(cohort.keys(),
                                                   add_unk=False, return_unk=False,
                                                   max_vocab_size=None)

        patients = []
        for subject_id, admissions in cohort.items():
            patient_vocabulary.add_term(subject_id)
            patients.append(admissions)

        return cls(patients, vocabulary, patient_vocabulary)

    @classmethod
    def from_chronologies(cls, patient_vectors, vocabulary, max_vocab_size=50_000):
        """
        Load cohort
        :param patient_vectors: file containing for chronology vectors for the cohort
        :param vocabulary: file containing the vocabulary used to generate chronology vectors
        :param max_vocab_size: maximum vocabulary size (words after this will be
        :return: Cohort object
        """
        if not isinstance(vocabulary, Vocabulary):
            vocabulary = Vocabulary.from_scribe_file(vocabulary, max_vocab_size=max_vocab_size)

        cohort = {}

        filtered_visits = 0
        with open(patient_vectors, 'rt') as vector_file:
            for line in tqdm(vector_file.readlines(), desc='Loading chronologies'):
                line = line.rstrip()
                subject_id, *dates = line.split('\t')
                if subject_id not in cohort:
                    cohort[subject_id] = []
                deltas = []
                labels = []
                docs = []
                for date in dates:
                    delta, label, *word_ids = date.split(' ')
                    deltas.append(int(delta))
                    if label == 'true':
                        labels.append(1)
                    elif label == 'false':
                        labels.append(0)
                    else:
                        raise ValueError('Encountered invalid label \'' + label + '\'')
                    docs.append([vocabulary.encode_term_id(int(word_id)) for word_id in word_ids])
                    if labels[-1] == 1:
                        break

                # We only care about patients who were eventually diagnosed, and who were not diagnosed in the first day
                if labels[-1] == 1 and len(deltas) > 2 and labels[0] == 0:
                    cohort[subject_id].append(Visit(deltas[1:], labels[1:], docs[:-1]))
                else:
                    filtered_visits += 1

        filtered_patients = 0
        for subject_id in list(cohort.keys()):
            if len(cohort[subject_id]) < 1:
                filtered_patients += 1
                del cohort[subject_id]

        print('Loaded cohort of %d patients with %d visits (after filtering %d patients and %d visits)' % (
            len(cohort.keys()), len(cohort.values()), filtered_patients, filtered_visits))

        return cls.from_dict(cohort, vocabulary)

    def __len__(self):
        return len(self.chronologies)

    def __getitem__(self, subject_id):
        if isinstance(subject_id, Iterable):
            indices = [self.patient_vocabulary.lookup_term_id_by_term(term) for term in subject_id]
            return Cohort(self.chronologies[indices],
                          self.vocabulary,
                          self.patient_vocabulary)
        else:
            return Cohort(self.chronologies[self.patient_vocabulary.lookup_term_by_term_id(subject_id)],
                          self.vocabulary,
                          self.patient_vocabulary)

    def patients(self):
        return self.patient_vocabulary.terms

    def visits(self):
        return [visit for patient in self.chronologies for visit in patient]

    def items(self):
        for subject_id in self.patient_vocabulary.terms:
            yield (subject_id, self[subject_id])

    def make_epoch_batches(self, batch_size, max_doc_len, max_seq_len, limit=None, **kwargs):
        visits = np.random.permutation(self.visits())

        balanced_visits = np.empty(2 * len(visits), dtype=visits.dtype)
        balanced_visits[0::2] = visits
        balanced_visits[1::2] = np.random.permutation([visit.truncate(-1) for visit in visits])

        # truncated_visits = []
        # for visit in visits:
        #     if len(visit) > 1:
        #         truncated_visits.append(visit.truncate(-1))
        #
        # balanced_visits = np.empty(len(visits) + len(truncated_visits), dtype=visits.dtype)
        # balanced_visits[0:2 * len(truncated_visits):2] = visits[:len(truncated_visits)]
        # balanced_visits[1:2 * len(truncated_visits) + 1:2] = np.random.permutation(truncated_visits)
        # balanced_visits[2 * len(truncated_visits):] = visits[len(truncated_visits):]

        # balanced_visits = np.concatenate([visits, np.random.permutation(truncated_visits)])

        num_batches = balanced_visits.shape[0] // batch_size
        if limit is not None:
            num_batches = min(num_batches, limit)

        # Throw away the last batch if its incomplete (shouldn't be an issue since we are permuting the order)
        truncate_length = num_batches * batch_size
        # print('Truncate:', truncate_length, 'vs.', 'Actual:', len(balanced_visits))
        balanced_visits = balanced_visits[:truncate_length]
        # print('Balanced.shape:', balanced_visits.shape, 'Num. batches:', num_batches)
        batches = np.split(balanced_visits, num_batches, axis=0)

        return [VisitBatch.from_visits(batch, max_doc_len, max_seq_len) for batch in batches]

    def make_subseq_epoch_batches(self, batch_size, max_doc_len, max_seq_len, limit=None, **kwargs):
        # Permute visit order
        visits = np.random.permutation(self.visits())

        all_visits = []
        for visit in visits:
            for i in range(len(visit.deltas)):
                all_visits.append(visit.truncate(i + 1))

        visits = np.asarray(sorted(all_visits, key=lambda v: len(v.deltas)))

        num_batches = visits.shape[0] // batch_size

        if limit is not None:
            num_batches = min(num_batches, limit)

        # Throw away the last batch if its incomplete (shouldn't be an issue since we are permuting the order)
        truncate_length = num_batches * batch_size
        visits = visits[:truncate_length]

        return [VisitBatch.from_visits(np.random.permutation(batch_visits),
                                       max_doc_len=max_doc_len,
                                       max_seq_len=max_seq_len) for batch_visits in
                np.split(visits, num_batches, axis=0)]

    def make_sampled_subseq_epoch_batches(self, batch_size, max_doc_len, max_seq_len, limit=None, **kwargs):
        # Permute visit order
        visits = np.random.permutation(self.visits())

        truncated_visits = []
        for visit in visits:
            truncate_length = np.random.random_integers(len(visit.deltas))
            truncated_visits.append(visit.truncate(truncate_length))

        visits = np.asarray(sorted(truncated_visits, key=lambda v: len(v.deltas)))

        num_batches = visits.shape[0] // batch_size

        if limit is not None:
            num_batches = min(num_batches, limit)

        # Throw away the last batch if its incomplete (shouldn't be an issue since we are permuting the order)
        truncate_length = num_batches * batch_size
        visits = visits[:truncate_length]

        return [VisitBatch.from_visits(batch_visits,
                                       max_doc_len=max_doc_len,
                                       max_seq_len=max_seq_len) for batch_visits in
                np.split(visits, num_batches, axis=0)]


class VisitBatch(object):

    def __init__(self, batch_size, deltas, seq_lens, labels, docs, doc_lens):
        self.batch_size = batch_size
        self.deltas = deltas
        self.seq_lens = seq_lens
        self.labels = labels
        self.docs = docs
        self.doc_lens = doc_lens

    @classmethod
    def from_visits(cls, visits, max_doc_len, max_seq_len):
        batch_size = batch_size = visits.shape[0]
        deltas = np.zeros([batch_size, max_seq_len, len(_DELTA_BUCKETS)], np.float32)
        seq_lens = np.zeros(batch_size, np.int32)
        labels = np.zeros(batch_size, np.int32)
        docs = np.zeros([batch_size, max_seq_len, max_doc_len], np.int32)
        doc_lens = np.ones([batch_size, max_seq_len], np.int32)

        # print('Batch Visits:', visits)

        for i, visit in enumerate(visits):
            seq_end = min(len(visit.deltas), max_seq_len)
            for j, delta in enumerate(visit.deltas[:seq_end]):
                # We discretize elapsed time into buckets
                # To preserve ordinality, we put a 1 into *every* bucket that is <= delta
                deltas[i, j] = [1 if delta >= bucket else 0 for bucket in _DELTA_BUCKETS]
            # deltas[i, :seq_end] = visit.deltas[:seq_end]
            labels[i] = visit.labels[seq_end - 1]
            seq_lens[i] = seq_end
            # print('Visit Docs:', visit.docs)
            for j, doc in enumerate(visit.docs[:seq_end]):
                # print('Visit %d Doc %d:' % (i, j), doc)
                doc_end = min(len(doc), max_doc_len)
                docs[i, j, :doc_end] = doc[:doc_end]
                doc_lens[i, j] = doc_end

        return cls(batch_size, deltas, seq_lens, labels, docs, doc_lens)

    def perturb_labels(self):
        return VisitBatch(self.batch_size,
                          self.deltas,
                          self.seq_lens,
                          np.random.permutation(self.labels),
                          self.docs,
                          self.doc_lens)

    def feed(self, model):
        return {model.words: self.docs,
                model.deltas: self.deltas,
                model.doc_lengths: self.doc_lens,
                model.seq_lengths: self.seq_lens,
                model.labels: self.labels}
