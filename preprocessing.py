"""Contains classes and methods for parsing and representing chronologies.

Attributes:
    _UNK (str): Module level private-variable containing the unknown token symbol
"""
from collections import Sized, ItemsView
from datetime import datetime
from datetime import timedelta

from dateutil.parser import parse

import typing
import io

import numpy as np

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from mock_tqdm import trange, tqdm

from absl import logging, flags

# Prediction window settings
flags.DEFINE_integer('min_start_window', default=48, lower_bound=0,
                     help='The minimum length of time (in hours) between the start time and first label '
                          '(chronologies for which the label occurs within this window will be discarded)')
flags.DEFINE_integer('min_pred_window', default=24, lower_bound=0,
                     help='The minimum time (in hours) between the last snapshot and the label '
                          '(chronologies will be truncated to end at least <min_pred_window> hours before '
                          'the first label)')
flags.DEFINE_integer('max_pred_window', default=76, lower_bound=0,
                     help='The maximum time (in hours) between the last snapshot and the label '
                          '(chronologies whose last snapshot occurs more than this many hours before the label will be '
                          'discarded)')
flags.DEFINE_integer('max_snapshot_delay', default=96, lower_bound=0,
                     help='The maximum number of hours between admission and first snapshot'
                          '(chronologies whose first snapshot occurs after this value will be discarded)')

# Chronology settings
flags.DEFINE_integer('max_chrono_length', default=7, lower_bound=1,
                     help='The maximum number of snapshots per chronology '
                          '(chronologies will be truncated from the end).')
flags.DEFINE_integer('min_chrono_length', default=3, lower_bound=2,
                     help='The minimum number of snapshots per chronology '
                          '(chronologies with fewer than the given length will be discarded).')

# Snapshot settings
flags.DEFINE_integer('min_snapshot_size', default=10, lower_bound=0,
                     help='The minimum number of observations to consider per snapshot '
                          '(snapshots with fewer observations will be discarded).')
flags.DEFINE_integer('max_snapshot_size', default=500, lower_bound=1,
                     help='The maximum number of observations to consider per snapshot '
                          '(observations will be truncated as read).')

# Vocabulary parameters
flags.DEFINE_integer('max_vocab_size', default=50000, lower_bound=1,
                     help='The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be '
                          'used to encode clinical snapshots. Any remaining observations will be discarded.')

# Elapsed time parameters
flags.DEFINE_enum('delta_enc', default='logsig', enum_values=['logsig', 'logtanh', 'discrete', 'raw', 'sinusoid'],
                  help='Method for encoding elapsed time.')
flags.DEFINE_enum('delta_combine', default='concat', enum_values=['concat', 'add'],
                  help='How to combine deltas and observation embeddings')
flags.DEFINE_enum('time_repr', default='prev', enum_values=['prev', 'start', 'both'],
                  help='Whether to encode elapsed times as time since previous snapshot, '
                       'time since chronology start, or both.')

FLAGS = flags.FLAGS

# Symbol used to denote unknown or out-of-vocabulary words
_UNK = 'UNK'


class DeltaEncoder:

    def encode_delta(self, elapsed_seconds: int) -> typing.Sequence[typing.Union[int, float]]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


class DiscreteDeltaEncoder(DeltaEncoder):
    DELTA_BUCKETS = [1, 2, 3, 4, 8, 12, 20, 30]

    def encode_delta(self, elapsed_seconds: int):
        """Encode deltas into discrete buckets indicating:
            1. if elapsed_days <= 1 day
            2. if elapsed_days <= 1 week
            3. if elapsed_days <= 30 days (1 month)
            4. if elapsed_days <= 60 days (2 months)
            5. if elapsed_days <= 90 days (3 months)
            6. if elapsed_days <= 182 days (half-a-year)
            7. if elapsed_days <= 365 days (1 year)
            8. if elapsed_days <= 730 days (2 years)

        :rtype: typing.List[int]
        :param elapsed_seconds: number of seconds between this clinical snapshot and the previous
        :type elapsed_seconds int
        :return: 8-dimensional discrete binary representation of elapsed_seconds
        """
        elapsed_days = elapsed_seconds / 60. / 60. / 24.
        return [1 if elapsed_days <= bucket else 0 for bucket in DiscreteDeltaEncoder.DELTA_BUCKETS]

    @property
    def size(self) -> int:
        return len(DiscreteDeltaEncoder.DELTA_BUCKETS)


class TanhLogDeltaEncoder(DeltaEncoder):
    def encode_delta(self, elapsed_seconds: int):
        """Encode deltas into discrete buckets

        :param elapsed_seconds: number of seconds between this clinical snapshot and the previous
        :return: tanh(log(elapsed days + 1))
        """
        elapsed_days = elapsed_seconds / 60. / 60. / 24.
        if elapsed_days == 1:
            return [0.]
        return [np.tanh(np.log(elapsed_days + 1))]

    @property
    def size(self):
        return 1


class LogSigmoidDeltaEncoder(DeltaEncoder):
    def encode_delta(self, elapsed_seconds: int):
        """Encode deltas into discrete buckets

        :param elapsed_seconds: number of seconds between this clinical snapshot and the previous
        :return: tanh(log(elapsed days + 1))
        """
        elapsed_days = elapsed_seconds / 60. / 60. / 24.
        return [elapsed_days / (np.abs(elapsed_days) + 1.)]

    @property
    def size(self):
        return 1


class RawDeltaEncoder(DeltaEncoder):
    def encode_delta(self, elapsed_seconds: int):
        return [elapsed_seconds / 60. / 60. / 24.]

    @property
    def size(self):
        return 1


# Modified from: https://github.com/jadore801120/attention-is-all-you-need-pytorch
class SinusoidalEncoder(DeltaEncoder):

    def __init__(self, dimensions: int, base=10000):
        super(SinusoidalEncoder, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def cal_angle(self, position, idx):
        return position / np.power(10000, 2 * (idx // 2) / self.dimensions)

    def get_angle_vector(self, position):
        return [self.cal_angle(position, idx) for idx in range(self.dimensions)]

    def encode_delta(self, elapsed_seconds: int) -> typing.Sequence[typing.Union[int, float]]:
        elapsed_days = elapsed_seconds / 60. / 60. / 24.

        sinusoid_vector = self.get_angle_vector(elapsed_days)

        sinusoid_vector[0::2] = np.sin(sinusoid_vector[0::2])  # dim 2i
        sinusoid_vector[1::2] = np.cos(sinusoid_vector[1::2])  # dim 2i+1

        return sinusoid_vector

    @property
    def size(self) -> int:
        return self.dimensions


class Vocabulary(object):
    """A bi-directional lookup table between categorical terms and integer term ids.
    Term ids are contiguous, and start from 0.

    Attributes:
        terms (list): a list of terms in the vocabulary; maps term IDs to terms
        term_index (dict): a dict mapping terms to their term ID
        term_frequencies (list): a list mapping term IDs to term frequencies
    """

    def __init__(self,
                 term_index=None,
                 term_frequencies=None,
                 terms=None,
                 return_unk: bool = True):
        """Creates a vocabulary from a given term_index, term_frequency and term list

        :param term_index (dict, optional): dictionary mapping terms to integer term ids
            (term ids must start from zero and be contiguous)
        :type term_index: typing.Optional[typing.Dict[typing.Text, int]]
        :param term_frequencies: a dict or list associating each term with its frequency. If passed as a dict,
            each key must be a term and each value must be a positive integer indicating the frequency of that term.
            If passed as a list,  each entry in the list is assumed to be a positive integer indicating the frequency
            of the term whose term ID matches the index of that entry
        :type term_frequencies: typing.Optional[typing.Union[typing.Dict[typing.Text, int], typing.Iterable[int]]
        :param terms: a list of terms where the index of each term is the term ID of that term
        :type terms: typing.Optional[typing.Iterable[int]]
        :param return_unk: whether the unknown term symbol should be returned when attempting to access
            out-of-vocabulary terms from this Vocabulary
        :type return_unk: bool
        """
        if terms is None:
            self.terms = []
        else:
            self.terms = list(terms)

        if term_frequencies is None:
            self.term_frequencies = []
        # Convert term -> frequency dict to term ID -> frequency list
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

        self._identify_np = None

    @property
    def identify_np(self):
        if not self._identify_np:
            self._identify_np = np.frompyfunc(self.identify, 1, 1)
        return self._identify_np

    @classmethod
    def empty(cls, add_unk: bool = True, return_unk: bool = True):
        term_index = {}
        term_frequencies = []
        terms = []

        if add_unk:
            term_index[_UNK] = 0
            term_frequencies.append(0)
            terms.append(_UNK)

        return cls(term_index, term_frequencies, terms, return_unk)

    @classmethod
    def from_tsv(cls,
                 vocabulary_file: str,
                 add_unk: bool = True,
                 return_unk: bool = True,
                 max_vocab_size: typing.Optional[int] = None):
        """Loads a Vocabulary object from a given vocabulary TSV file path.

        The TSV file is assumed to be formatted as follows::

            [term]\t[frequency]

        such that the line number indicates the ID of the ``[term]`` on each line, ``[frequency]`` indicates the number
        of occurrences of that term in the training dataset, and terms are listed in descending order of their
        frequency.

        :param vocabulary_file: path to the TSV-formatted vocabulary file to load
        :param add_unk: whether to add the unknown term symbol the vocabulary (set to false if vocabulary
            already includes the unknown term symbol)
        :param return_unk: whether this Vocabulary object should return the unknown term symbol or raise a KeyError
            when asked to lookup out-of-vocabulary terms
        :param max_vocab_size: maximum vocabulary size. When loading from the disk, only the top max_vocab_size
            most-frequent terms will be retained. All other terms will be associated with the unknown term symbol (and
            their frequency will included in the unknown symbol's frequency). If None all terms are loaded.
        :return: a new Vocabulary object
        """
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
                    term_index[term] = len(terms)
                    terms.append(term)
                    term_frequencies.append(int(frequency))
                else:
                    _, frequency = line.split('\t')
                    term_frequencies[0] += int(frequency)

        return cls(term_index, term_frequencies, terms, return_unk=return_unk)

    @classmethod
    def from_terms(cls,
                   terms: typing.Sequence[typing.Text],
                   add_unk: bool = True,
                   return_unk: bool = True,
                   max_vocab_size: typing.Optional[int] = None):
        """Creates a vocabulary from an iterable of (possibly duplicate) terms.

        Unlike from_tsv, if max_vocab_size is specified the vocabulary will be created from the first max_vocab_size
        encountered unique terms, rather than the top max_vocab_size most frequent terms

        :param terms: iterable containing terms to add to the vocabulary
        :param add_unk: whether the unknown term symbol should be added to the vocabulary
        :param return_unk: whether this Vocabulary should return the unknown term symbol or raise KeyError when
            looking up out-of-vocabulary terms
        :param max_vocab_size: maximum vocabulary size. Unlike from_tsv, if max_vocab_size is specified, the vocabulary
            will be created from the first max_vocab_size encountered unique terms, rather than the top max_vocab_size
            most frequent terms.
        :return: a shiny new Vocabulary object
        """
        term_index = {}
        term_frequencies = []
        vocab_terms = []

        if add_unk:
            term_index[_UNK] = 0
            term_frequencies.append(0)
            vocab_terms.append(_UNK)

        for i, term in enumerate(terms):
            if term in term_index:
                term_frequencies[term_index[term]] += 1
            elif max_vocab_size is None or len(terms) < max_vocab_size - 1:
                term_index[term] = len(vocab_terms)
                term_frequencies.append(1)
                vocab_terms.append(term)
            elif add_unk:
                term_frequencies[0] += 1

        return cls(term_index, term_frequencies, terms, return_unk=return_unk)

    def encode_term(self, term: str):
        """Encode the given term using this vocabulary.

        Terms included in this vocabulary will be returned as-is. Out-of-vocabulary terms will be returned as the
        unknown term symbol if this Vocabulary was created with  return_unk=True. Otherwise, encoding out-of-vocabulary
        terms will raise a KeyError.

        :param term: term to encode
        :return:  term or unknown term symbol
        """
        if term in self.term_index:
            return term
        elif self.return_unk:
            return _UNK
        else:
            raise KeyError('Term \'' + term + '\' not found in vocabulary')

    def encode_term_id(self, term_id: int):
        """Encode the given term ID using this vocabulary.

        Term IDs included in this vocabulary will be returned as-is. Out-of-vocabulary term iss will be returned as the
        unknown term symbol ID (0) if this Vocabulary was created with return_unk=True. Otherwise, encoding
        out-of-vocabulary term IDs will raise a KeyError.

        :param term_id: term ID to encode
        :return:  term_id or unknown term symbol ID
        """
        if term_id < len(self.terms):
            return term_id
        elif self.return_unk:
            return self.term_index[_UNK]
        else:
            raise KeyError('Term ID %d not valid for vocabulary' % term_id)

    def lookup_term_by_term_id(self, term_id: int):
        """Look-up term by term by given term ID.

        Raises KeyException for invalid term IDs.

        :param term_id: ID of term to lookup
        :return: term associated with term_id in this vocabulary
        """
        return self.terms[term_id]

    def identify(self, term: typing.Text) -> int:
        """Look-up term by term ID for given term.

        Returns unknown term symbol if this vocabulary was created with return_unk=True, otherwise raises KeyError

        :param term: term to lookup
        :return: term ID associated with given term in this vocabulary
        """
        if self.return_unk:
            assert _UNK in self.term_index
            return self.term_index.get(term, _UNK)
        else:
            return self.term_index[term]

    def resize(self,
               size: int,
               add_unk: bool = True,
               return_unk: bool = True):
        has_unk = self.terms[0] == _UNK
        if add_unk and not has_unk and size - 1 >= len(self.terms):
            return self
        elif size >= len(self.terms):
            return self

        terms = self.terms
        term_index = self.term_index
        term_frequencies = self.term_frequencies
        if add_unk:
            if has_unk:
                unk_freq = self.term_frequencies[0]
                sorted_ids = np.argsort(self.term_frequencies[1:])
            else:
                unk_freq = 0
                sorted_ids = np.argsort(self.term_frequencies)
                terms = [_UNK] + terms
                term_index = {k: v + 1 for k, v in term_index.items()}
                term_index[_UNK] = 0
                term_frequencies = [0] + term_frequencies

            for id_ in sorted_ids[:-size]:
                unk_freq += self.term_frequencies[id_]

            term_frequencies[0] += unk_freq
        else:
            sorted_ids = np.argsort(self.term_frequencies)

        top_ids = sorted_ids[-size::-1]
        terms = terms[top_ids]
        term_index = {term: term_id for term_id, term in enumerate(terms)}
        term_frequencies = term_frequencies[top_ids]

        return Vocabulary(term_index=term_index,
                          terms=terms,
                          term_frequencies=term_frequencies,
                          return_unk=return_unk)

    def add_term(self, term: typing.Text) -> int:
        """Adds the given term to this vocabulary and returns its ID.

        If the given term is already in this vocabulary, its frequency will be updated. This method ignores any
        maximum_vocabulary_size given when creating the vocabulary.

        :param term: term to add
        :return: new or existing ID of the given term in this vocabulary
        """
        if term in self.term_index:
            self.term_frequencies[self.term_index[term]] += 1
        else:
            self.term_index[term] = len(self.terms)
            self.terms.append(term)
            self.term_frequencies.append(1)
        return self.term_index[term]

    def __len__(self):
        """Return the number of unique terms in this vocabulary"""
        return len(self.terms)

    def __contains__(self, item):
        return item in self.terms


class Admission(object):
    __slots__ = ['hadm_id', 'patient_id']

    def __init__(self, hadm_id: str, patient_id: str):
        self.hadm_id = hadm_id
        self.patient_id = patient_id

    # pool = {}
    #
    # def __new__(cls, hadm_id: str, patient_id: str):
    #     o = cls.pool.get(hadm_id)
    #     if o:
    #         assert(o.patient_id == patient_id)
    #         return o
    #     else:
    #         hadm = super(Admission, cls).__new__(cls)
    #         cls.pool[hadm_id] = hadm
    #         hadm.hadm_id = hadm_id
    #         hadm.patient_id = patient_id
    #         return hadm


class Label(object):
    __slots__ = ['value', 'timestamp', 'hadm']

    def __init__(self, value: int, timestamp: datetime, hadm: Admission):
        self.value = value
        self.timestamp = timestamp
        self.hadm = hadm

    def __int__(self) -> int:
        return self.value

    def __repr__(self):
        return 'Label(%d, %s)' % (self.value, self.timestamp)

    def __str__(self):
        return '%d@%s' % (self.value, self.timestamp)


class Snapshot(Sized, object):
    __slots__ = ['_observations', 'timestamp', 'vocabulary', 'hadm', '__observations', '__vector', '__len']

    def __init__(self,
                 observations: typing.Sequence[str],
                 timestamp: datetime,
                 vocabulary: Vocabulary,
                 hadm: Admission):
        self._observations = observations
        self.timestamp = timestamp
        self.vocabulary = vocabulary
        self.hadm = hadm
        self.__observations = None
        self.__vector = None
        self.__len = None

    # def __getstate__(self):
    #     state = dict(self.__dict__)
    #     del state['_Snapshot__observations']
    #     del state['_Snapshot__vector']
    #     del state['_Snapshot__len']
    #     return state

    @property
    def raw_observations(self):
        return self._observations

    @property
    def observations(self) -> typing.Sequence[int]:
        if self.__observations is None:
            self.__observations = self.vocabulary.identify_np(np.asarray(self._observations, np.str))
        return self.__observations

    def to_vector(self) -> np.ndarray:
        if self.__vector is None:
            self.__vector = np.zeros(FLAGS.max_snapshot_size, dtype=np.uint32)
            obs_ids = self.observations
            length = min(len(obs_ids), FLAGS.max_snapshot_size)
            self.__vector[:length] = obs_ids[:length]
        return self.__vector

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = min(len(self.observations), FLAGS.max_snapshot_size)
        return self.__len


class Chronology(Sized, object):
    __slots__ = ['start_time', 'label', 'snapshots', 'hadm',
                 '__matrix', '__deltas_matrix', '__snapshot_size_array', '__len']

    def __init__(self, start_time: datetime, snapshots: typing.List[Snapshot], label: Label, hadm: Admission):
        # assert all(ss.timestamp > start_time for ss in snapshots)
        self.start_time = start_time  # type: datetime
        self.label = label  # or   # type: Label
        assert len(set(ss.vocabulary for ss in snapshots)) == 1
        self.snapshots = sorted(snapshots, key=lambda ss: ss.timestamp)
        assert (self.snapshots == snapshots)
        self.hadm = hadm
        self.__matrix = None
        self.__deltas_matrix = None
        self.__snapshot_size_array = None
        self.__len = None

    # def __getstate__(self):
    #     state = dict(self.__dict__)
    #     del state['_Chronology__matrix']
    #     del state['_Chronology__deltas_matrix']
    #     del state['_Chronology__snapshot_size_array']
    #     del state['_Chronology__len']
    #     return state

    @property
    def observation_matrix(self) -> np.ndarray:
        if self.__matrix is None:
            self.__matrix = np.zeros(shape=(FLAGS.max_chrono_length, FLAGS.max_snapshot_size), dtype=np.uint32)
            len_ = len(self)
            seq_end = min(len_, FLAGS.max_chrono_length)

            for i in range(seq_end):
                self.__matrix[i, :] = self.snapshots[i].to_vector()

        return self.__matrix

    @property
    def class_label(self) -> int:
        return int(self.label)

    @property
    def deltas_prev(self) -> np.ndarray:
        # Subtract successive pairs of timestamps
        return np.ediff1d(
            np.asarray([self.start_time] + [ss.timestamp for ss in self.snapshots], dtype='datetime64[s]')
        )

    @property
    def deltas_start(self) -> np.ndarray:
        return np.asarray(
            [ss.timestamp - self.start_time for ss in self.snapshots],
            dtype='timedelta64[s]'
        )

    @property
    def delta_matrix(self) -> np.ndarray:
        if self.__deltas_matrix is None:
            if FLAGS.delta_enc == 'logsig':
                encoder = LogSigmoidDeltaEncoder()
            elif FLAGS.delta_enc == 'logtanh':
                encoder = TanhLogDeltaEncoder()
            elif FLAGS.delta_enc == 'discrete':
                encoder = DiscreteDeltaEncoder()
            elif FLAGS.delta_enc == 'sinusoid':
                if FLAGS.delta_combine == 'add':
                    if FLAGS.time_repr == 'both':
                        dimensions = FLAGS.observation_embedding_size // 2
                    else:
                        dimensions = FLAGS.observation_embedding_size
                else:
                    dimensions = FLAGS.sinusoidal_embedding_size
                encoder = SinusoidalEncoder(dimensions=dimensions)
            else:
                encoder = RawDeltaEncoder()  # type: DeltaEncoder

            # The number of snapshots is 1 more than the number of deltas, so we start at 1 instead of 0
            seq_end = min(len(self), FLAGS.max_chrono_length)

            do_both = FLAGS.time_repr == 'both'
            if FLAGS.time_repr == 'prev' or do_both:
                prev = np.array(list(map(encoder.encode_delta, self.deltas_prev.astype(dtype=np.uint32))))

                if not do_both:
                    self.__deltas_matrix = np.zeros([FLAGS.max_chrono_length, encoder.size], dtype=np.float32)
                    try:
                        self.__deltas_matrix[:seq_end] = prev
                    except ValueError:
                        print('Deltas[prev]:', self.deltas_prev.astype(dtype=np.uint32), 'shape:',
                              self.deltas_prev.shape)
                        print('Encoded deltas[prev]:', prev, ' shape:', prev.shape)
                        print('Desired length: %d to %d' % (0, seq_end))
                        print('Desired shape:', self.__deltas_matrix[:seq_end].shape)
                        raise

            if FLAGS.time_repr == 'start' or do_both:
                start = np.array(list(map(encoder.encode_delta, self.deltas_start.astype(dtype=np.float32))))
                if not do_both:
                    self.__deltas_matrix = np.zeros([FLAGS.max_chrono_length, encoder.size], dtype=np.float32)
                    self.__deltas_matrix[:seq_end] = start

            if do_both:
                self.__deltas_matrix = np.zeros([FLAGS.max_chrono_length, 2 * encoder.size], dtype=np.float32)
                # noinspection PyUnboundLocalVariable
                self.__deltas_matrix[:seq_end] = np.concatenate([start, prev], axis=1)

        return self.__deltas_matrix

    def truncate_to(self, index, label_value=0):
        if abs(index) == len(self):
            return self
        elif abs(index) > len(self) or index == 0:
            raise IndexError
        return Chronology(start_time=self.start_time,
                          snapshots=self.snapshots[:index],
                          label=Label(label_value, self.snapshots[index].timestamp, self.hadm),
                          hadm=self.hadm)

    @property
    def snapshot_size_array(self) -> np.ndarray:
        if self.__snapshot_size_array is None:
            self.__snapshot_size_array = np.zeros(FLAGS.max_chrono_length, dtype=np.uint32)
            seq_end = min(len(self), FLAGS.max_chrono_length)
            self.__snapshot_size_array[:seq_end] = [len(ss) for ss in self.snapshots]
        return self.__snapshot_size_array

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = self.snapshots.__len__()
        return self.__len


class Cohort(ItemsView, object):
    __slots__ = ['admissions', 'chronologies']

    def __init__(self,
                 admissions: typing.Mapping[str, typing.AbstractSet[str]],
                 chronologies: typing.Mapping[str, Chronology]):
        self.admissions = admissions
        self.chronologies = chronologies
        assert len(set(ss.vocabulary for chronology in self.chronologies.values() for ss in chronology.snapshots)) == 1

    @property
    def vocabulary(self) -> Vocabulary:
        for chronology in self.chronologies.values():
            for snapshot in chronology.snapshots:
                return snapshot.vocabulary

    @property
    def patients(self) -> typing.AbstractSet[str]:
        """Returns the set of distinct subject_ids comprising the cohort

        :return: the set of subject_ids in the cohort
        """
        return self.admissions.keys()

    def __getitem__(self, subject_id: str):
        """Given an iterable or single external id, return the sub-cohort of patients associated with that/those ids"""
        if isinstance(subject_id, typing.Iterable):
            return Cohort(admissions={sid: self.admissions[sid] for sid in subject_id},
                          chronologies={sid: self.chronologies[sid] for sid in subject_id})
        else:
            return Cohort(admissions={subject_id: self.admissions[subject_id]},
                          chronologies={subject_id: self.chronologies[subject_id]})

    def to_list(self) -> typing.Sequence[Chronology]:
        """Returns a flattened list of all chronologies in this cohort
        """
        return list(self.chronologies.values())

    def items(self) -> typing.Iterable[typing.Tuple[str, Chronology]]:
        """Returns tuples of subject IDs to list of chronologies (for dictionary-like iteration)"""
        for subject_id in self.patients:
            yield (subject_id, self[subject_id])

    @classmethod
    def from_csv_files(cls,
                       feature_csv: typing.Text,
                       admission_csv: typing.Text,
                       label_csv: typing.Text,
                       vocab=None,
                       lock_vocab=False):
        import csv

        # Store admissions by subject, sorted by hadm_id
        hadms_by_subject = {}  # type: typing.Dict[str, typing.MutableSet[str]]

        # Store snapshots by admission, sorted by timestamp
        snapshots_by_hadm = {}  # type: typing.Dict[str, typing.Dict[datetime, typing.List[str]]]

        # Step 1: Parse snapshot file
        with open(feature_csv) as snapshots_file:
            snapshots_file = io.StringIO(snapshots_file.read())
            snapshots_reader = csv.reader(snapshots_file)
            # Skip header
            next(snapshots_reader)
            for row in snapshots_reader:
                assert len(row) == 4
                subject_id, hadm_id, timestamp_, observations_ = row  # type: str
                try:
                    timestamp = parse(timestamp_)
                except ValueError:
                    if timestamp_ or observations_:
                        print('Failed to parse row', row)
                    continue
                observations = observations_.split()  # type: typing.List[str]

                if hadm_id not in snapshots_by_hadm:
                    snapshots_by_hadm[hadm_id] = dict()

                if timestamp not in snapshots_by_hadm[hadm_id]:
                    # Associate observations with the given timestamp
                    snapshots_by_hadm[hadm_id][timestamp] = observations
                else:
                    # We already have observations for this timestamp, so merge in the new ones
                    # print('Merging snapshots for %s @ %s' % (hadm_id, timestamp))
                    snapshots_by_hadm[hadm_id][timestamp].extend(observations)

                if subject_id not in hadms_by_subject:
                    hadms_by_subject[subject_id] = set()
                hadms_by_subject[subject_id].add(hadm_id)

        # Step 2: Parse start (admission) times for each hadm
        start_time_by_hadm = {}  # type: typing.Dict[str, datetime]
        with open(admission_csv) as admissions_file:
            admissions_file = io.StringIO(admissions_file.read())
            admissions_reader = csv.reader(admissions_file)
            # Skip header
            next(admissions_reader)
            for row in admissions_reader:
                assert len(row) == 3
                subject_id, hadm_id, timestamp_ = row  # type: str
                timestamp = parse(timestamp_)

                if subject_id not in hadms_by_subject:
                    continue

                if hadm_id not in snapshots_by_hadm:
                    continue

                if hadm_id not in hadms_by_subject[subject_id]:
                    print('Found admission %s for subject %s which was not in %s', hadm_id, subject_id, feature_csv)

                if hadm_id in start_time_by_hadm:
                    print('Found multiple starttimes for %s: %s & %s', hadm_id, timestamp, start_time_by_hadm[hadm_id])

                start_time_by_hadm[hadm_id] = timestamp

        labels_by_hadm = {}  # type: typing.Dict[str, Label]
        with open(label_csv) as label_file:
            label_file = io.StringIO(label_file.read())
            label_reader = csv.reader(label_file)
            # Skip header
            next(label_reader)
            for row in label_reader:
                assert len(row) == 4
                subject_id, hadm_id, timestamp_, label_ = row  # type: str
                timestamp = parse(timestamp_)
                value = int(label_)

                if subject_id not in hadms_by_subject:
                    continue

                if hadm_id not in snapshots_by_hadm:
                    continue

                if hadm_id not in hadms_by_subject[subject_id]:
                    print('Found label %s for subject %s which was not in %s', hadm_id, subject_id, feature_csv)

                label = Label(value, timestamp, hadm=Admission(hadm_id, subject_id))

                # If we haven't seen a label for this admission, or if this label pre-dates the previous label
                if hadm_id not in labels_by_hadm:
                    labels_by_hadm[hadm_id] = label
                else:
                    raise ValueError(
                        'hadm %s had multiple labels: %s and %s' % (hadm_id, label, labels_by_hadm[hadm_id]))

        # Debugging variables
        ss_too_small, ss_too_early, ss_too_late = 0, 0, 0
        label_too_early = 0
        chrono_too_short = 0
        chrono_too_late = 0
        chrono_window_too_wide = 0
        empty_subjects = 0

        min_start_window = timedelta(hours=FLAGS.min_start_window)
        min_pred_window = timedelta(hours=FLAGS.min_pred_window)
        min_proceeding_window = timedelta(hours=24)
        max_snapshot_delay = timedelta(hours=FLAGS.max_snapshot_delay)
        max_pred_window = timedelta(hours=FLAGS.max_pred_window)

        # Build vocabulary
        vocab = vocab or Vocabulary(return_unk=False)  # type: Vocabulary
        for subject_id, hadm_ids in list(hadms_by_subject.items()):
            for hadm_id in list(hadm_ids):
                snapshot_list = snapshots_by_hadm[hadm_id]  # type: typing.Dict[datetime, typing.List[str]]
                start_time = start_time_by_hadm[hadm_id]
                label = labels_by_hadm[hadm_id]

                # Determine if label is too early

                if label.timestamp <= start_time + min_start_window:
                    label_too_early += 1
                    del snapshots_by_hadm[hadm_id]
                    hadms_by_subject[subject_id].remove(hadm_id)
                    if not hadms_by_subject[subject_id]:
                        empty_subjects += 1
                        del hadms_by_subject[subject_id]
                    continue

                # Determine minimum elapsed time between last snapshot and the first label
                end_threshold = label.timestamp - min_pred_window  # type: datetime
                start_threshold = start_time - min_proceeding_window
                for timestamp, snapshot in list(snapshot_list.items()):
                    # Check if our snapshot has too few observations
                    if len(set(snapshot)) < FLAGS.min_snapshot_size:
                        logging.log_first_n(logging.DEBUG,
                                            'Filtering snapshot at %s of hadm %s for subject %s '
                                            'with only %d observations',
                                            3,
                                            timestamp, hadm_id, subject_id, len(set(snapshot)))
                        ss_too_small += 1
                        del snapshot_list[timestamp]
                        continue

                    # Check if our snapshot occurs more than a day before admission
                    if timestamp < start_threshold:
                        logging.log_first_n(logging.DEBUG,
                                            'Filtering snapshot at %s of hadm %s for subject %s '
                                            'which occurred %s before admission time %s',
                                            3,
                                            timestamp, hadm_id, subject_id, start_time - timestamp, start_time)
                        ss_too_early += 1
                        del snapshot_list[timestamp]
                        continue

                    # Check if our snapshot falls during or after the prediction window
                    if timestamp > end_threshold:
                        logging.log_first_n(logging.DEBUG,
                                            'Filtering snapshot at %s of hadm %s for subject %s '
                                            'which occurred %s before label time %s',
                                            3,
                                            timestamp, hadm_id, subject_id, label.timestamp - timestamp,
                                            label.timestamp)
                        ss_too_late += 1
                        del snapshot_list[timestamp]
                        continue

                # Check if our admission contains too few snapshots
                if len(snapshot_list) < FLAGS.min_chrono_length:
                    logging.log_first_n(logging.DEBUG,
                                        'Filtering chronology for hadm %s for subject %s '
                                        'which contained only %d snapshots',
                                        3,
                                        hadm_id, subject_id, len(snapshot_list))
                    chrono_too_short += 1
                    del snapshots_by_hadm[hadm_id]
                    hadms_by_subject[subject_id].remove(hadm_id)
                    if not hadms_by_subject[subject_id]:
                        empty_subjects += 1
                        del hadms_by_subject[subject_id]
                    continue

                # Check if our first snapshot occurs too late
                snapshot_timestamps = snapshot_list.keys()
                min_snapshot = min(snapshot_timestamps)
                if min_snapshot - start_time >= max_snapshot_delay:
                    logging.log_first_n(logging.DEBUG,
                                        'Filtering chronology for hadm %s for subject %s '
                                        'whose first snapshot occurred %s after admission %s',
                                        3,
                                        hadm_id, subject_id, min_snapshot - start_time, start_time)
                    chrono_too_late += 1
                    del snapshots_by_hadm[hadm_id]
                    hadms_by_subject[subject_id].remove(hadm_id)
                    if not hadms_by_subject[subject_id]:
                        empty_subjects += 1
                        del hadms_by_subject[subject_id]
                    continue

                # Check if our last snapshot occurs too early
                max_snapshot = max(snapshot_timestamps)
                if label.timestamp - max_snapshot > max_pred_window:
                    logging.log_first_n(logging.DEBUG,
                                        'Filtering chronology for hadm %s for subject %s '
                                        'whose last snapshot occurred %s before label %s',
                                        3,
                                        hadm_id, subject_id, label.timestamp - max_snapshot, label.timestamp)
                    chrono_window_too_wide += 1
                    del snapshots_by_hadm[hadm_id]
                    hadms_by_subject[subject_id].remove(hadm_id)
                    if not hadms_by_subject[subject_id]:
                        empty_subjects += 1
                        del hadms_by_subject[subject_id]
                    continue

                # Actually build the vocabulary
                if not lock_vocab:
                    for snapshot in snapshot_list.values():
                        for term in snapshot:
                            vocab.add_term(term)

        # Debugging info
        print('Filtered %d snapshots for having fewer than %d observation(s)'
              % (ss_too_small, FLAGS.min_snapshot_size))
        print('Filtered %d snapshots for occurring within %s hours of event'
              % (ss_too_late, FLAGS.min_pred_window))
        print('Filtered %d snapshots for occurring more than %s hours before admission'
              % (ss_too_early, 24))
        print('Filtered %d chronologies for having an event within %s hours of admission'
              % (label_too_early, FLAGS.min_start_window))
        print('Filtered %d chronologies for having fewer than %d snapshot(s)'
              % (chrono_too_short, FLAGS.min_chrono_length))
        print('Filtered %d chronologies in which the first snapshot occurred more than %d hours after admission'
              % (chrono_too_late, FLAGS.max_snapshot_delay))
        print('Filtered %d chronologies in which the prediction window exceeded %d hours'
              % (chrono_window_too_wide, FLAGS.max_pred_window))
        print('Filtered %d subjects for having empty chronologies'
              % empty_subjects)

        # Shrink vocabulary to consist of MAX_VOCAB_SIZE most frequently-occurring terms
        if not lock_vocab:
            vocab.resize(FLAGS.max_vocab_size)
            print('Created vocabulary of %d observations' % len(vocab))

        vocab_terms = set(vocab.terms)

        def get_words(it: typing.Sequence):
            seen = set()
            seen_add = seen.add
            return [o for o in it if o in vocab_terms and not (o in seen or seen_add(o))]

        # Build our chronologies
        chronologies = {}  # type: typing.Dict[str, Chronology]
        for subject_id, hadm_ids in hadms_by_subject.items():
            for hadm_id in hadm_ids:
                hadm = Admission(hadm_id, subject_id)
                snapshot_list = sorted(snapshots_by_hadm[hadm_id].items(),
                                       key=lambda t: t[0])  # type: typing.List[datetime]
                length = min(FLAGS.max_chrono_length, len(snapshot_list))
                chronology = Chronology(
                    start_time=start_time_by_hadm[hadm_id],
                    snapshots=[Snapshot(get_words(obs), ts, vocab, hadm=hadm) for ts, obs in snapshot_list[-length:]],
                    label=labels_by_hadm[hadm_id],
                    hadm=hadm
                )
                if len(chronology) < FLAGS.min_chrono_length:
                    print('Skipping chronology %s with only %d snapshots (from %d) (< %d)' %
                          (hadm_id, len(chronology), len(snapshots_by_hadm[hadm_id]), FLAGS.min_chrono_length))
                else:
                    chronologies[hadm_id] = chronology

        print('Created cohort with %d patients and %s chronologies' % (
            len(
                set(
                    subject_id for subject_id, hadms in hadms_by_subject.items()
                    for hadm in hadms if hadm in chronologies)),
            len(chronologies.values())))

        # Construct our cohort
        return Cohort(admissions=hadms_by_subject, chronologies=chronologies)

    def infer_negatives_from_positives(self, ratio=1.0):
        subjects_by_hadm = {hadm: subject for subject, hadms in self.admissions.items() for hadm in hadms}

        positive_entries = np.array([c for c in self.chronologies.items() if (c[1].class_label == 1 and len(c[1]) > 1)])
        num_positives = len(positive_entries)
        sample = np.random.choice(range(num_positives),
                                  int(num_positives * ratio),
                                  replace=False)  # type: typing.Sequence[int]

        hadms_by_subject = {subject: set(hadms) for subject, hadms in self.admissions.items()}
        chronologies_by_hadm = {hadm: chronology for hadm, chronology in self.chronologies.items()}
        for hadm_id, chronology in positive_entries[sample]:
            t = -np.random.randint(1, len(chronology))
            truncated_hadm_id = hadm_id + str(t) + 'T'
            subject_id = subjects_by_hadm[hadm_id]
            hadms_by_subject[subject_id].add(truncated_hadm_id)
            chronologies_by_hadm[truncated_hadm_id] = chronology.truncate_to(t)

        print('Inferring %d negative chronologies from %d positive examples' % (len(sample), num_positives))

        return Cohort(admissions=hadms_by_subject, chronologies=chronologies_by_hadm)

    def balance_classes(self, method='downsample'):
        positives = []
        negatives = []
        for chronology in self.chronologies.items():
            if chronology[1].class_label == 1:
                positives.append(chronology)
            else:
                negatives.append(chronology)

        if method == 'downsample':
            count = min(len(positives), len(negatives))
            replace = False
        elif method == 'upsample':
            count = max(len(positives), len(negatives))
            replace = True
        else:
            raise ValueError

        sampled_positives = np.random.choice(np.array(positives, dtype=np.dtype('object,object')),
                                             size=count, replace=replace)
        sampled_negatives = np.random.choice(np.array(negatives, dtype=np.dtype('object,object')),
                                             size=count, replace=replace)

        chronologies_by_hadm = {k: v for k, v in np.concatenate([sampled_positives, sampled_negatives], axis=0)}
        hadms_by_subject = {}
        for subject, hadms in self.admissions.items():
            for hadm in hadms:
                if hadm not in chronologies_by_hadm:
                    continue
                if subject not in hadms_by_subject:
                    hadms_by_subject[subject] = {hadm}
                else:
                    hadms_by_subject[subject].add(hadm)

        # print('Reduced positive chronologies from %d and negative from %d to %d.' % (
        #     len(positives),
        #     len(negatives),
        #     min_count
        # ))

        return Cohort(admissions=hadms_by_subject, chronologies=chronologies_by_hadm)

    def filter(self, func: typing.Callable[[str, str, Chronology], bool]):
        admissions = {}  # type: typing.Dict[str, typing.MutableSet[str]]
        chronologies = {}  # type: typing.Dict[str, Chronology]

        for subject_id, hadms in self.admissions.items():
            for hadm_id in hadms:
                chronology = self.chronologies[hadm_id]
                if func(subject_id, hadm_id, chronology):
                    admissions.setdefault(subject_id, set())
                    admissions[subject_id].add(hadm_id)
                    chronologies[hadm_id] = chronology

        return Cohort(admissions=admissions, chronologies=chronologies)

    def batched(self, batch_size: int, permute: bool = True, limit: typing.Optional[int] = None, distribute=True):
        # Shuffle chronologies
        if permute:
            chronologies = np.random.permutation(self.to_list())  # type: typing.Sequence[Chronology]
        else:
            chronologies = np.array(self.to_list())  # type: typing.Sequence[Chronology]

        # Determine the number of batches we will produce
        num_batches = len(chronologies) // batch_size
        if limit is not None:
            num_batches = min(num_batches, limit)

        # Throw away the last batch if its incomplete (shouldn't be an issue since we are permuting the order)
        truncate_length = num_batches * batch_size

        if distribute:
            import itertools

            def roundrobin(*iterables):
                """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
                # Recipe credited to George Sakkis
                num_active = len(iterables)
                nexts = itertools.cycle(iter(it).__next__ for it in iterables)
                while num_active:
                    try:
                        # noinspection PyShadowingBuiltins
                        for next in nexts:
                            yield next()
                    except StopIteration:
                        # Remove the iterator we just exhausted from the cycle.
                        num_active -= 1
                        nexts = itertools.cycle(itertools.islice(nexts, num_active))

            positives = []
            negatives = []
            for chronology in chronologies:
                if chronology.class_label == 1:
                    positives.append(chronology)
                else:
                    negatives.append(chronology)

            # Interleave examples based on labels, e.g., [p, n, p, n, n, n, ..] so we can truncate proportionally
            examples = np.array(list(itertools.islice(roundrobin(positives, negatives), truncate_length)))

            # Reshape based on number of classes, so we have one row for each class,
            # padded to the same length from the majority class
            examples = examples.reshape([-1, 2]).transpose()

            # Reshape based on number of batches, so we have one row per batch
            examples = examples.reshape([batch_size, num_batches]).transpose()

            # Split each row into a batch
            batches = np.split(examples, num_batches, axis=0)

            # Flatten each batch, and shuffle the order of chronologies within the batch
            batches = [np.random.permutation(batch.flatten()) for batch in batches]
        else:
            # Split chronologies into mini-batches
            # noinspection PyTypeChecker
            batches = np.split(chronologies[:truncate_length], num_batches, axis=0)

        # Encode each mini-batch as a ChronologyBatch object
        return [ChronologyBatch.from_chronologies(batch) for batch in batches]

    def make_classification(self):
        def get_instance(chronology: Chronology):
            meta = [chronology.hadm.patient_id, chronology.hadm.hadm_id]
            last = len(chronology) - 1
            x = np.concatenate([
                chronology.observation_matrix[last, :],
                chronology.delta_matrix[last, :]
            ])
            y = chronology.label.value
            return x, y, meta

        return map(np.asarray, zip(*map(get_instance, self.to_list())))


class ChronologyBatch(object):
    """Represents a batch of chronologies as zero-padded equal-length numpy ndarrays

    Attributes:
        batch_size(int): number of chronologies in this batch
        deltas: [batch_size x max_seq_len x delta_encoding_size] 3-d float32 numpy array encoding deltas for each
            snapshot in each chronology in this batch
        lengths: [batch_size] 1-d int32 numpy array encoding the non-zero-padded length (number of snapshots) for each
            chronology in this batch
        labels: [batch_size] 1-d int32 numpy array encoding the label for the final prediction for each chronology in
            this batch
        snapshots: [batch_size x max_seq_len x max_snap_size] 3-d int32 numpy array encoding the clinical observations
            recorded for each snapshot in each chronology in this batch
        snapshot_sizes: [batch_size x max_seq_len] 2-d int32 numpy array encoding the number of clinical observations
            recorded for each snapshot in each chronology in this batch
    """

    __slots__ = ['chronologies', 'batch_size', 'deltas', 'lengths', 'labels', 'snapshots', 'snapshot_sizes']

    def __init__(self, chronologies: typing.Sequence[Chronology],
                 batch_size: int,
                 snapshots: np.ndarray,
                 snapshot_sizes: np.ndarray,
                 deltas: np.ndarray,
                 labels: np.ndarray,
                 chrono_lengths: np.ndarray):
        self.chronologies = chronologies
        self.batch_size = batch_size
        self.deltas = deltas
        self.lengths = chrono_lengths
        # assert len(np.unique(labels)) > 1
        self.labels = labels
        self.snapshots = snapshots
        self.snapshot_sizes = snapshot_sizes

    @classmethod
    def from_chronologies(cls, chronologies: typing.Sequence[Chronology]):
        """Create a zero-padded/trimmed ChronologyBatch from a given batch of Chronology objects

        :param chronologies: batch of Chronology objects
        :return: new ChronologyBatch object
        """
        return cls(chronologies=chronologies,
                   batch_size=len(chronologies),
                   snapshots=np.stack([c.observation_matrix for c in chronologies], axis=0),
                   snapshot_sizes=np.stack([c.snapshot_size_array for c in chronologies], axis=0),
                   deltas=np.stack([c.delta_matrix for c in chronologies], axis=0),
                   labels=np.asarray([c.class_label for c in chronologies]),
                   chrono_lengths=np.asarray([len(c) for c in chronologies]),
                   )

    def perturb_labels(self):
        """Return a copy of this ChronologyBatch with the labels shuffled across chronologies"""
        return ChronologyBatch(chronologies=self.chronologies,
                               batch_size=self.batch_size,
                               snapshots=self.snapshots,
                               snapshot_sizes=self.snapshot_sizes,
                               deltas=self.deltas,
                               labels=np.random.permutation(self.labels),
                               chrono_lengths=self.lengths)

    def feed(self, model, training=False):
        """Feed this chronology batch to a CANTRIPModel object

        :param model: CANTRIP model which will be fed the data in this batch
        :type model: modeling.CANTRIPModel
        :param training: whether CANTRIP is in training or inference mode (i.e., whether to use dropout)
        :return: a feed dict for use with TensorFlow session.run()
        """
        return {model.observations: self.snapshots,
                model.deltas: self.deltas,
                model.snapshot_sizes: self.snapshot_sizes,
                model.seq_lengths: self.lengths,
                model.labels: self.labels,
                model.training: training}
