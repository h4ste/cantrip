"""Contains classes and methods for parsing and representing chronologies.

Attributes:
    _UNK (str): Module level private-variable containing the unknown token symbol
"""

import typing

import numpy as np

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from mock_tqdm import trange, tqdm

from modeling import CANTRIPModel

# Symbol used to denote unknown or out-of-vocabulary words
_UNK = 'UNK'


class DeltaEncoder:

    def encode_delta(self, elapsed_seconds: int) -> typing.Sequence[typing.Union[int, float]]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


class DiscreteDeltaEncoder(DeltaEncoder):
    DELTA_BUCKETS = [1, 7, 30, 60, 90, 182, 365, 730]

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
        elapsed_days = elapsed_seconds / 60 / 60 / 24
        return [1 if elapsed_days <= bucket else 0 for bucket in DiscreteDeltaEncoder.DELTA_BUCKETS]

    @property
    def size(self):
        return 8


class TanhLogDeltaEncoder(DeltaEncoder):
    def encode_delta(self, elapsed_seconds: int):
        """Encode deltas into discrete buckets

        :param elapsed_seconds: number of seconds between this clinical snapshot and the previous
        :return: tanh(log(elapsed days + 1))
        """
        elapsed_days = elapsed_seconds / 60 / 60 / 24
        return np.tanh(np.log(elapsed_days + 1))

    @property
    def size(self):
        return 1


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

    def lookup_term_id_by_term(self, term: typing.Text) -> int:
        """Look-up term by term ID for given term.

        Returns unknown term symbol if this vocabulary was created with return_unk=True, otherwise raises KeyError

        :param term: term to lookup
        :return: term ID associated with given term in this vocabulary
        """
        if self.return_unk:
            return self.term_index.get(term, self.term_index.get[_UNK])
        else:
            return self.term_index[term]

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


class Chronology(object):
    """Representation of a single Chronology.

    Note: unlike the AMIA paper, this chronology includes an additional vector indicating the label (i.e., pneumonia)
    for each snapshot.

    Attributes:
        deltas: list/array containing elapsed time (in days) since previous snapshot where delta[0] = 0
        labels: list/array indicating the (binary) label (e.g., pneumonia) for each snapshot
        snapshots: matrix indicating the observations in each snapshot such that each column indicates a snapshot, and
        each row j indicates the j-th observation in that snapshot
    """

    def __init__(self, deltas, labels, snapshots):
        """ Creates a Chronology from the given deltas, labels, and snapshots.
        Note: no defensive copies are made!

        :param deltas: list/array containing elapsed time (in days) since previous snapshot where delta[0] = 0
        :param labels: list/array indicating the (binary) label (e.g., pneumonia) for each snapshot
        :param snapshots: matrix indicating the observations in each snapshot such that each column indicates a snapshot
            and each row j indicates the j-th observation in that snapshot
        """
        self.deltas = deltas
        self.labels = labels
        self.snapshots = snapshots

    def truncate(self, truncate_length):
        """Truncates this chronology to end at the given length (i.e., returns a temporal slice).

        :param truncate_length: the new length of this chronology (if negative, the new length will be
            len(chronology) - abs(truncate_length)
        :return: truncated view of this chronology
        """
        return Chronology(self.deltas[:truncate_length],
                          self.labels[:truncate_length],
                          self.snapshots[:truncate_length])

    def __len__(self):
        """Returns the number of snapshots in this chronology"""
        return len(self.deltas)


class Cohort(object):
    """Data structure containing the chronologies and vocabulary associated with a cohort of patients.

    Attributes:
        vocabulary (Vocabulary): the vocabulary of observations reported for this cohort
        patient_vocabulary (Vocabulary): a vocabulary mapping patients to contiguous integer IDs
    """

    def __init__(self, patient_chronologies=None, vocabulary=Vocabulary(), patient_vocabulary=Vocabulary()):
        """Create a (possibly-empty) Chronology.

        :param patient_chronologies: list of Chronology vectors associated with each patient. Index of each patient
            in this list should correspond to the ID for that patient in the patient_vocabulary
        :param vocabulary(Vocabulary): vocabulary of observations associated with this chronology
        :param patient_vocabulary(Vocabulary): vocabulary of patient IDs associated with this chronology
        """
        self.vocabulary = vocabulary
        if patient_chronologies is None:
            self._patient_chronologies = np.empty(0, dtype=np.object)
        else:
            self._patient_chronologies = np.asarray(patient_chronologies, dtype=np.object)
        self._patient_vocabulary = patient_vocabulary

    @classmethod
    def from_dict(cls, cohort, vocabulary):
        """Create a chronology from a given dict of patient id to chronologies and vocabulary of observations.

        :param cohort: a dict associating each patient in the cohort with a list of his or her chronologies
        :param vocabulary: vocabulary of observations documented in this cohort
        :return: Shiny new Cohort object
        """
        # Create a patient vocabulary to associate external patient IDs to internal, contiguous integer IDs
        patient_vocabulary = Vocabulary.from_terms(cohort.keys(),
                                                   add_unk=False, return_unk=False,
                                                   max_vocab_size=None)

        # Create a nested array associated each internal patient ID to his or her chronologies
        patients = []
        for subject_id, admissions in cohort.items():
            patient_vocabulary.add_term(subject_id)
            patients.append(admissions)

        return cls(patients, vocabulary, patient_vocabulary)

    @classmethod
    def from_disk(cls, patient_vectors, vocabulary, max_vocab_size=50000):
        """Load cohort from a given chronology file and vocabulary file or Vocabulary object.

        The format of this file is assumed to be as follows::

            [external_patient_id]\t[chronology]

        where each ``[chronology]`` is encoded as as sequence of snapshots::

            [[snapshot]\t...]

        and each ``[snapshot]`` is encoded as::

            [delta] [label] [observation IDs..]

        Note that snapshots are delimited by *spaces*, label must be 'true' or 'false', delta is represented in seconds
        since previous chronology, and observation IDs should be the IDs associated with the observation in the given
        vocabulary file.

        For example, the line::

            11100004a   0 false 1104 1105 2300 25001    86400 false 1104 2300   172800 true 1104 2300 3500

        would indicate that patient with external ID '11100004a' had a chronology including 3 snapshots
        where:

        * the first snapshot was negative for pneumonia, had a delta of 0, and contained only three clinical
          observations: those associated with vocabulary terms 1104, 1105, 2300, and 25001;
        * the second snapshot was negative for pneumonia, had a delta of 86400 seconds (1 day), and included only
          two clinical observations: 1104 and 2300
        * the third snapshot was positive for pneumonia, had a delta of 172800 seconds (2 days), and included only
          three clinical observations: 1104, 2300, and 3500

        The vocabulary file is assumed to be formatted as follows::

            [observation]\t[frequency]

        where the line number indicates the ID of the observation int he chronology (e.g., 1104), ``[observation]`` is a
        human-readable string describing the observation, and ``[frequency]`` is the frequency of that observation in
        the dataset (this value is only important if specifying a max_vocabulary_size as terms will be sorted in
        descending frequency before the cut-off is made)

        Note: as described in the AMIA paper , chronologies are truncated to terminate at the first positive label.
        Chronologies in which the first snapshot is positive or in which no snapshot is positive are discarded.

        :param patient_vectors: file containing for chronology vectors for the cohort
        :param vocabulary: file containing the vocabulary used to generate chronology vectors or an existing Vocabulary
            object
        :param max_vocab_size: maximum vocabulary size (if given, only the top max_vocabulary_size most frequent
            observations will be retained, and all other observations will mapped to the unknown term symbol "_UNK")
        :return: Cohort object
        """

        # Load vocabulary from file if we weren't already given a Vocabulary object
        if not isinstance(vocabulary, Vocabulary):
            vocabulary = Vocabulary.from_tsv(vocabulary, max_vocab_size=max_vocab_size)

        # We represent our cohort as a dictionary of external_patient_id to lists of one or more chronologies
        cohort = {}

        # Keep track of how many chronologies we have discarded
        filtered_chronologies = 0
        with open(patient_vectors, 'rt') as vector_file:
            for line in tqdm(vector_file.readlines(), desc='Loading chronologies'):
                # For whatever reason windows likes to add newlines to chronologies files when they are opened with
                # certain editors, so we need to strip them
                line = line.rstrip()

                # Load the subject id and list of snapshots
                # Each chronology is represented by [external_patient_id]\t[[snapshots]\t...]
                fields = line.split('\t')
                subject_id, snapshots = fields[0], fields[1:]

                # We sometimes represent chronologies as subject_id:visit_id
                delim = subject_id.find(':')
                if delim > -1:
                    subject_id = subject_id[:delim]

                # If this is a new subject, initialize his or her list of chronologies to be empty
                if subject_id not in cohort:
                    cohort[subject_id] = []

                # Parse snapshots to create a chronology
                deltas = []
                labels = []
                observations = []
                for snapshot in snapshots:
                    # Each snapshot is encoded as [delta] [label] [[observation ids] ...]
                    fields = snapshot.split(' ')
                    delta, label, observation_ids = fields[0], fields[1], fields[2:]
                    deltas.append(int(delta))

                    # Parse labels
                    label = label.lower()
                    if label == 'true':
                        labels.append(1)
                    elif label == 'false':
                        labels.append(0)
                    else:
                        raise ValueError('Encountered invalid label \'' + label + '\'')

                    # Encode observations  using our vocabulary (that is, associated any out-of-vocabulary observations
                    # with the unknown term symbol _UNK
                    observations.append([vocabulary.encode_term_id(int(word_id)) for word_id in observation_ids])

                    # Terminate after first positive label
                    if labels[-1] == 1:
                        break

                # We only care about patients who were
                # (1) not diagnosed in their first snapshot,
                # (2) had at least two snapshots, and
                # (3) were eventually diagnosed
                if labels[0] == 0 and len(deltas) > 2 and labels[-1] == 1:
                    # We discard the delta and label from the first snapshot (since we can't predict the first snapshot)
                    # and discard the observations in the final snapshot (since that is when we are trying to predict).
                    # Unlike the AMIA paper, we shift our deltas (and labels) within the chronology datastructure
                    # such that label[t] is the label we are trying to predict given snapshot[t] and delta[t]
                    # is the elapsed time from snapshot[t] to the label[t] prediction
                    cohort[subject_id].append(Chronology(deltas[1:], labels[1:], observations[:-1]))
                else:
                    filtered_chronologies += 1

        filtered_patients = 0
        for subject_id in list(cohort.keys()):
            # Any patients with zero chronologies after filtering are removed
            if len(cohort[subject_id]) < 1:
                filtered_patients += 1
                del cohort[subject_id]

        print('Loaded cohort of %d patients with %d visits (after filtering %d patients and %d visits)' % (
            len(cohort.keys()),  # Number of unique patients with chronologies
            sum([len(value) for value in cohort.values()]),  # Number of chronologies across all patients
            filtered_patients,  # Number of patients removed by chronology filtering
            filtered_chronologies)  # Number of chronologies removed across all patients
              )

        # Convert the dict of external id -> [chronology] list into a Chronology object
        return cls.from_dict(cohort, vocabulary)

    def __len__(self):
        """Return the number of chronologies in this cohort"""
        return len(self._patient_chronologies)

    def __getitem__(self, subject_id):
        """Given an iterable or single external id, return the sub-cohort of patients associated with that/those ids"""
        if isinstance(subject_id, typing.Iterable):
            indices = [self._patient_vocabulary.lookup_term_id_by_term(term) for term in subject_id]
            return Cohort(self._patient_chronologies[indices],
                          self.vocabulary,
                          self._patient_vocabulary)
        else:
            return Cohort(self._patient_chronologies[self._patient_vocabulary.lookup_term_by_term_id(subject_id)],
                          self.vocabulary,
                          self._patient_vocabulary)

    def patients(self):
        """Return a list of external patient IDs indicating the members of this cohort"""
        return self._patient_vocabulary.terms

    def chronologies(self):
        """Returns a flattened list of all chronologies in this cohort"""
        return [chronology for patient in self._patient_chronologies for chronology in patient]

    def items(self):
        """Returns tuples of subject IDs to list of chronologies (for dictionary-like iteration)"""
        for subject_id in self._patient_vocabulary.terms:
            yield (subject_id, self[subject_id])

    def balance_chronologies(self):
        """Return a view of this cohort with balanced chronologies.

        Specifically, return a copy of this cohort in which each patient has an equal number of positive and
        negative chronology examples for training.

        As described in the AMIA paper, we use each chronology as-is as a positive example and create a negative example
        by predicting the label associated with the second-to-last chronology (which is always negative)
        from the previous chronologies
        """
        balanced_chronologies = []
        for patient in self._patient_chronologies:
            # We interleave positive and negative examples
            balanced_visits = np.empty(2 * len(patient), dtype=np.object)
            # Even indicates are positive examples
            balanced_visits[0::2] = patient
            # Randomly truncate visits to end before the final snapshot to create negative examples
            negative_examples = [chronology.truncate(-np.random.randint(1, len(chronology))) for chronology in patient]
            # Odd indices are negative examples
            balanced_visits[1::2] = negative_examples
            balanced_chronologies.append(balanced_visits)
        return Cohort(balanced_chronologies, self.vocabulary, self._patient_vocabulary)

    def make_simple_classification(self, delta_encoder=None, final_only=False):
        """Represent this cohort as a data and label vector amenable to Sci-kit learn.

        :param delta_encoder: type of delta encoding to use
        :type delta_encoder: DeltaEncoder
        :param final_only: whether to convert each pair of successive chronologies to an example (default) or to only
            take the final positive and negative examples
        :return: an observation matrix X such that each row indicates a snapshot and each column indicates the
            presence of absence of that feature (with deltas encoded as an extra feature) and a label vector y
            indicating the label in the next snapshot
        """
        if not delta_encoder:
            delta_encoder = TanhLogDeltaEncoder()
        x = []
        y = []
        # Shuffle chronologies for science
        chronologies = np.random.permutation(self.chronologies())
        vocabulary_size = len(self.vocabulary)
        for chronology in chronologies:
            if final_only:
                slices = [-np.random.randint(1, len(chronology))]
            else:
                slices = range(len(chronology))

            for i in slices:
                # Convert sequence of observations into bag-of-observations vector
                bow = np.zeros(shape=vocabulary_size, dtype=np.int32)
                bow[chronology.docs[i]] = 1

                # Deltas and labels are already time-shifted when the cohort is created, so we don't need to shift them
                # here
                deltas = np.asarray([delta_encoder.encode_delta(chronology.deltas[i])])
                x.append(np.concatenate([deltas, bow], axis=0))
                y.append(chronology.labels[i])
        return np.asarray(x), np.asarray(y)

    def make_epoch_batches(self, batch_size, max_snapshot_size, max_chrono_length, limit=None,
                           delta_encoder=None):
        """Create shuffled, equal-size mini-batches from the entire cohort.

        :param batch_size: size of mini-batches (e.g., number of chronologies in each batch) :param
        max_snapshot_size: maximum number of observations to consider in each snapshot (will be trimmed/zero-padded)
        :param max_chrono_length: maximum number of snapshots to consider in each chronology (will be
        trimmed/zero-padded) :param limit: take only the first limit mini-batches, rather than all mini-batches for
        the cohort :param delta_encoder: encoder to use for encoding deltas :return: a list of ChronologyBatch objects
        """
        if not delta_encoder:
            delta_encoder = TanhLogDeltaEncoder()

        # Shuffle chronologies
        chronologies = np.random.permutation(self.chronologies())

        # Determine the number of batches we will produce
        num_batches = chronologies.shape[0] // batch_size
        if limit is not None:
            num_batches = min(num_batches, limit)

        # Throw away the last batch if its incomplete (shouldn't be an issue since we are permuting the order)
        truncate_length = num_batches * batch_size
        chronologies = chronologies[:truncate_length]

        # Split chronologies into mini-batches
        batches = np.split(chronologies, num_batches, axis=0)

        # Encode each minibatch as a ChronologyBatch object
        chronology_batches = [ChronologyBatch.from_chronologies(batch,
                                                                max_snapshot_size,
                                                                max_chrono_length,
                                                                delta_encoder) for batch in batches]

        return chronology_batches


class ChronologyBatch(object):
    """Represents a batch of chronologies as zero-padded equal-length numpy ndarrays

    Attributes:
        batch_size(int): number of chronologies in this batch
        deltas: [batch_size x max_seq_len x delta_encoding_size] 3-d float32 numpy array encoding deltas for each
            snapshot in each chronology in this batch
        seq_lens: [batch_size] 1-d int32 numpy array encoding the non-zero-padded length (number of snapshots) for each
            chronology in this batch
        labels: [batch_size] 1-d int32 numpy array encoding the label for the final prediction for each chronology in
            this batch
        snapshots: [batch_size x max_seq_len x max_snap_size] 3-d int32 numpy array encoding the clinical observations
            recorded for each snapshot in each chronology in this batch
        snapshot_sizes: [batch_size x max_seq_len] 2-d int32 numpy array encoding the number of clinical observations
            recorded for each snapshot in each chronology in this batch
    """

    def __init__(self, batch_size, deltas, seq_lens, labels, snapshots, snapshot_sizes):
        self.batch_size = batch_size
        self.deltas = deltas
        self.seq_lens = seq_lens
        self.labels = labels
        self.snapshots = snapshots
        self.snapshot_sizes = snapshot_sizes

    @classmethod
    def from_chronologies(cls, chronologies: typing.Sequence[Chronology], max_snapshot_size, max_chron_len,
                          delta_encoder):
        """Create a zero-padded/trimmed ChronologyBatch from a given batch of Chronology objects

        :param chronologies: batch of Chronology objects
        :param max_snapshot_size: maximum number of observations for each chronology (used to trim/zero-pad)
        :param max_chron_len: maximum number of snapshots for each chronology (used to trim/zero-pad)
        :param delta_encoder: delta encoder to use
        :type delta_encoder: DeltaEncoder
        :return: new ChronologyBatch object
        """
        # Infer batch size from the number of chronologies given to this method
        batch_size = len(chronologies)

        # Zero-pad everything to the indicated maximum sizes
        deltas = np.zeros([batch_size, max_chron_len, delta_encoder.size], np.float32)
        seq_lens = np.zeros(batch_size, np.int32)
        labels = np.zeros(batch_size, np.int32)
        snapshots = np.zeros([batch_size, max_chron_len, max_snapshot_size], np.int32)
        snapshot_sizes = np.ones([batch_size, max_chron_len], np.int32)

        for i, chronology in enumerate(chronologies):
            # Get the trimmed but non-padded length of this chronology
            seq_end = min(len(chronology.deltas), max_chron_len)
            seq_lens[i] = seq_end

            # Convert deltas using delta encoder
            for j, delta in enumerate(chronology.deltas[:seq_end]):
                deltas[i, j] = delta_encoder.encode_delta(delta)

            # Use final label as the prediction label for this chronology
            labels[i] = chronology.labels[seq_end - 1]

            # Convert sequences of observations to sequences of one-hots
            for j, snapshot in enumerate(chronology.snapshots[:seq_end]):
                # Get the trimmed but non-padded length of this snapshot
                snapshot_size = min(len(snapshot), max_snapshot_size)
                snapshot_sizes[i, j] = snapshot_size

                # Take the first snapshot_size observations
                snapshots[i, j, :snapshot_size] = snapshot[:snapshot_size]

        # Convert this into a ChronologyBatch object
        return cls(batch_size, deltas, seq_lens, labels, snapshots, snapshot_sizes)

    def perturb_labels(self):
        """Return a copy of this ChronologyBatch with the labels shuffled across chronologies"""
        return ChronologyBatch(self.batch_size,
                               self.deltas,
                               self.seq_lens,
                               np.random.permutation(self.labels),
                               self.snapshots,
                               self.snapshot_sizes)

    def feed(self, model: CANTRIPModel, training=False):
        """Feed this chronology batch to a CANTRIPModel object

        :param model: CANTRIP model which will be fed the data in this batch
        :param training: whether CANTRIP is in training or inference mode (i.e., whether to use dropout)
        :return: a feed dict for use with TensorFlow session.run()
        """
        return {model.observations: self.snapshots,
                model.deltas: self.deltas,
                model.snapshot_sizes: self.snapshot_sizes,
                model.seq_lengths: self.seq_lens,
                model.labels: self.labels,
                model.training: training}
