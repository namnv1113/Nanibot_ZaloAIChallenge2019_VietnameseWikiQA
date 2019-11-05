from tqdm import tqdm
from os.path import join
import json
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf


class InputExample(object):
    """A single training/test example in Zalo format for simple sequence classification."""

    def __init__(self, guid, question, text, title=None, label=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      question: string. The untokenized text of the first sequence.
      text: (Optional) string. The untokenized text of the second sequence
      label: (Optional) string. The label of the example. This should be
      title: (Optinal) string. The Wikipedia title where the text is retrieved
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.question = question
        self.text = text
        self.title = title
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class PaddingInputExample(object):
    """ Fake example so the num input examples is a multiple of the batch size.

        When running eval/predict on the TPU, we need to pad the number of examples
        to be a multiple of the batch size, because the TPU requires a fixed batch
        size. The alternative is to drop the last batch, which is bad because it means
        the entire output data won't be generated.

        We use this class instead of `None` because treating `None` as padding
        battches could cause silent errors.
    """


class ZaloDatasetProcessor(object):
    """ Base class to process & store input data for the Zalo AI Challenge dataset"""
    label_list = ['False', 'True']

    def __init__(self, dev_size=0.2, force_data_balance=False):
        """
        Parameters:
            dev_size: The size of the development set taken from the training set
            force_data_balance: Balance training data by truncate training instance whose label is overwhelming
        """
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.dev_size = dev_size
        self.force_data_balance = force_data_balance

    def load_from_path(self, dataset_path, encode='utf-8', train_filepath='train.json', test_filepath='test.json'):
        """
            Load data from file & store into memory

            Need to be called before preprocess(before write_all_to_tfrecords) is called
        """
        with open(join(dataset_path, train_filepath), 'r', encoding=encode) as train_file:
            # Get train data
            train_data = json.load(train_file)
            train_data_formatted = []
            train_data_formatted.extend(
                InputExample(guid=data_instance['id'],
                             question=data_instance['question'],
                             title=data_instance['title'],
                             text=data_instance['text'],
                             label=self.label_list[data_instance['label']]) for data_instance in tqdm(train_data)
            )

            # Divide into train set and dev set while maintain label ratio in each set
            if self.force_data_balance:
                min_label_data_size = min([len([data for data in train_data_formatted if data.label == label])]
                                          for label in self.label_list)
            for label in self.label_list:
                train_data_by_label = [data for data in train_data_formatted if data.label == label]
                if self.force_data_balance:
                    train_data_by_label = train_data_by_label[0:min_label_data_size]
                _train_data, _dev_data = train_test_split(train_data_by_label, shuffle=True, test_size=self.dev_size)
                self.train_data.extend(_train_data)
                self.dev_data.extend(_dev_data)

        with open(join(dataset_path, test_filepath), 'r', encoding=encode) as test_file:
            # Get test data
            test_data = json.load(test_file)
            for data_instance in tqdm(test_data):
                self.test_data.extend(
                    InputExample(guid=data_instance['__id__'] + '$' + paragraph_instance['id'],
                                 question=data_instance['question'],
                                 title=data_instance['title'],
                                 text=paragraph_instance['text'],
                                 label=None)
                    for paragraph_instance in data_instance['paragraphs']
                )

    def write_all_to_tfrecords(self, output_folder, tokenier, max_sequence_length,
                               train_filename='train.tfrecords', dev_filename='dev.tfrecords',
                               test_filename='test.tfrecords', encoding='utf-8'):
        """ Write data to tfrecords format, prepare for training/testing """

        # Extract features & store to file
        self._file_based_convert_examples_to_features(examples=self.train_data,
                                                      label_list=self.label_list,
                                                      max_seq_length=max_sequence_length,
                                                      tokenizer=tokenier,
                                                      output_file=join(output_folder, train_filename),
                                                      encoding=encoding)
        self._file_based_convert_examples_to_features(examples=self.dev_data,
                                                      label_list=self.label_list,
                                                      max_seq_length=max_sequence_length,
                                                      tokenizer=tokenier,
                                                      output_file=join(output_folder, dev_filename),
                                                      encoding=encoding)
        self._file_based_convert_examples_to_features(examples=self.test_data,
                                                      label_list=self.label_list,
                                                      max_seq_length=max_sequence_length,
                                                      tokenizer=tokenier,
                                                      output_file=join(output_folder, test_filename),
                                                      encoding=encoding)

    def _file_based_convert_examples_to_features(
            self, examples, label_list, max_seq_length, tokenizer, output_file, encoding='utf-8'):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.io.TFRecordWriter(output_file)

        for (ex_index, example) in tqdm(enumerate(examples)):
            feature = self._convert_single_example(example, label_list, max_seq_length, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            def create_bytes_feature(value, encoding=encoding):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
                value = value.encode(encoding)
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            features = collections.OrderedDict()
            features["guid"] = create_bytes_feature(feature.guid)
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])
            features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def _convert_single_example(self, example, label_list, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        # Return dummy features if fake example (for batch padding purpose)
        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                guid="",
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

        # Labels mapping
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        # Text tokenization
        tokens_a = tokenizer.tokenize(example.question)
        tokens_b = None
        if example.text:
            tokens_b = tokenizer.tokenize(example.text)

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        # Truncate text if total length of combinec input > max sequence length for the model
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] if example.label is not None else -1

        feature = InputFeatures(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """ Convert a set of `InputExample`s to a list of `InputFeatures`.
            Helper class for prediction
        """
        features = []
        for (ex_index, example) in enumerate(examples):
            feature = self._convert_single_example(example, label_list, max_seq_length, tokenizer)
            features.append(feature)
        return features
