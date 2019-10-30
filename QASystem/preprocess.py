import tensorflow as tf
from tqdm import tqdm
import collections
from os.path import join
import json
from sklearn.model_selection import train_test_split

from bert import run_classifier, tokenization


class InputData(object):
    """ Represent a single input example (Zalo format)"""

    def __init__(self, qid, question, title, text, label=None):
        self.qid = qid
        self.question = question
        self.title = title
        self.text = text
        self.label = label


class ZaloDatasetProcessor(object):
    """ Base class to process & store input data for the Zalo AI Challenge dataset"""
    label_list = [True, False]

    def __init__(self):
        self.train_data = []
        self.dev_data = []
        self.test_data = []

    def load_from_path(self, dataset_path, encode='utf-8', train_filepath='train.json', test_filepath='test.json'):
        """
            Load data from file & store into memory

            Need to be called before preprocess(before write_all_to_tfrecords) is called
        """
        with open(join(dataset_path, train_filepath), 'r', encoding=encode) as train_file:
            train_data = json.load(train_file)
            train_data_formatted = []
            train_data_formatted.extend(
                InputData(qid=data_instance['id'],
                          question=data_instance['question'],
                          title=data_instance['title'],
                          text=data_instance['text'],
                          label=data_instance['label']) for data_instance in tqdm(train_data)
            )
            self.train_data, self.dev_data = train_test_split(train_data_formatted, shuffle=True, test_size=0.2)

        with open(join(dataset_path, test_filepath), 'r', encoding=encode) as test_file:
            test_data = json.load(test_file)
            for data_instance in tqdm(test_data):
                self.test_data.extend(
                    InputData(qid=data_instance['__id__'] + '$' + paragraph_instance['id'],
                              question=data_instance['question'],
                              title=data_instance['title'],
                              text=paragraph_instance['text'],
                              label=True)
                    for paragraph_instance in data_instance['paragraphs']
                )

    def write_all_to_tfrecords(self, output_folder, bert_pretrained_model_path, max_sequence_length, do_lowercase=True,
                               train_filename='train.tfrecords', dev_filename='dev.tfrecords',
                               test_filename='test.tfrecords', bert_vocab_file='vocab.txt'):
        """ Write data to tfrecords format, prepare for training/testing """
        # Convert data to BERT-based
        train_data_bert = [run_classifier.InputExample(guid=data_instance.qid,
                                                       text_a=data_instance.question,
                                                       text_b=data_instance.text,
                                                       label=data_instance.label) for data_instance in self.train_data]
        dev_data_bert = [run_classifier.InputExample(guid=data_instance.qid,
                                                     text_a=data_instance.question,
                                                     text_b=data_instance.text,
                                                     label=data_instance.label) for data_instance in self.dev_data]
        test_data_bert = [run_classifier.InputExample(guid=data_instance.qid,
                                                      text_a=data_instance.question,
                                                      text_b=data_instance.text,
                                                      label=data_instance.label) for data_instance in self.test_data]

        tokenier = tokenization.FullTokenizer(vocab_file=join(bert_pretrained_model_path, bert_vocab_file),
                                              do_lower_case=do_lowercase)

        # Extract features & store to file
        run_classifier.file_based_convert_examples_to_features(examples=train_data_bert,
                                                               label_list=self.label_list,
                                                               max_seq_length=max_sequence_length,
                                                               tokenizer=tokenier,
                                                               output_file=join(output_folder, train_filename))
        run_classifier.file_based_convert_examples_to_features(examples=dev_data_bert,
                                                               label_list=self.label_list,
                                                               max_seq_length=max_sequence_length,
                                                               tokenizer=tokenier,
                                                               output_file=join(output_folder, dev_filename))
        run_classifier.file_based_convert_examples_to_features(examples=test_data_bert,
                                                               label_list=self.label_list,
                                                               max_seq_length=max_sequence_length,
                                                               tokenizer=tokenier,
                                                               output_file=join(output_folder, test_filename))
