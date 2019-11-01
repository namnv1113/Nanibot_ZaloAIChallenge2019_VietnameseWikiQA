import tensorflow as tf
from bert import optimization, modeling
from os.path import join
import pandas as pd
from preprocess import InputExample, ZaloDatasetProcessor


class BertClassifierModel(object):
    def __init__(self, max_sequence_len, label_list,
                 learning_rate, batch_size, epochs, dropout_rate, warmup_proportion,
                 model_dir, save_checkpoint_steps, save_summary_steps, keep_checkpoint_max, bert_model_path, tokenizer,
                 train_file=None, evaluation_file=None, zalo_prediction_output_path=None, encoding='utf-8'):
        """ Constructor for TextCNN
            Parameters:
                max_sequence_len (int): Maximum length of input sequence
                label_list (list): List of labels to classify
                learning_rate (float): Initial learning rate
                batch_size (int): Batch size
                epochs (int): Train for how many epochs?
                dropout_rate (float): The dropout rate of the fully connected layer input (pretrained BERT output)
                warmup_proportion (float): The amount of training steps is used for warmup
                model_dir (string): Folder path to store the model
                save_checkpoint_steps (int): The number of steps to save checkpoints
                save_summary_steps (int): The number of steps to save summary
                keep_checkpoint_max (int): The maximum number of checkpoints to keep
                bert_model_path (string): The path to BERT pretrained model
                tokenizer (FullTokenier): BERT tokenizer for data processing
                train_file (string): The path to the tfrecords file that is used for training
                evaluation_file (string): The path to the tfrecords file that is used for evaluation
                zalo_prediction_output_path (string): The default output file path for the Zalo submission predict file
                encoding (string): The encoding used in the dataset
        """
        # Variable initialization
        self.max_sequence_len = max_sequence_len
        self.labels_list = label_list
        self.num_labels = len(self.labels_list)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.train_file = train_file
        self.evaluation_file = evaluation_file
        # self.bert_model_path = bert_model_path
        self.bert_configfile = join(bert_model_path, 'bert_config.json')
        self.init_checkpoint = join(bert_model_path, 'bert_model.ckpt')
        self.tokenizer = tokenizer
        self.zalo_prediction_output_path = zalo_prediction_output_path
        self.encoding = encoding

        # Specify outpit directory and number of checkpoint steps to save
        self.run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoint_steps,
            keep_checkpoint_max=keep_checkpoint_max)

        # Specify training steps
        if self.train_file:
            self.num_train_steps = int(sum(1 for _ in tf.python_io.tf_record_iterator(train_file)) /
                                       self.batch_size * self.epochs)
            # self.num_train_steps = int(sum(1 for _ in tf.data.TFRecordDataset(train_file)) /
            #                            self.batch_size * self.epochs)
            self.num_warmup_steps = int(self.num_train_steps * warmup_proportion)

        # Create the Estimator
        self.classifier = tf.estimator.Estimator(model_fn=self.model_fn_builder(),
                                                 config=self.run_config,
                                                 params={"batch_size": self.batch_size})

    def create_model(self, is_training, input_ids, input_mask, segment_ids, labels):
        """ Create a classification model based on BERT """
        bert_module = modeling.BertModel(
            config=modeling.BertConfig.from_json_file(self.bert_configfile),
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False,  # True if use TPU
        )

        # Use model.get_pooled_output() for classification tasks on an entire sentence.
        # Use model.get_sequence_output() for token-level output.
        output_layer = bert_module.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.compat.v1.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        with tf.compat.v1.variable_scope("loss"):
            # Dropout helps prevent overfitting
            if is_training:
                output_layer = tf.nn.dropout(output_layer, rate=self.dropout_rate)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            predicted_labels = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, predicted_labels, log_probs, probabilities

    def model_fn_builder(self):
        """Returns `model_fn` closure for Estimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for Estimator."""
            guid = features["guid"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_real_example = None
            if "is_real_example" in features:
                is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            (total_loss, _, predicted_labels, log_probs, probabilities) = self.create_model(
                is_training, input_ids, input_mask, segment_ids, label_ids)

            (assignment_map, initialized_variable_names) \
                = modeling.get_assignment_map_from_checkpoint(tf.trainable_variables(), self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)

                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=total_loss,
                                                  train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(label_ids, predicted_labels, is_real_example):
                    """ Calculate evaluation metrics  """
                    accuracy = tf.metrics.accuracy(labels=label_ids,
                                                   predictions=predicted_labels,
                                                   weights=is_real_example)
                    f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                    recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                    precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                    return {
                        "accuracy": accuracy,
                        "f1_score": f1_score,
                        "recall": recall,
                        "precision": precision
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels, is_real_example)
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=total_loss,
                                                  eval_metric_ops=eval_metrics)
            elif mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    "guid": guid,
                    'input_texts': input_ids,
                    'prediction': predicted_labels,
                    'probabilities': probabilities,
                    'labels': label_ids
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            else:
                raise ValueError(
                    "Only TRAIN, EVAL and PREDICT modes are supported: %s" % mode)

        # Return the actual model function in the closure
        return model_fn

    def _file_based_input_fn_builder(self, input_file, is_training, drop_remainder=False):
        """Creates an `input_fn` closure to be passed to Estimator - Used for tfrecord files """

        name_to_features = {
            "guid": tf.io.FixedLenFeature([], tf.string),
            "input_ids": tf.io.FixedLenFeature([self.max_sequence_len], tf.int64),
            "input_mask": tf.io.FixedLenFeature([self.max_sequence_len], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([self.max_sequence_len], tf.int64),
            "label_ids": tf.io.FixedLenFeature([], tf.int64),
            "is_real_example": tf.io.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, dtype=tf.int32)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.map(map_func=lambda record: _decode_record(record, name_to_features))\
                .batch(batch_size=batch_size, drop_remainder=drop_remainder)

            return d

        return input_fn

    def _input_fn_builder(self, input_features, is_training, drop_remainder=False):
        """Creates an `input_fn` closure to be passed to Estimator - Used for predicting """
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []

        for feature in input_features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(input_features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, self.max_sequence_len],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, self.max_sequence_len],
                        dtype=tf.int32),
                "segment_ids":
                    tf.constant(
                        all_segment_ids,
                        shape=[num_examples, self.max_sequence_len],
                        dtype=tf.int32),
                "label_ids":
                    tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
            })

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d

        return input_fn

    def train(self):
        """ Training data based on predefined training record (train set) """
        if not self.train_file:
            return

        train_input_fn = self._file_based_input_fn_builder(
            input_file=self.train_file,
            is_training=True,
            drop_remainder=True
        )

        self.classifier.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def eval(self):
        """ Evaluate model based on predefined evaluation record (test set)
            Return
                eval_results (dictionary): Evaluation results (accuracy, f1, precision & recall)
        """
        if not self.evaluation_file:
            return

        eval_input_fn = self._file_based_input_fn_builder(
            input_file=self.evaluation_file,
            is_training=False,
            drop_remainder=False
        )

        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        return eval_results

    def predict(self, qas):
        """ Get a prediction for the input qa pair
            Parameters:
                qas (list of tuple): A list of question-paragraph pairs
            Return
                is_answers (list): Corresponding to each qa pairs, is the paragraph contains the answer for the question
        """
        sentences_formatted = [InputExample(guid="",
                                            question=qa[0],
                                            text=qa[1],
                                            label=None) for qa in qas]
        sentences_features = ZaloDatasetProcessor().convert_examples_to_features(examples=sentences_formatted,
                                                                                 label_list=self.labels_list,
                                                                                 max_seq_length=self.max_sequence_len,
                                                                                 tokenizer=self.tokenizer)

        predict_input_fn = self._input_fn_builder(
            input_features=sentences_features,
            is_training=False,
            drop_remainder=False
        )

        predict_results = self.classifier.predict(input_fn=predict_input_fn, yield_single_examples=False)

        results = []
        for index, prediction in enumerate(predict_results):
            results.append({
                "input_question": qas[index][0],
                "input_paragraph": qas[index][1],
                "prediction": self.labels_list[prediction["prediction"][index]],
                "probabilities": prediction["probabilities"][index][prediction["prediction"][index]]
            })

        return results

    def predict_from_eval_file(self, test_file, output_file=None):
        """ Get prediction from predefined evaluation record (test set)
            Parameters:
                test_file (str): The path to the tfrecords (preprocessed) file that need predicting
                output_file (str): Desired path to store the result
            Return:
                results (Dataframe): Prediction results
        """
        if not test_file:
            return

        predict_input_fn = self._file_based_input_fn_builder(
            input_file=test_file,
            is_training=False,
            drop_remainder=False
        )

        predict_results = self.classifier.predict(input_fn=predict_input_fn)

        results = []
        for i, prediction in enumerate(predict_results):
            _dict = {
                "guid": prediction["guid"].decode(self.encoding),
                "input_text": self.tokenizer.convert_ids_to_tokens(prediction["input_texts"]),
                "prediction": self.labels_list[prediction["prediction"]],
                # "label": self.labels_list[prediction["labels"]],
                "probabilities": prediction["probabilities"][prediction["prediction"]]
            }
            results.append(_dict)

        if output_file:
            trueonly_results = []
            for result in results:
                if result['prediction'] == 'True':
                    result_test_id = result['guid'].split('$')[0]
                    result_answer = result['guid'].split('$')[1]
                    trueonly_results.append({
                        "test_id": result_test_id,
                        "answer": result_answer
                    })

            trueonly_results_dataframe = pd.DataFrame.from_records(trueonly_results)
            trueonly_results_dataframe.to_csv(path_or_buf=self.zalo_prediction_output_path, encoding=self.encoding,
                                              index=False)

        return pd.DataFrame.from_records(results)
