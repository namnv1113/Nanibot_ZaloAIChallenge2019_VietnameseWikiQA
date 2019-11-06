import tensorflow as tf
from bert import optimization, modeling
from os.path import join
import pandas as pd
from preprocess import InputExample, ZaloDatasetProcessor


class BertClassifierModel(object):
    def __init__(self, max_sequence_len, label_list,
                 learning_rate, batch_size, epochs, dropout_rate, warmup_proportion,
                 model_dir, save_checkpoint_steps, save_summary_steps, keep_checkpoint_max, bert_model_path, tokenizer,
                 textcnn_filter_sizes, textcnn_num_filters,
                 train_file=None, evaluation_file=None,
                 zalo_prediction_output_path=None, encoding='utf-8'):
        """ Constructor for BERT model for classification (BERT + TextCNN)
            :parameter max_sequence_len (int): Maximum length of input sequence
            :parameter label_list (list): List of labels to classify
            :parameter learning_rate (float): Initial learning rate
            :parameter batch_size (int): Batch size
            :parameter epochs (int): Train for how many epochs?
            :parameter dropout_rate (float): The dropout rate of the fully connected layer input
            :parameter warmup_proportion (float): The amount of training steps is used for warmup
            :parameter model_dir (string): Folder path to store the model
            :parameter save_checkpoint_steps (int): The number of steps to save checkpoints
            :parameter save_summary_steps (int): The number of steps to save summary
            :parameter keep_checkpoint_max (int): The maximum number of checkpoints to keep
            :parameter bert_model_path (string): The path to BERT pretrained model
            :parameter tokenizer (FullTokenier): BERT tokenizer for data processing
            :parameter textcnn_filter_sizes: Stride of each filter in each convolution layer
                    The number of <filter_size> also indicates the number of different (parallel) convolution layers
            :parameter textcnn_num_filters: The number of filters in each convolution layers
            :parameter train_file (string): The path to the tfrecords file that is used for training
            :parameter evaluation_file (string): The path to the tfrecords file that is used for evaluation
            :parameter zalo_prediction_output_path (string): The default output file path for the Zalo submission file
            :parameter encoding (string): The encoding used in the dataset
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
        self.bert_configfile = join(bert_model_path, 'bert_config.json')
        self.init_checkpoint = join(bert_model_path, 'bert_model.ckpt')
        self.tokenizer = tokenizer
        self.zalo_prediction_output_path = zalo_prediction_output_path
        self.encoding = encoding

        self.textcnn_filter_sizes = textcnn_filter_sizes
        self.textcnn_num_filters = textcnn_num_filters

        # Specify outpit directory and number of checkpoint steps to save
        self.run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoint_steps,
            keep_checkpoint_max=keep_checkpoint_max)

        # Specify training steps
        if self.train_file:
            self.num_train_steps = int(sum(1 for _ in tf.python_io.tf_record_iterator(train_file))
                                       / self.batch_size * self.epochs)
            # self.num_train_steps = int(sum(1 for _ in tf.data.TFRecordDataset(train_file)) /
            #                            self.batch_size * self.epochs)
            self.num_warmup_steps = int(self.num_train_steps * warmup_proportion)

        if self.evaluation_file:
            self.num_eval_steps = int(sum(1 for _ in tf.python_io.tf_record_iterator(evaluation_file))
                                      / self.batch_size)

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

        def _convolution_layer(input_x, output_size, width, stride):
            """ Convolution layer (1D convolution followed by relu)
                Convolution 1D
                :parameter input_x: A tensor of embedded tokens with shape [batch_size, seq_length, embedding_size]
                :parameter output_size: The number of feature maps we'd like to calculate (# of filters)
                :parameter width: The filter width
                :parameter stride: The stride
                :return: Convolution output tensor with shape [batch_size, 1, max_length, output_size]
            """
            # Convolution 1D
            input_size = input_x.get_shape()[-1]  # How many channels on the input (The size of our embedding)
            # Make our text an image of height 1
            input_x = tf.expand_dims(input_x, axis=1)  # Change the shape to [batch, H, W, channels]
            # Filter shape will be [filter H = 1, filter W, Channels In, Channel Out]
            _filter = tf.get_variable("conv_filter", shape=[1, width, input_size, output_size])
            # Run the convolution as if this were an image
            convolved = tf.nn.conv2d(input_x, filter=_filter, strides=[1, 1, stride, 1], padding="VALID")

            # Pass through Relu
            b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="conv_bias")
            return tf.nn.relu(convolved + b, name="relu")

        # Use model.get_pooled_output() for classification tasks on an entire sentence.
        # Use model.get_sequence_output() for token-level output.
        output_layer = bert_module.get_sequence_output()

        # Create a TextCNN on top of BERT for classification

        # Convolution layers
        pooled_conv_outputs = []
        for filter_size in self.textcnn_filter_sizes:
            with tf.compat.v1.variable_scope('conv_{}'.format(filter_size)):
                branch = _convolution_layer(output_layer, output_size=self.textcnn_num_filters,
                                            width=filter_size, stride=1)
                branch = tf.nn.max_pool(branch, ksize=[1, 1, self.max_sequence_len - filter_size + 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="max_pool")
                pooled_conv_outputs.append(branch)

        # Combined all pooled features
        with tf.compat.v1.variable_scope("merge"):
            output_layer = tf.concat(pooled_conv_outputs, 3, name='concat')
            output_layer = tf.reshape(output_layer, [-1, self.textcnn_num_filters * len(self.textcnn_filter_sizes)])

        # Passed the convoluted features into a fully connected layer
        hidden_size = output_layer.shape[-1].value
        with tf.compat.v1.variable_scope("fully_connected"):
            # Dropout helps prevent overfitting
            if is_training:
                output_layer = tf.nn.dropout(output_layer, rate=self.dropout_rate)
            fc_weights = tf.compat.v1.get_variable("fc_weights", [self.num_labels, hidden_size],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            fc_bias = tf.compat.v1.get_variable("fc_bias", [self.num_labels],
                                                initializer=tf.zeros_initializer())

            logits = tf.matmul(output_layer, fc_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, fc_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            predicted_labels = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        # If we're train/eval, compute loss between predicted and actual label
        with tf.compat.v1.variable_scope("fully_connected_loss"):
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, predicted_labels, log_probs, probabilities

    def model_fn_builder(self):
        """ Returns `model_fn` closure for Estimator. """

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for Estimator."""
            # Get input features
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

            # Pass through model
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            (total_loss, _, predicted_labels, log_probs, probabilities) = self.create_model(
                is_training, input_ids, input_mask, segment_ids, label_ids)

            (assignment_map, initialized_variable_names) \
                = modeling.get_assignment_map_from_checkpoint(tf.trainable_variables(), self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # Optimize/Predict
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)

                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=total_loss,
                                                  train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(label_ids, predicted_labels, is_real_example):
                    """ Calculate evaluation metrics  """
                    accuracy = tf.compat.v1.metrics.accuracy(labels=label_ids,
                                                             predictions=predicted_labels,
                                                             weights=is_real_example)
                    f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                    recall = tf.compat.v1.metrics.recall(label_ids, predicted_labels)
                    precision = tf.compat.v1.metrics.precision(label_ids, predicted_labels)
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
        """ Creates an `input_fn` closure to be passed to Estimator - Used for tfrecord files
            :parameter input_file: The path to a TFRecord file (preprocessed file)
            :parameter is_training: Is the input_file used for training?
            :parameter drop_remainder: Should drop the last batch where there is not enough data to form a batch
            :returns A function to generate input data to the model
        """

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

            d = d.map(map_func=lambda record: _decode_record(record, name_to_features)) \
                .batch(batch_size=batch_size, drop_remainder=drop_remainder)

            return d

        return input_fn

    def _input_fn_builder(self, input_features, is_training, drop_remainder=False):
        """ Creates an `input_fn` closure to be passed to Estimator - Used for predicting
            :parameter input_features: List of processed input data (InputFeatures)
            :parameter is_training: Is the input_features used for training?
            :parameter drop_remainder: Should drop the last batch where there is not enough data to form a batch
            :returns A function to generate input data to the model
        """
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
        """ Training model based on predefined training record (train set) """
        if not self.train_file:
            return

        train_input_fn = self._file_based_input_fn_builder(
            input_file=self.train_file,
            is_training=True,
            drop_remainder=True
        )

        self.classifier.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def train_and_eval(self):
        """ Training & evaluate model
            :returns eval_results (dictionary): Evaluation results (accuracy, f1, precision & recall)
        """
        if not self.train_file or not self.evaluation_file:
            return

        train_input_fn = self._file_based_input_fn_builder(
            input_file=self.train_file,
            is_training=True,
            drop_remainder=True
        )

        eval_input_fn = self._file_based_input_fn_builder(
            input_file=self.evaluation_file,
            is_training=False,
            drop_remainder=False
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=self.num_train_steps,
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=self.num_eval_steps,
        )

        tf.estimator.train_and_evaluate(self.classifier, train_spec, eval_spec)
        return self.eval()

    def eval(self):
        """ Evaluate model based on predefined evaluation record (development set)
            :returns eval_results (dictionary): Evaluation results (accuracy, f1, precision & recall)
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
        """ Get a prediction for each input qa pairs
            :parameter qas: (list of tuple) A list of question-paragraph pairs
            :returns is_answers: (list) Corresponding to each qa pairs, is the paragraph contains the answer for the question
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
            :parameter test_file: The path to the tfrecords (preprocessed) file that need predicting
            :parameter output_file: Desired path to store the result
            :returns results (Dataframe): Prediction results
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
