# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling
import tokenization
import tensorflow as tf
from run_squad import model_fn_builder, input_fn_builder, convert_examples_to_features, SquadExample, FeatureWriter, get_final_text, _get_best_indexes, _compute_softmax, RawResult

flags = tf.flags
FLAGS = flags.FLAGS

# Prediction model
class Inference(object):
	def validate_flags_or_throw(self, bert_config):
		if FLAGS.max_seq_length > bert_config.max_position_embeddings:
			raise ValueError(
					"Cannot use sequence length %d because the BERT model "
					"was only trained up to sequence length %d" %
					(FLAGS.max_seq_length, bert_config.max_position_embeddings))

		if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
			raise ValueError(
					"The max_seq_length (%d) must be greater than max_query_length "
					"(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))
	
	def __init__(self, model):
		# Define required variables

		if (model == 'en'):
			self.model = 'en'
			bert_config_file = './model/base/bert_config.json'
			vocab_file = './model/base/vocab.txt'
			init_checkpoint = './model/base/bert_model.ckpt'
			self.output_dir = './finetuned/eng'
		elif (model == 'vi'):
			self.model = 'vi'
			bert_config_file = './model/multi_cased/bert_config.json'
			vocab_file = './model/multi_cased/vocab.txt'
			init_checkpoint = './model/multi_cased/bert_model.ckpt'
			self.output_dir = './finetuned/vi'
		elif (model == 'vi_uit'):
			self.model = 'vi'
			bert_config_file = './model/multi_cased/bert_config.json'
			vocab_file = './model/multi_cased/vocab.txt'
			init_checkpoint = './model/multi_cased/bert_model.ckpt'
			self.output_dir = './finetuned/vi_uit'
		
		
		# Set up model for prediction
		tf.logging.set_verbosity(tf.logging.INFO)
		bert_config = modeling.BertConfig.from_json_file(bert_config_file)
		self.validate_flags_or_throw(bert_config)
		tf.gfile.MakeDirs(self.output_dir)
		
		self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

		tpu_cluster_resolver = None
		if FLAGS.use_tpu and FLAGS.tpu_name:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

		is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
		run_config = tf.contrib.tpu.RunConfig(
				cluster=tpu_cluster_resolver,
				master=FLAGS.master,
				model_dir=self.output_dir,
				save_checkpoints_steps=FLAGS.save_checkpoints_steps,
				tpu_config=tf.contrib.tpu.TPUConfig(
						iterations_per_loop=FLAGS.iterations_per_loop,
						num_shards=FLAGS.num_tpu_cores,
						per_host_input_for_training=is_per_host))

		model_fn = model_fn_builder(
				bert_config=bert_config,
				init_checkpoint=init_checkpoint,
				learning_rate=FLAGS.learning_rate,
				num_train_steps=None,
				num_warmup_steps=None,
				use_tpu=FLAGS.use_tpu,
				use_one_hot_embeddings=FLAGS.use_tpu)

		# If TPU is not available, this will fall back to normal Estimator on CPU
		# or GPU.
		self.estimator = tf.contrib.tpu.TPUEstimator(
				use_tpu=FLAGS.use_tpu,
				model_fn=model_fn,
				config=run_config,
				train_batch_size=FLAGS.train_batch_size,
				predict_batch_size=FLAGS.predict_batch_size)

	def process_example(self, data):
		"""Create a SquadExample from the paragraph and question"""
		def is_whitespace(c):
			return (c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F)
		qas_id = 0
		examples = []
		for temp in data:
			context = temp[0]
			question = temp[1]
			doc_tokens = []
			prev_is_whitespace = True
			for c in context:
				if is_whitespace(c):
					prev_is_whitespace = True
				else:
					if prev_is_whitespace:
						doc_tokens.append(c)
					else:
						doc_tokens[-1] += c
					prev_is_whitespace = False

			examples.append(SquadExample(qas_id=qas_id, question_text=question, doc_tokens=doc_tokens))
			qas_id = qas_id + 1
		
		return examples

	def predict(self, all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case):
		example_index_to_features = collections.defaultdict(list)
		for feature in all_features:
			example_index_to_features[feature.example_index].append(feature)

		unique_id_to_result = {}
		for result in all_results:
			unique_id_to_result[result.unique_id] = result

		_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
				"PrelimPrediction",
				["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

		all_predictions = collections.OrderedDict()
		all_nbest_json = collections.OrderedDict()

		for (example_index, example) in enumerate(all_examples):
			features = example_index_to_features[example_index]

			prelim_predictions = []

			for (feature_index, feature) in enumerate(features):
				result = unique_id_to_result[feature.unique_id]
				start_indexes = _get_best_indexes(result.start_logits, n_best_size)
				end_indexes = _get_best_indexes(result.end_logits, n_best_size)
				# if we could have irrelevant answers, get the min score of irrelevant
				
				for start_index in start_indexes:
					for end_index in end_indexes:
						# We could hypothetically create invalid predictions, e.g., predict
						# that the start of the span is in the question. We throw out all
						# invalid predictions.
						if 	   (start_index >= len(feature.tokens)
							or 	end_index >= len(feature.tokens)
							or  start_index not in feature.token_to_orig_map
							or  end_index not in feature.token_to_orig_map
							or not feature.token_is_max_context.get(start_index, False)
							or end_index < start_index):
							continue

						length = end_index - start_index + 1
						if length > max_answer_length:
							continue

						prelim_predictions.append(
								_PrelimPrediction(
										feature_index=feature_index,
										start_index=start_index,
										end_index=end_index,
										start_logit=result.start_logits[start_index],
										end_logit=result.end_logits[end_index]))

			prelim_predictions = sorted(
					prelim_predictions,
					key=lambda x: (x.start_logit + x.end_logit),
					reverse=True)

			_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
					"NbestPrediction", ["text", "start_logit", "end_logit"])

			seen_predictions = {}
			nbest = []
			for pred in prelim_predictions:
				if len(nbest) >= n_best_size:
					break
				feature = features[pred.feature_index]
				if pred.start_index > 0:  # this is a non-null prediction
					tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
					orig_doc_start = feature.token_to_orig_map[pred.start_index]
					orig_doc_end = feature.token_to_orig_map[pred.end_index]
					orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
					tok_text = " ".join(tok_tokens)

					# De-tokenize WordPieces that have been split off.
					tok_text = tok_text.replace(" ##", "")
					tok_text = tok_text.replace("##", "")

					# Clean whitespace
					tok_text = tok_text.strip()
					tok_text = " ".join(tok_text.split())
					orig_text = " ".join(orig_tokens)

					final_text = get_final_text(tok_text, orig_text, do_lower_case)
					if final_text in seen_predictions:
						continue

					seen_predictions[final_text] = True
				else:
					final_text = ""
					seen_predictions[final_text] = True

				nbest.append(
						_NbestPrediction(
								text=final_text,
								start_logit=pred.start_logit,
								end_logit=pred.end_logit))

			# In very rare edge cases we could have no valid predictions. So we
			# just create a nonce prediction in this case to avoid failure.
			if not nbest:
				nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

			assert len(nbest) >= 1

			total_scores = []
			best_non_null_entry = None
			for entry in nbest:
				total_scores.append(entry.start_logit + entry.end_logit)
				if not best_non_null_entry:
					if entry.text:
						best_non_null_entry = entry

			probs = _compute_softmax(total_scores)

			nbest_json = []
			for (i, entry) in enumerate(nbest):
				output = collections.OrderedDict()
				output["text"] = entry.text
				output["probability"] = probs[i]
				output["start_logit"] = entry.start_logit
				output["end_logit"] = entry.end_logit
				nbest_json.append(output)

			assert len(nbest_json) >= 1

			all_predictions[example.qas_id] = nbest_json[0]["text"]
			all_nbest_json[example.qas_id] = nbest_json

		return all_nbest_json

	def response(self, data):
		# data = [[context, question], ...]
		eval_examples = self.process_example(data)

		eval_writer = FeatureWriter(
					filename=os.path.join(self.output_dir, "eval.tf_record"),
					is_training=False)
		eval_features = []

		def append_feature(feature):
			eval_features.append(feature)
			eval_writer.process_feature(feature)

		convert_examples_to_features(
					examples=eval_examples,
					tokenizer=self.tokenizer,
					max_seq_length=FLAGS.max_seq_length,
					doc_stride=FLAGS.doc_stride,
					max_query_length=FLAGS.max_query_length,
					is_training=False,
					output_fn=append_feature)
		eval_writer.close()

		all_results = []

		predict_input_fn = input_fn_builder(
					input_file=eval_writer.filename,
					seq_length=FLAGS.max_seq_length,
					is_training=False,
					drop_remainder=False)

		# If running eval on the TPU, you will need to specify the number of steps.
		all_results = []
		for result in self.estimator.predict(predict_input_fn, yield_single_examples=True):
			if len(all_results) % 1000 == 0:
				tf.logging.info("Processing example: %d" % (len(all_results)))
			unique_id = int(result["unique_ids"])
			start_logits = [float(x) for x in result["start_logits"].flat]
			end_logits = [float(x) for x in result["end_logits"].flat]
			all_results.append(RawResult(unique_id=unique_id,
										 start_logits=start_logits,
										 end_logits=end_logits))

		predictions = self.predict(eval_examples, eval_features, all_results,
				FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)

		return predictions
