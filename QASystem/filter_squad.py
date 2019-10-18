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

import os
import sys
import collections
import json
import tensorflow as tf

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from QASystem import modeling
from QASystem import tokenization
from QASystem.run_squad import model_fn_builder, input_fn_builder, FeatureWriter, _compute_softmax, RawResult, \
	read_squad_examples, convert_examples_to_features

flags = tf.flags

FLAGS = flags.FLAGS

def write_predictions(all_examples, all_features, all_results, output_file):
	example_index_to_features = collections.defaultdict(list)
	for feature in all_features:
		example_index_to_features[feature.example_index].append(feature)

	unique_id_to_result = {}
	for result in all_results:
		unique_id_to_result[result.unique_id] = result

	scores = {}
	for (example_index, example) in enumerate(all_examples):
		features = example_index_to_features[example_index]

		score = 0
		for (feature_index, feature) in enumerate(features):
			if (feature.start_position == 0 and feature.end_position == 0):
				continue
			result = unique_id_to_result[feature.unique_id]
			start_logits_softmax = _compute_softmax(result.start_logits)
			end_logits_softmax = _compute_softmax(result.end_logits)
			score = start_logits_softmax[feature.start_position] * end_logits_softmax[feature.end_position] 
		scores[example.qas_id] = score
			
	with tf.gfile.GFile(output_file, "w") as writer:
		writer.write(json.dumps(scores, indent=4, ensure_ascii=False) + "\n")

def validate_flags_or_throw(bert_config):

	if not FLAGS.predict_file:
		raise ValueError("`predict_file` must be specified.")

	if FLAGS.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError(
				"Cannot use sequence length %d because the BERT model "
				"was only trained up to sequence length %d" %
				(FLAGS.max_seq_length, bert_config.max_position_embeddings))

	if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
		raise ValueError(
				"The max_seq_length (%d) must be greater than max_query_length "
				"(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
	validate_flags_or_throw(bert_config)
	tf.gfile.MakeDirs(FLAGS.output_dir)
	tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
	tpu_cluster_resolver = None
	if FLAGS.use_tpu and FLAGS.tpu_name:
		tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			master=FLAGS.master,
			model_dir=FLAGS.output_dir,
			save_checkpoints_steps=FLAGS.save_checkpoints_steps,
			tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=FLAGS.iterations_per_loop,
					num_shards=FLAGS.num_tpu_cores,
					per_host_input_for_training=is_per_host))

	model_fn = model_fn_builder(
			bert_config=bert_config,
			init_checkpoint=FLAGS.init_checkpoint,
			learning_rate=FLAGS.learning_rate,
			num_train_steps=None,
			num_warmup_steps=None,
			use_tpu=FLAGS.use_tpu,
			use_one_hot_embeddings=FLAGS.use_tpu)

	# If TPU is not available, this will fall back to normal Estimator on CPU
	# or GPU.
	estimator = tf.contrib.tpu.TPUEstimator(
			use_tpu=FLAGS.use_tpu,
			model_fn=model_fn,
			config=run_config,
			train_batch_size=FLAGS.train_batch_size,
			predict_batch_size=FLAGS.predict_batch_size)

	eval_examples = read_squad_examples(input_file=FLAGS.predict_file, is_training=True)

	eval_writer = FeatureWriter(
			filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
			is_training=False)
	eval_features = []

	def append_feature(feature):
		eval_features.append(feature)
		eval_writer.process_feature(feature)

	convert_examples_to_features(
			examples=eval_examples,
			tokenizer=tokenizer,
			max_seq_length=FLAGS.max_seq_length,
			doc_stride=FLAGS.doc_stride,
			max_query_length=FLAGS.max_query_length,
			is_training=True,
			output_fn=append_feature)
	eval_writer.close()

	all_results = []

	predict_input_fn = input_fn_builder(
			input_file=eval_writer.filename,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=False)

	# If running eval on the TPU, you will need to specify the number of
	# steps.
	all_results = []
	for result in estimator.predict(
			predict_input_fn, yield_single_examples=True):
		if len(all_results) % 1000 == 0:
			tf.logging.info("Processing example: %d" % (len(all_results)))
		unique_id = int(result["unique_ids"])
		start_logits = [float(x) for x in result["start_logits"].flat]
		end_logits = [float(x) for x in result["end_logits"].flat]
		all_results.append(
				RawResult(
						unique_id=unique_id,
						start_logits=start_logits,
						end_logits=end_logits))

	output_file = os.path.join(FLAGS.output_dir, "scores.json")

	write_predictions(eval_examples, eval_features, all_results, output_file)


if __name__ == "__main__":
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()
