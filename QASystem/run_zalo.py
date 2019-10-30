import tensorflow as tf
from os.path import join, exists
from preprocess import ZaloDatasetProcessor

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", None,
                    "Training or Testing?")
flags.DEFINE_bool("train_display_info", True,
                  "Display tensorflow flag to track training progress")
flags.DEFINE_string("dataset_path", None,
                    "The path to the dataset")
flags.DEFINE_string("bert_model_path", None,
                    "Link to BERT cased model")
flags.DEFINE_integer("max_sequence_len", 300,
                     "The maximum input sequence length for embeddings")
flags.DEFINE_bool("do_lowercase", False,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")


def main(_):
    # tf.logging.set_verbosity(tf.logging.info if FLAGS.train_display_info else tf.logging.FATAL)

    print("[Main] Starting....")

    # Data initialization
    train_file = join(FLAGS.dataset_path, "train.tfrecords")
    dev_file = join(FLAGS.dataset_path, "dev.tfrecords")
    test_file = join(FLAGS.dataset_path, "test.tfrecords")

    def is_preprocessed():
        if FLAGS.mode == 'train':
            return not exists(train_file) or not exists(dev_file)
        elif FLAGS.mode == 'test':
            return not exists(test_file)
        else:
            return False

    if not is_preprocessed():
        print('[Main] No preprocess data found. Begin preprocess')
        dataset_processor = ZaloDatasetProcessor()
        dataset_processor.load_from_path(dataset_path=FLAGS.dataset_path)
        dataset_processor.write_all_to_tfrecords(output_folder=FLAGS.dataset_path,
                                                 bert_pretrained_model_path=FLAGS.bert_model_path,
                                                 do_lowercase=FLAGS.do_lowercase,
                                                 max_sequence_length=FLAGS.max_sequence_len)
        print('[Main] Preprocess complete')

    # Define model & training/testing


def flags_check():
    pass


if __name__ == "__main__":
    flags_check()
    tf.compat.v1.app.run()
