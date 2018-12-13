import tensorflow as tf
import os
import json


path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(path, 'config')
params_path = os.path.join(config_path, 'kaggle_movie_review.json')

with open(params_path, 'r') as fin:
    options = json.load(fin)
config = tf.contrib.training.HParams(**options)


class TextCnn(object):
    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, embedding_input):
        conv_output = self.build_conv_layers(embedding_input)
        return self.build_fully_connected_layers(conv_output)

    def build_conv_layers(self, embedding_input):
        with tf.variable_scope("convolutions", dtype=self.dtype) as scope:
            pooled_outputs = self._build_conv_maxpool(embedding_input)

            num_total_filters = config.model['num_filters'] * len(config.model['filter_sizes'])
            concat_pooled = tf.concat(pooled_outputs, 3)
            flat_pooled = tf.reshape(concat_pooled, [-1, num_total_filters])

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                h_dropout = tf.layers.dropout(flat_pooled, config.model['dropout'])
            else:
                h_dropout = tf.layers.dropout(flat_pooled, 0)
            return h_dropout

    def _build_conv_maxpool(self, embedding_input):
        pooled_outputs = []
        for filter_size in config.model['filter_sizes']:
            with tf.variable_scope(f"conv-maxpool-{filter_size}-filter"):
                conv = tf.layers.conv2d(
                        embedding_input,
                        config.model['num_filters'],
                        (filter_size, config.model['embed_dim']),
                        activation=tf.nn.relu)

                pool = tf.layers.max_pooling2d(
                        conv,
                        (config.data['max_seq_length'] - filter_size + 1, 1),
                        (1, 1))

                pooled_outputs.append(pool)
        return pooled_outputs

    def build_fully_connected_layers(self, conv_output):
        with tf.variable_scope("fully-connected", dtype=self.dtype) as scope:
            return tf.layers.dense(
                    conv_output,
                    config.data['num_classes'],
                    kernel_initializer=tf.contrib.layers.xavier_initializer())


