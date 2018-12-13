import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import json
import shutil
import os
from model.text_cnn import TextCnn
import numpy as np


tf.reset_default_graph()
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
params_path = os.path.join(config_path, 'kaggle_movie_review.json')
with open(params_path) as param:
    params_dict = json.load(param)
# print(params_dict)
config = tf.contrib.training.HParams(**params_dict)
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
TOTAL_STEPS = int((config.train['train_size']/config.model['batch_size']) * config.train['num_epochs'])
model_dir = 'trained_models_distribute/{}'.format(config.model_name)


hparams = tf.contrib.training.HParams(num_epochs=config.train['num_epochs'],
                                      batch_size=config.model['batch_size'])

HEADER = ['instances', 'Sentiment']
HEADER_DEFAULTS = [["NA"],["NA"]]
TEXT_FEATURE_NAME = "instances"
TARGET_NAME = 'Sentiment'
EVAL_AFTER_SEC = 60
RESUME_TRAINING = False
TARGET_LABELS = ["0", "1", "2", "3", "4"]
VOCAB_LIST_FILE = 'data/kaggle_processed_data/vocab'



PAD_WORD = '<pad>'
N_WORDS = 15180
# label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
label_dict = {}
for i, label in enumerate(TARGET_LABELS):
    label_dict[label] = i
word_dict = {}
k = 0
with open(VOCAB_LIST_FILE) as f:
    for word in f:
        word = word.strip()
        word_dict[word] = k
        k += 1

def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=HEADER_DEFAULTS, field_delim='\t', select_cols=[0, 1])
    features = dict(zip(HEADER, columns))

    target = features.pop(TARGET_NAME)
    return features, target



def input_fn(filename, mode=tf.estimator.ModeKeys.EVAL,
             num_epochs=1,
             batch_size=200):
    with open(filename) as f:
        lines = f.readlines()
        res = []
        target = []
        for line in lines[1:]:
            _, _, line, label = line.strip().split('\t')
            words = line.split(' ')
            words_id = []
            for word in words:
                if word in word_dict:
                    words_id.append(str(word_dict[word]))
                else:
                    words_id.append(str(word_dict[PAD_WORD]))
            max_seq_length = config.data['max_seq_length']
            words_id = words_id[:max_seq_length] if len(words_id) >= max_seq_length else words_id + [0] * (max_seq_length - len(words_id))
            res.append(np.array(words_id))
            target.append(int(label))
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count()
    buffer_size = 2 * batch_size + 1
    print("")
    print("* data input_fn:")
    print("================")
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    dataset = tf.data.Dataset.from_tensor_slices(({"instances": np.array(res, dtype=np.int32)}, np.array(target, dtype=np.int32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size)

    # dataset = dataset.map(lambda x,y:(process_text(x[TEXT_FEATURE_NAME]), parse_label_column(y)))
    return dataset


def model_fn(features, labels, mode, params):

    word_embeddings = tf.contrib.layers.embed_sequence(features['instances'], vocab_size=N_WORDS,
                                                       embed_dim=config.model['embed_dim'])
    text_cnn = TextCnn(mode=mode)
    word_embeddings = tf.expand_dims(word_embeddings, -1)
    output = text_cnn.build(word_embeddings)
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(output)
        predicted_indices = tf.argmax(probabilities, axis=1)
        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': output
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
    labels_one_hot = tf.one_hot(
        labels,
        depth=len(TARGET_LABELS),
        on_value=True,
        off_value=False,
        dtype=tf.bool
    )
    print("==========================================")
    print(labels_one_hot)
    loss = tf.losses.softmax_cross_entropy(labels_one_hot, output, scope="loss")
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(config.train['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(output)
        predicted_indices = tf.argmax(probabilities, 1)



        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities)
        }

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=hparams,
                                       config=run_config)
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


def serving_input_fn():
    receiver_tensor = {
        'instances': tf.placeholder(tf.int32, [None])
    }
    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)

if __name__ == '__main__':
    # ==============另一训练方式===============
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    run_config = tf.estimator.RunConfig(log_step_count_steps=config.train['log_step_count_steps'],
                                        tf_random_seed=config.train['tf_random_seed'],
                                        model_dir=model_dir,
                                        session_config=tf.ConfigProto(allow_soft_placement=True,
                                                                      log_device_placement=True),
                                        train_distribute=distribution)
    estimator = create_estimator(run_config, hparams)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn('data/kaggle_movie_reviews/train.tsv',
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  num_epochs=config.train['num_epochs'],
                                  batch_size=config.model['batch_size']),
        max_steps=TOTAL_STEPS,
        hooks=None
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn('data/kaggle_movie_reviews/train.tsv',
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=config.model['batch_size']),
        exporters=[tf.estimator.LatestExporter(name="predict",
                                               serving_input_receiver_fn=serving_input_fn,
                                               exports_to_keep=1,
                                               as_text=True)],
        steps=None,
        throttle_secs=EVAL_AFTER_SEC
    )
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)



    # =============转训练文本为ids==============
    # file = "data/kaggle_movie_reviews/train.tsv"
    # out_file = "data/kaggle_movie_reviews/train_ids.tsv"
    # with open(file) as f:
    #     lines = f.readlines()
    #     res = []
    #
    #     for line in lines[1:]:
    #         _, _, line, label = line.strip().split('\t')
    #         words = line.split(' ')
    #         words_id = []
    #         for word in words:
    #             if word in word_dict:
    #                 words_id.append(str(word_dict[word]))
    #             else:
    #                 words_id.append(str(word_dict[PAD_WORD]))
    #         res.append(' '.join(words_id) + '\t' + str(label) + '\n')
    # with open(out_file, 'w') as f:
    #     f.writelines(res)

    # ==============训练阶段===========================
    # distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    #
    # run_config = tf.estimator.RunConfig(log_step_count_steps=config.train['log_step_count_steps'],
    #                                     tf_random_seed=config.train['tf_random_seed'],
    #                                     model_dir=model_dir,
    #                                     session_config=tf.ConfigProto(allow_soft_placement=True,
    #                                                                   log_device_placement=True),
    #                                     train_distribute=distribution)
    # params = {}
    # estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params=params)
    # estimator.train(input_fn=lambda: input_fn('data/kaggle_movie_reviews/train.tsv',
    #                               mode=tf.estimator.ModeKeys.TRAIN,
    #                               num_epochs=config.train['num_epochs'],
    #                               batch_size=config.model['batch_size']),
    #                 max_steps=TOTAL_STEPS)









