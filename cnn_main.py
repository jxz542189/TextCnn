import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import json
import shutil
import os
from model.text_cnn import TextCnn


tf.reset_default_graph()
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
params_path = os.path.join(config_path, 'kaggle_movie_review.json')
with open(params_path) as param:
    params_dict = json.load(param)
# print(params_dict)
config = tf.contrib.training.HParams(**params_dict)
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
TOTAL_STEPS = int((config.train['train_size']/config.model['batch_size']) * config.train['num_epochs'])
model_dir = 'trained_models/{}'.format(config.model_name)
run_config = tf.estimator.RunConfig(log_step_count_steps=config.train['log_step_count_steps'],
                                    tf_random_seed=config.train['tf_random_seed'],
                                    model_dir=model_dir)

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
def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=HEADER_DEFAULTS, field_delim='\t', select_cols=[2, 3])
    features = dict(zip(HEADER, columns))

    target = features.pop(TARGET_NAME)
    return features, target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)

def input_fn(file_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
             skip_header_lines=1,
             num_epochs=1,
             batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count()
    buffer_size = 2 * batch_size + 1
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(file_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    dataset = data.TextLineDataset(filenames=file_name_pattern)
    dataset = dataset.skip(skip_header_lines)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda tsv_row:parse_tsv_row(tsv_row),
                          num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, parse_label_column(target)


def process_text(text_feature):

    smss = text_feature
    words = tf.string_split(smss)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=PAD_WORD)
    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=VOCAB_LIST_FILE,
                                                          num_oov_buckets=0, default_value=0)
    word_ids = vocab_table.lookup(dense_words)
    padding = tf.constant([[0, 0], [0, config.data['max_seq_length']]])
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, config.data['max_seq_length']])

    return word_id_vector


def model_fn(features, labels, mode, params):
    word_id_vector = process_text(features[TEXT_FEATURE_NAME])
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector, vocab_size=N_WORDS,
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
        'instances': tf.placeholder(tf.string, [None])
    }
    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)

if __name__ == '__main__':
    # ==============预测阶段===========================
    export_dir = model_dir + "/export/predict/"
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]
    print(saved_model_dir)
    print("")

    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir,
                                                         signature_def_key="prediction")
    output = predictor_fn(
        {'instances':[
            'An intermittently pleasing but mostly routine effort .',
            'watching in Birthday Girl , a film by the stage-trained Jez Butterworth -LRB- Mojo -RRB- that serves'
        ]}
    )
    print(output)
    # ==============训练阶段===========================
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn('data/kaggle_movie_reviews/train.tsv',
    #                               mode=tf.estimator.ModeKeys.TRAIN,
    #                               num_epochs=config.train['num_epochs'],
    #                               batch_size=config.model['batch_size']),
    #     max_steps=TOTAL_STEPS,
    #     hooks=None
    # )
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn('data/kaggle_movie_reviews/train.tsv',
    #                               mode=tf.estimator.ModeKeys.EVAL,
    #                               batch_size=config.model['batch_size']),
    #     exporters=[tf.estimator.LatestExporter(name="predict",
    #                                            serving_input_receiver_fn=serving_input_fn,
    #                                            exports_to_keep=1,
    #                                            as_text=True)],
    #     steps=None,
    #     throttle_secs=EVAL_AFTER_SEC
    # )
    #
    # if not RESUME_TRAINING:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(model_dir, ignore_errors=True)
    # else:
    #     print("Resuming training...")
    #
    # tf.logging.set_verbosity(tf.logging.INFO)
    #
    # time_start = datetime.utcnow()
    # print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    #
    # estimator = create_estimator(run_config, hparams)
    #
    # tf.estimator.train_and_evaluate(estimator=estimator,
    #                                 train_spec=train_spec,
    #                                 eval_spec=eval_spec)
    # time_end = datetime.utcnow()
    # print(".......................................")
    # print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    # print("")
    # time_elapsed = time_end - time_start
    # print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

# =====================测试input_fn函数=========================
    # features, target = input_fn('/root/PycharmProjects/textcnn/data/kaggle_movie_reviews/train.tsv',
    #                             mode=tf.estimator.ModeKeys.EVAL,
    #                             skip_header_lines=1,
    #                             batch_size=2)
    # sess = tf.Session()
    # table_init = tf.tables_initializer()
    # sess.run(table_init)
    # features, target = sess.run([features, target])
    # print(features)
    # print(target)






