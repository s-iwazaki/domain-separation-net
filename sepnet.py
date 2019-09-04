import argparse
import os
import sys

from tqdm import trange
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from utils import *

FLAGS = None

## 前処理で生のデータ化しておく
# 
# 何番目までが一つのカテゴリか定義しておく

SRC_CATEGORY_SPLITER_IN = [
    2, 3, 4 
]

SRC_CATEGORY_SPLITER_OUT = [
    2, 4, 3
]

SRC_FEATURE_TYPE_IN = [
    'cat', 'num'
]
SRC_FEATURE_TYPE_OUT = [
    'cat', 'num'
]

SRC_OUT_LIST = [3, 15] # 分割後の3番目要素までがoutput
N_SRC_IN = 10 # 前処理後の入力が10次元

SRC_CATEGORY_SPLITER_IN = [
    2, 3, 4 
]

SRC_CATEGORY_SPLITER_OUT = [
    2, 4, 3
]

TGT_FEATURE_TYPE_IN = [
    'cat', ''
]
TGT_FEATURE_TYPE_OUT = [
    'cat', ''
]

TGT_OUT = [3, 10] # 分割後の3番目要素までがoutput
N_TGT_IN = 10 # 前処理後の入力が10次元

src_train_file_path = 'src_sample_train.csv'
src_eval_file_path = 'src_sample_eval.csv'
src_vis_file_path = 'src_sample_vis.csv' # データ数10000で
tgt_train_file_path = 'tgt_sample_train.csv'
tgt_eval_file_path = 'tgt_sample_eval.csv'
tgt_vis_file_path = 'tgt_sample_vis.csv' # データ数10000で

def train():
    batch_size = 2048
    n_vis = 10000

    ## dataloaderの設定 #
    src_train_ds = get_dataset(src_train_file_path, batch_size)
    src_eval_ds = get_dataset(src_eval_file_path)
    src_vis_ds = get_dataset(src_vis_file_path, is_shuffle=False)

    tgt_train_ds = get_dataset(tgt_train_file_path, batch_size)
    tgt_eval_ds = get_dataset(tgt_eval_file_path)
    tgt_vis_ds = get_dataset(tgt_vis_file_path, is_shuffle=False)

    src_iterator = tf.data.Iterator.from_structure(src_train_ds.output_types,
                                           src_train_ds.output_shapes)

    tgt_iterator = tf.data.Iterator.from_structure(tgt_train_ds.output_types,
                                           tgt_train_ds.output_shapes) 

    src_train_init_op = src_iterator.make_initializer(src_train_ds)
    src_eval_init_op = src_iterator.make_initializer(src_eval_ds)
    src_vis_init_op = src_iterator.make_initializer(src_vis_ds)

    tgt_train_init_op = tgt_iterator.make_initializer(tgt_train_ds)
    tgt_eval_init_op = tgt_iterator.make_initializer(tgt_eval_ds)
    tgt_vis_init_op = tgt_iterator.make_initializer(tgt_vis_ds)
    
    src_ds = src_iterator.get_next()
    tgt_ds = tgt_iterator.get_next()
    src_y_, src_x = tf.split(src_ds, SRC_OUT)
    tgt_y_, tgt_x = tf.split(tgt_ds, TGT_OUT)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        # initial = tf.random.truncated_normal(shape, stddev=0.1)
        initial = tf.random.normal(shape, stddev=np.sqrt(2/shape[0]))
        return tf.get_variable('bias', initializer=initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable('weights', initializer=initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(input_tensor=var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(input_tensor=var))
            tf.summary.scalar('min', tf.reduce_min(input_tensor=var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, reuse=False):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.variable_scope(layer_name, reuse=reuse):
        # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations

    def gradient_reversal_layer(x):
        positive_path = tf.stop_gradient(x * tf.cast(2, tf.float32))
        negative_path = -x * tf.cast(1, tf.float32)
        return positive_path + negative_path

    ## private encoder ##
    penc_src_hidden1 = nn_layer(src_x, N_SRC_IN, 512, 'penc_src_layer1')
    pfeature_src = nn_layer(penc_src_hidden1, 512, 128, 'penc_src_layer2', act=tf.nn.sigmoid)

    penc_tgt_hidden1 = nn_layer(tgt_x, N_TGT_IN, 1024, 'penc_tgt_layer1')
    pfeature_tgt = nn_layer(penc_tgt_hidden1, 1024, 128, 'penc_tgt_layer2', act=tf.nn.sigmoid)

    ## shared encoder ##
    senc_tgt_hidden1 = nn_layer(tgt_x, N_TGT_IN, 512, 'senc_tgt_layer1')
    sfeature_tgt = nn_layer(senc_tgt_hidden1, 512, 128, 'senc_layer2', act=tf.nn.sigmoid)

    senc_src_hidden1 = nn_layer(src_x, N_SRC_IN, 512, 'senc_src_layer1')
    sfeature_src = nn_layer(senc_src_hidden1, 512, 128, 'senc_layer2', act=tf.nn.sigmoid, reuse=True)

    ## shared decoder ##
    dec_src_hidden1 = nn_layer(pfeature_src + sfeature_src, 128, 512, 'dec_layer1')
    dec_src = nn_layer(dec_src_hidden1, 512, N_SRC_OUT, 'pdec_src_layer2', act=tf.identity)
    src_dec_list = []
    cur = 0
    for i in range(len(SRC_CATEGORY_SPLITER_IN)):
        part_dec_src = tf.slice(dec_src, [0, cur], [-1, SRC_CATEGORY_SPLITER_IN[i]])
        src_dec_list.append(part_dec_src)
        cur += SRC_CATEGORY_SPLITER_IN[i]    
        
    dec_tgt_hidden1 = nn_layer(pfeature_tgt + sfeature_tgt, 128, 512, 'dec_layer1', reuse=True)
    dec_tgt = nn_layer(dec_tgt_hidden1, 512, N_TGT_OUT, 'dec_tgt_layer2', act=tf.identity)
    tgt_dec_list = []
    cur = 0
    for i in len(TGT_CATEGORY_SPLITER_IN):    
        part_dec_tgt = tf.slice(dec_tgt, [0, cur], [-1, TGT_CATEGORY_SPLITER_IN[i]])
        tgt_dec_list.append(part_dec_tgt)
        cur += TGT_CATEGORY_SPLITER_IN[i]

    ## classifier ##
    c_src_hidden1 = nn_layer(sfeature_src, 128, 512, 'c_layer1')
    src_y = nn_layer(c_src_hidden1, 512, N_SRC_OUT, 'c_src_layer2', act=tf.identity)
    src_y_list = []
    cur = 0
    for i in len(SRC_CATEGORY_SPLITER_OUT):    
        part_y_src = tf.slice(src_y_, [0, cur], [-1, SRC_CATEGORY_SPLITER_OUT[i]])
        src_y_list.append(part_y_src)
        cur += SRC_CATEGORY_SPLITER_OUT[i]
    
    c_tgt_hidden1 = nn_layer(sfeature_tgt, 128, 512, 'c_layer1', reuse=True)
    tgt_y = nn_layer(c_tgt_hidden1, 512, N_TGT_OUT, 'c_tgt_layer2', act=tf.identity)
    tgt_y_list = []
    cur = 0
    for i in len(TGT_CATEGORY_SPLITER_OUT):    
        part_y_tgt = tf.slice(tgt_y_, [0, cur], [-1, TGT_CATEGORY_SPLITER_OUT[i]])
        tgt_y_list.append(part_y_tgt)
        cur += TGT_CATEGORY_SPLITER_OUT[i]

    ## adversarial network ##
    a_src_input = gradient_reversal_layer(sfeature_src)
    a_src_hidden1 = nn_layer(a_src_input, 128, 512, 'a_layer1')
    a_src_y = nn_layer(a_src_hidden1, 512, 1, 'a_layer2', act=tf.identity)

    a_tgt_input = gradient_reversal_layer(sfeature_tgt)
    a_tgt_hidden1 = nn_layer(a_tgt_input, 128, 512, 'a_layer1', reuse=True)
    a_tgt_y = nn_layer(a_tgt_hidden1, 512, 1, 'a_layer2', act=tf.identity, reuse=True)
        
    ## src classifier loss ##
    with tf.name_scope('src_classifier'):
        with tf.name_scope('total'):
            src_classifier_loss = 0
            cur = 0
            for i in range(len(SRC_CATEGORY_SPLITER_OUT)):
                if SRC_FEATURE_TYPE_OUT[i] == 'cat':
                    src_classifier_loss += tf.losses.categorical_cross_entropy(
                        labels=tf.slice(src_y_, [0, cur], [-1, SRC_CATEGORY_SPLITER_OUT[i]]), logits=src_y_list[i])
                else:
                    src_classifier_loss += tf.reduce_sum((tf.slice(src_y_, [0, cur], [-1, SRC_CATEGORY_SPLITER_OUT[i]]) - src_y_list[i])**2, axis=-1)
                cur += SRC_CATEGORY_SPLITER_OUT[i]
            src_classifier_loss = tf.reduce_mean(src_classifier_loss)
        tf.summary.scalar('src_classifier_loss', src_classifier_loss)
    
    ## tgt classifier loss ##
    with tf.name_scope('tgt_classifier'):
        with tf.name_scope('total'):
            tgt_classifier_loss = 0
            cur = 0
            for i in range(len(TGT_CATEGORY_SPLITER_OUT)):
                if TGT_FEATURE_TYPE_OUT[i] == 'cat':
                    tgt_classifier_loss += tf.losses.categorical_cross_entropy(
                        labels=tf.slice(tgt_y_, [0, cur], [-1, TGT_CATEGORY_SPLITER_OUT[i]]), logits=tgt_y_list[i])
                else:
                    tgt_classifier_loss += tf.reduce_sum((tf.slice(tgt_y_, [0, cur], [-1, TGT_CATEGORY_SPLITER_OUT[i]]) - tgt_y_list[i])**2, axis=-1)
                tgt_classifier_loss = tf.reduce_mean(tgt_classifier_loss)
        tf.summary.scalar('tgt_classifier_loss', tgt_classifier_loss)

    ## difference loss ##
    with tf.name_scope('difference'):
        with tf.name_scope('total'):
            tsfeature_src = sfeature_src - tf.reduce_mean(sfeature_src, 0)
            tpfeature_src = pfeature_src - tf.reduce_mean(pfeature_src, 0)
            tsfeature_src = tf.nn.l2_normalize(tsfeature_src, 1)
            tpfeature_src = tf.nn.l2_normalize(tpfeature_src, 1)

            tsfeature_tgt = sfeature_tgt - tf.reduce_mean(sfeature_tgt, 0)
            tpfeature_tgt = pfeature_tgt - tf.reduce_mean(pfeature_tgt, 0)
            tsfeature_tgt = tf.nn.l2_normalize(tsfeature_tgt, 1)
            tpfeature_tgt = tf.nn.l2_normalize(tpfeature_tgt, 1)

            cor_mat_src = tf.matmul(tsfeature_src, tpfeature_src, transpose_a=True)
            cor_mat_tgt = tf.matmul(tsfeature_tgt, tpfeature_tgt, transpose_a=True)
            difference = tf.reduce_mean(tf.reduce_sum(cor_mat_src**2, axis=[1, 2]) + tf.reduce_sum(cor_mat_tgt**2, axis=[1,2]))
        tf.summary.scalar('difference', difference)
    
    ## recons loss ##
    with tf.name_scope('recons_loss'):
        with tf.name_scope('total'):
            src_recons_loss = 0
            tgt_recons_loss = 0
            src_cur = 0
            tgt_cur = 0
            for i in range(len(SRC_FEATURE_TYPE_IN)):
                if SRC_FEATURE_TYPE_IN[i] == 'cat':
                    src_recons_loss += tf.losses.categorical_cross_entropy(
                        labels=tf.slice(src_x, [0, src_cur], [-1, SRC_CATEGORY_SPLITER_IN[i]]), logits=src_dec_list[i])
                else:
                    src_recons_loss += tf.reduce_sum((tf.slice(src_x, [0, src_cur], [-1, SRC_CATEGORY_SPLITER_IN[i]]) - src_dec_list[i])**2, axis=-1)
                src_cur += SRC_CATEGORY_SPLITER_IN[i]
            for i in range(len(TGT_FEATURE_TYPE_IN)):
                if TGT_FEATURE_TYPE_IN[i] == 'cat':
                    tgt_recons_loss += tf.losses.categorical_cross_entropy(
                        labels=tf.slice(tgt_x, [0, tgt_cur], [-1, TGT_CATEGORY_SPLITER_IN[i]]), logits=tgt_dec_list[i])
                else:
                    tgt_recons_loss += tf.reduce_sum((tf.slice(tgt_x, [0, tgt_cur], [-1, TGT_CATEGORY_SPLITER_IN[i]]) - tgt_dec_list[i])**2, axis=-1)
                tgt_cur += TGT_CATEGORY_SPLITER_IN[i]
            recons_loss = tf.reduce_mean(src_recons_loss) + tf.reduce_mean(tgt_recons_loss)
        tf.summary.scalar('reconstruction', recons_loss)
    
    ## similarity loss ##
    with tf.name_scope('similarity_loss'):
        with tf.name_scope('total'):
            sim_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([tf.shape(a_src_y)[0], 1]), logits=a_src_y) + \
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([tf.shape(a_tgt_y)[0], 1]), logits=a_tgt_y))
        tf.summary.scalar('similarity', sim_loss)

    with tf.name_scope('train'):
        loss = tgt_classifier_loss + \
                FLAGS.eta * src_classifier_loss + \
                    FLAGS.alpha * recons_loss + \
                        FLAGS.beta * difference + \
                            FLAGS.gamma * sim_loss
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            loss)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    merged = tf.summary.merge_all()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="0"
        )
    )
    # config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run([init])

    saver = tf.train.Saver()

    for i in trange(FLAGS.max_step):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            sess.run([src_eval_init_op, tgt_eval_init_op])
            summary = sess.run([merged])
            test_writer.add_summary(summary, i)
            print('at step %s' % (i,))
            sess.run([src_train_init_op, tgt_train_init_op])
        if i % 500 = 499:
            saver.save(sess, FLAGS.model_dir + '/model.ckpt', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step])
            train_writer.add_summary(summary, i)

    saver.save(sess, FLAGS.model_dir + '/model.ckpt', i)
    sess.run([src_vis_init_op, tgt_vis_init_op])
    sfeature_src_xs, sfeature_tgt_xs = sess.run([sfeature_src, sfeature_tgt])
    sess.run([src_vis_init_op, tgt_vis_init_op])
    src_xs, tgt_xs, src_ys, tgt_ys = sess.run([src_x, tgt_x, src_y_, tgt_y_])
    src_info = np.concatenate([src_xs, src_ys], axis=1)
    tgt_info = np.concatenate([tgt_xs, tgt_ys], axis=1)
    train_writer.close()
    test_writer.close()
    sess.close()

    sfeature = np.concatenate([sfeature_src_xs, sfeature_tgt_xs], axis=0)
    is_src = np.concatenate([np.ones(src_ys.shape[0]), np.zeros(tgt_ys.shape[0])])
    src_info = np.concatenate([src_info, np.zeros([src_info.shape[0], N_TGT_OUT + N_TGT_IN])], axis=1)
    tgt_info = np.concatenate([np.zeros([tgt_info.shape[0], N_SRC_OUT + N_SRC_IN]), tgt_info], axis=1)
    meta_info = np.concatenate([src_info, tgt_info], axis=0)

    return sfeature, meta_info, is_src

def visualize(sfeature, meta, is_src):
    ### visualize middle layer ###
    embeddings = tf.Variable(sfeature, trainable=False, name='embeddings')

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="0"
        )
    )

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)

    sess.run(init)
    proj_writer = tf.summary.FileWriter(FLAGS.log_dir)
    proj_config = projector.ProjectorConfig()
    embedding_sfeature = proj_config.embeddings.add()
    embedding_sfeature.tensor_name = embeddings.name

    meta_path = './meta.tsv'
    embedding_sfeature.metadata_path = meta_path

    projector.visualize_embeddings(proj_writer, proj_config)

    saver_emb = tf.train.Saver([embeddings])
    saver_emb.save(sess, FLAGS.log_dir + '/emb_vis.ckpt')
    with open(FLAGS.log_dir + '/meta.tsv', 'w') as fout:
        fout.write("Index\tLabel\tis_src\n")
        for index in range(is_src.shape[0]):
            fout.write("%d\t%d\t%d\n" % (index, meta[index], is_src[index]))
    
    proj_writer.close()
    sess.close()
    
def main(_):
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.io.gfile.rmtree(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.log_dir)
    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)
    with tf.Graph().as_default():
        middle_data, labels, is_src = train()
    
    with tf.Graph().as_default():
        visualize(middle_data, labels, is_src)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_step', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=0.075)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Summaries log directory')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)