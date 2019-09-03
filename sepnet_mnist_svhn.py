import argparse
import os
import sys

from tqdm import trange
import numpy as np
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets as tfds
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None

#header = ''

def train():
    batch_size = 512
    n_src_eval = 5000
    n_src_train = 50000

    n_tgt_eval = 5000
    n_tgt_train = 20000

    ## dataloaderの設定 ##
    src_ds = tfds.load(name='mnist', split=tfds.Split.TRAIN,
                        as_supervised=True)
    
    tgt_ds = tfds.load(name='svhn_cropped', split=tfds.Split.TRAIN,
                        as_supervised=True)
    
    def _standardize(image, label):
        return image/255, label

    src_ds = src_ds.map(_standardize)
    src_eval_ds = src_ds.take(n_src_eval)
    src_vis_ds = src_ds.take(n_src_eval)
    src_ds = src_ds.skip(n_src_eval)
    src_train_ds = src_ds.take(n_src_train)

    tgt_ds = tgt_ds.map(_standardize)
    tgt_eval_ds = tgt_ds.take(n_tgt_eval)
    tgt_vis_ds = tgt_ds.take(n_tgt_eval)
    tgt_ds = tgt_ds.skip(n_tgt_eval)
    tgt_train_ds = tgt_ds.take(n_tgt_train)

    src_eval_ds = src_eval_ds.batch(n_src_eval)
    src_eval_ds = src_eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

    src_vis_ds = src_vis_ds.batch(n_src_eval)

    tgt_eval_ds = tgt_eval_ds.batch(n_tgt_eval)
    tgt_eval_ds = tgt_eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

    tgt_vis_ds = tgt_vis_ds.batch(n_tgt_eval)

    src_train_ds = src_train_ds.shuffle(buffer_size=n_src_train)
    src_train_ds = src_train_ds.repeat()
    src_train_ds = src_train_ds.batch(batch_size)
    src_train_ds = src_train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    tgt_train_ds = tgt_train_ds.shuffle(buffer_size=n_tgt_train)
    tgt_train_ds = tgt_train_ds.repeat()
    tgt_train_ds = tgt_train_ds.batch(batch_size)
    tgt_train_ds = tgt_train_ds.prefetch(tf.data.experimental.AUTOTUNE)

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
    
    src_x, src_y_ = src_iterator.get_next()
    tgt_x, tgt_y_ = tgt_iterator.get_next()

    src_x = tf.reshape(src_x, [-1, 784])
    tgt_x = tf.reshape(tgt_x, [-1, 3072])

    with tf.name_scope('mnist_input_reshape'):
        src_image_shaped_input = tf.reshape(src_x, [-1, 28, 28, 1])
        tf.summary.image('input', src_image_shaped_input, 5)
    
    with tf.name_scope('tgt_input_reshape'):
        tgt_image_shaped_input = tf.reshape(tgt_x, [-1, 32, 32, 3])
        tf.summary.image('input', tgt_image_shaped_input, 5)

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
    penc_src_hidden1 = nn_layer(src_x, 784, 512, 'penc_src_layer1')
    pfeature_src = nn_layer(penc_src_hidden1, 512, 128, 'penc_src_layer2', act=tf.identity)

    penc_tgt_hidden1 = nn_layer(tgt_x, 3072, 1024, 'penc_tgt_layer1')
    pfeature_tgt = nn_layer(penc_tgt_hidden1, 1024, 128, 'penc_tgt_layer2', act=tf.identity)

    ## shared encoder ##
    senc_tgt_hidden1 = nn_layer(tgt_x, 3072, 512, 'senc_tgt_layer1')
    sfeature_tgt = nn_layer(senc_tgt_hidden1, 512, 128, 'senc_layer2', act=tf.identity)

    senc_src_hidden1 = nn_layer(src_x, 784, 512, 'senc_src_layer1')
    sfeature_src = nn_layer(senc_src_hidden1, 512, 128, 'senc_layer2', act=tf.identity, reuse=True)

    ## shared decoder ##
    dec_src_hidden1 = nn_layer(pfeature_src + sfeature_src, 128, 512, 'dec_layer1')
    dec_src = nn_layer(dec_src_hidden1, 512, 784, 'pdec_src_layer2', act=tf.identity)

    dec_tgt_hidden1 = nn_layer(pfeature_tgt + sfeature_tgt, 128, 512, 'dec_layer1', reuse=True)
    dec_tgt = nn_layer(dec_tgt_hidden1, 512, 3072, 'dec_tgt_layer2', act=tf.identity)

    ## classifier ##
    c_src_hidden1 = nn_layer(sfeature_src, 128, 512, 'c_src_layer1')
    src_y = nn_layer(c_src_hidden1, 512, 10, 'c_src_layer2', act=tf.identity)

    c_tgt_hidden1 = nn_layer(sfeature_tgt, 128, 512, 'c_tgt_layer1')
    tgt_y = nn_layer(c_tgt_hidden1, 512, 10, 'c_tgt_layer2', act=tf.identity)

    ## adversarial network ##
    a_src_input = gradient_reversal_layer(sfeature_src)
    a_src_hidden1 = nn_layer(a_src_input, 128, 512, 'a_layer1')
    a_src_y = nn_layer(a_src_hidden1, 512, 1, 'a_layer2', act=tf.identity)

    a_tgt_input = gradient_reversal_layer(sfeature_tgt)
    a_tgt_hidden1 = nn_layer(a_tgt_input, 128, 512, 'a_layer1', reuse=True)
    a_tgt_y = nn_layer(a_tgt_hidden1, 512, 1, 'a_layer2', act=tf.identity, reuse=True)
        
    ## src classifier loss ##
    with tf.name_scope('src_cross_entropy'):
        with tf.name_scope('total'):
            src_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=src_y_, logits=src_y))
        tf.summary.scalar('src_cross_entropy', src_cross_entropy)
    
    ## tgt classifier loss ##
    with tf.name_scope('tgt_cross_entropy'):
        with tf.name_scope('total'):
            tgt_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=tgt_y_, logits=tgt_y))
        tf.summary.scalar('tgt_cross_entropy', tgt_cross_entropy)

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
            difference = tf.reduce_mean(cor_mat_src**2) + tf.reduce_mean(cor_mat_tgt**2)
        tf.summary.scalar('difference', difference)
    
    ## recons loss ##
    with tf.name_scope('recons_loss'):
        with tf.name_scope('total'):
            recons_loss = tf.reduce_mean((dec_src - src_x)**2) + \
                        tf.reduce_mean((dec_tgt - tgt_x)**2)
        tf.summary.scalar('difference', recons_loss)
    
    ## similarity loss ##
    with tf.name_scope('similarity_loss'):
        with tf.name_scope('total'):
            sim_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([tf.shape(a_src_y)[0], 1]), logits=a_src_y) + \
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([tf.shape(a_tgt_y)[0], 1]), logits=a_tgt_y))
        tf.summary.scalar('similarity', sim_loss)

    with tf.name_scope('train'):
        loss = tgt_cross_entropy + \
                FLAGS.eta * src_cross_entropy + \
                    FLAGS.alpha * recons_loss + \
                        FLAGS.beta * difference + \
                            FLAGS.gamma * sim_loss
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(input=tgt_y, axis=1), tgt_y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                        tf.float32))
    tf.summary.scalar('accuracy', accuracy)

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
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run([init])

    saver = tf.train.Saver()

    for i in trange(FLAGS.max_step):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            sess.run([src_eval_init_op, tgt_eval_init_op])
            summary, acc = sess.run([merged, accuracy])
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
            sess.run([src_train_init_op, tgt_train_init_op])
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step])
            train_writer.add_summary(summary, i)

    sess.run([src_vis_init_op, tgt_vis_init_op])
    sfeature_src_xs, sfeature_tgt_xs = sess.run([sfeature_src, sfeature_tgt])
    sess.run([src_vis_init_op, tgt_vis_init_op])
    src_ys, tgt_ys = sess.run([src_y_, tgt_y_])
    train_writer.close()
    test_writer.close()
    sess.close()

    sfeature = np.concatenate([sfeature_src_xs, sfeature_tgt_xs])
    is_src = np.concatenate([np.ones(src_ys.shape[0]), np.zeros(tgt_ys.shape[0])])
    labels = np.concatenate([src_ys, tgt_ys])

    return sfeature, labels, is_src

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

    # embedding_src.metadata_path = FLAGS.log_dir + '/meta_src.tsv'
    # embedding_tgt.metadata_path = FLAGS.log_dir + '/meta_tgt.tsv'
    # embedding_src.metadata_path = './meta_src.tsv'
    # embedding_tgt.metadata_path = './meta_tgt.tsv'

    meta_path = './meta.tsv'
    embedding_sfeature.metadata_path = meta_path

    projector.visualize_embeddings(proj_writer, proj_config)

    saver_emb = tf.train.Saver([embeddings])
    saver_emb.save(sess, FLAGS.log_dir + '/emb_vis.ckpt')
    with open('./log/meta.tsv','w') as fout:
        fout.write("Index\tLabel\tis_src\n")
        for index in range(is_src.shape[0]):
            fout.write("%d\t%d\t%d\n" % (index, meta[index], is_src[index]))
    
    proj_writer.close()
    sess.close()
    
def main(_):
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.io.gfile.rmtree(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.log_dir)
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
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                            'tensorflow/mnist/logs/mnist_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)