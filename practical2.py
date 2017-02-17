import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vocab_size', 1000, 'Size of vocabulary derived from text corpus')
flags.DEFINE_integer('embedding_dim', 50, 'Dimension of embedding vectors')
flags.DEFINE_string('embedding_type', 'learnable', 'Either fixed, random, or learnable')
flags.DEFINE_integer('batch_size', 50, 'Training batch size')
flags.DEFINE_integer('hidden_layer_size', 20, 'Size of the hidden layer')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('summaries_dir', 'logs', 'Summaries directory')

import data
ted = data.ted(FLAGS.vocab_size)
embedding = data.glove(FLAGS.embedding_dim)

vocab_vectors = np.empty([FLAGS.vocab_size, FLAGS.embedding_dim])
for word, index in ted['vocab'].items():
  vocab_vectors[index] = embedding.get(word, np.zeros(FLAGS.embedding_dim))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  with tf.name_scope(layer_name):
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


# DEFINE NETWORK

with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, FLAGS.vocab_size], name='document')
  y = tf.placeholder(tf.int32, [None], 'label-int')
  y_hot = tf.squeeze(tf.one_hot(y, depth=8))

with tf.name_scope('embedding'):
  embedding = tf.Variable(
    tf.cast(vocab_vectors, tf.float32) if FLAGS.embedding_type != 'random' else
      tf.truncated_normal([FLAGS.vocab_size, FLAGS.embedding_dim], stddev=.2),
    trainable = (FLAGS.embedding_type != 'fixed'),
    name='embedding-matrix')

  embedding_sum = tf.matmul(x, embedding)
  doc_length = tf.expand_dims(tf.reduce_sum(x, 1), 1)
  doc_length_nonzero = tf.maximum(doc_length, 1)
  doc_embedding = tf.div(embedding_sum, doc_length_nonzero)

hidden_layer = nn_layer(doc_embedding, FLAGS.embedding_dim, FLAGS.hidden_layer_size, 'hidden_layer')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', FLAGS.dropout)
  dropped = tf.nn.dropout(hidden_layer, FLAGS.dropout)

y_hat = nn_layer(dropped, FLAGS.hidden_layer_size, 8, 'dropout_layer', act=tf.identity)

with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_hot, logits=y_hat)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_hot, 1), tf.argmax(y_hat, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:

  if tf.gfile.Exists(FLAGS.summaries_dir):
      tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

  tf.global_variables_initializer().run()

  x_train = np.copy(ted['documents'][:1585])
  y_train = np.copy(ted['categories'][:1585])

  x_validation = ted['documents'][1585:1835]
  y_validation = ted['categories'][1585:1835]


  iters = 1585 // FLAGS.batch_size

  from sklearn.utils import shuffle
  for reps in range(80):
    x_shuffled, y_shuffled = shuffle(x_train, y_train)
    for i in range(iters):
      step = reps*iters+i

      x_batch = x_shuffled[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size]
      y_batch = y_shuffled[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size]

      if step % 10 == 0:
          feed_dict = {x: x_validation, y: y_validation, keep_prob: 1.0}
          summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
          validation_writer.add_summary(summary, step)
          print('Accuracy at step %s: %s' % (step, acc))

      else:
          feed_dict = {x: x_batch, y: y_batch, keep_prob: FLAGS.dropout}
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
          train_writer.add_summary(summary, step)

  x_test = ted['documents'][1835:]
  y_test = ted['categories'][1835:]

  feed_dict = {x: x_test, y: y_test, keep_prob: 1.0}
  summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
  test_writer.add_summary(summary, step)

  print('Final test accuracy: %s' % acc)

  train_writer.close()
  validation_writer.close()
  test_writer.close()
