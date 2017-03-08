import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from data import TedDataWithLabels, Glove
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vocab_size', 1000, 'Size of vocabulary derived from text corpus')
flags.DEFINE_integer('embedding_dim', 50, 'Dimension of embedding vectors')
flags.DEFINE_integer('batch_size', 50, 'Training batch size')
flags.DEFINE_integer('rnn_internal_size', 20, 'Size of the RNN\'s state')
flags.DEFINE_integer('hidden_layer_size', 50, 'Size of the hidden layer')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('summaries_dir', 'logs', 'Summaries directory')

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

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.tanh):
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

with tf.name_scope('input'):
  X = tf.placeholder(tf.int64, [None, None], name='document')
  # [batch_size, length]
  Y = tf.placeholder(tf.int64, [None], 'label-int')
  # [batch_size]
  L = tf.placeholder(tf.int64, [None], name='length')
  Lplus = tf.cast(tf.maximum(L,1), tf.float32)
  # [batch_size]

with tf.name_scope('embedding'):
  init_embedding = tf.placeholder(tf.float32, [FLAGS.vocab_size, FLAGS.embedding_dim])
  embedding = tf.Variable(init_embedding, name='embedding')
  # [vocab_size, embedding_dim]
  Xe = tf.nn.embedding_lookup(embedding, X)
  # [batch_size, length, embedding_dim]

with tf.name_scope('rnn'):
  cell = rnn.GRUCell(FLAGS.rnn_internal_size)
  initial_state = tf.placeholder(tf.float32, [None, FLAGS.rnn_internal_size])
  Yr, final_state = tf.nn.dynamic_rnn(cell, Xe, sequence_length=L, initial_state=initial_state)
  # [batch_size, length, internal_size]
  Yavg = tf.div(tf.reduce_sum(Yr, axis=1), tf.expand_dims(Lplus,1))
  # [batch_size, internal_size]

hidden_layer = nn_layer(Yavg, FLAGS.rnn_internal_size, FLAGS.hidden_layer_size, 'hidden_layer')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder_with_default(1.0, [])
  dropped = tf.nn.dropout(hidden_layer, FLAGS.dropout)

Yh = nn_layer(dropped, FLAGS.hidden_layer_size, 8, 'dropout_layer', act=tf.identity)

with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y, depth=8), logits=Yh)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(Y, tf.argmax(Yh, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:

  if tf.gfile.Exists(FLAGS.summaries_dir):
      tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train' + time.strftime("%H%M%S"), sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation' + time.strftime("%H%M%S"))
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test' + time.strftime("%H%M%S"))

  ted_data = TedDataWithLabels(FLAGS.vocab_size)
  feed_dict_valid = {X: ted_data.x_valid, Y: ted_data.y_valid, L: ted_data.x_valid_l,
                     initial_state: np.zeros([ted_data.x_valid.shape[0], FLAGS.rnn_internal_size], dtype=np.float32)}
  feed_dict_test  = {X: ted_data.x_test,  Y: ted_data.y_test,  L: ted_data.x_test_l,
                     initial_state: np.zeros([ted_data.x_test.shape[0], FLAGS.rnn_internal_size], dtype=np.float32)}

  glove = Glove(FLAGS.embedding_dim)
  embedding_matrix = np.zeros([FLAGS.vocab_size, FLAGS.embedding_dim])
  for word, index in ted_data.vocabulary().items():
    embedding_matrix[index+1] = glove.get(word, np.zeros(FLAGS.embedding_dim))

  sess.run(tf.global_variables_initializer(), feed_dict={init_embedding: embedding_matrix})

  step = 0
  for epoch in range(80):
    for batch, (x_batch, x_batch_l, y_batch) in enumerate(ted_data.training_batches(FLAGS.batch_size)):

      # TBTT
      state = np.zeros([FLAGS.batch_size, FLAGS.rnn_internal_size], dtype=np.float32)
      for offset in range(0, np.max(x_batch_l), 10):
        feed_dict = {X: x_batch[:,offset:offset+10], L: np.maximum(0,x_batch_l - offset),
                     Y: y_batch, initial_state: state, keep_prob: FLAGS.dropout}
        summary, state, _ = sess.run([merged, final_state, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        step += 1

      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict_valid)
      validation_writer.add_summary(summary, step)
      print('Accuracy at epoch %s, batch %s: %s' % (epoch+1, batch+1, acc))

  summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
  test_writer.add_summary(summary, step)
  print('Final test accuracy: %s' % acc)

  train_writer.close()
  validation_writer.close()
  test_writer.close()
