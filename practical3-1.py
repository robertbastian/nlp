import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers
from data import TedDataWithLabels, Glove
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 80, 'Number of training epochs')
flags.DEFINE_integer('vocab_size', 1000, 'Size of vocabulary derived from text corpus')
flags.DEFINE_integer('embedding_dim', 50, 'Dimension of embedding vectors')
flags.DEFINE_integer('batch_size', 50, 'Training batch size')
flags.DEFINE_integer('rnn_internal_size', 20, 'Size of the RNN\'s state')
flags.DEFINE_integer('hidden_layer_size', 50, 'Size of the hidden layer')
flags.DEFINE_float('dropout', 1.0, 'Keep probability for training dropout.')
flags.DEFINE_string('summaries_dir', 'logs'+time.strftime("%H%M%S"), 'Summaries directory')


X = tf.placeholder(tf.int64, [None, None])
# [raw_batch_size, max_length]
Y = tf.placeholder(tf.int64, [None])
# [raw_batch_size]
L = tf.placeholder(tf.int64, [None])
# [raw_batch_size]
initial_states = tf.placeholder(tf.float32, [None, FLAGS.rnn_internal_size])
# [raw_batch_size, internal_size]

# TBTT will lead to some sequences being empty, filter those out
non_empty = tf.not_equal(L, tf.constant(0, tf.int64))
# [raw_batch_size]

X_ = tf.boolean_mask(X, non_empty)
# [batch_size, max_length]
Y_ = tf.boolean_mask(Y, non_empty)
# [batch_size]
L_ = tf.boolean_mask(L, non_empty)
# [batch_size]
initial_states_ = tf.boolean_mask(initial_states, non_empty)
# [batch_size, internal_size]

init_embedding = tf.placeholder(tf.float32, [FLAGS.vocab_size, FLAGS.embedding_dim])
embedding = tf.Variable(init_embedding)
# [vocab_size, embedding_dim]
Xe = tf.nn.embedding_lookup(embedding, X_)
# [batch_size, max_length, embedding_dim]

cell = rnn.GRUCell(FLAGS.rnn_internal_size)
Yr, final_states_ = tf.nn.dynamic_rnn(cell, Xe, sequence_length=L_, initial_state=initial_states_)
# [batch_size, max_length, internal_size], [batch_size, internal_size]
Yavg = tf.div(tf.reduce_sum(Yr, axis=1), tf.cast(tf.expand_dims(L_,1), tf.float32))
# [batch_size, internal_size]

hidden_layer = layers.fully_connected(Yavg, FLAGS.hidden_layer_size, activation_fn=tf.nn.tanh)
# [batch_size, hidden_layer_size]

keep_prob = tf.placeholder_with_default(1.0, [])
dropped = tf.nn.dropout(hidden_layer, FLAGS.dropout)
# [batch_size, hidden_layer_size]

Yh = layers.fully_connected(dropped, 8, activation_fn=tf.identity)
# [batch_size, 8]

diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y_, depth=8), logits=Yh)
cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

prediction = tf.argmax(Yh, 1)
correct_prediction = tf.equal(Y_, prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

merged = tf.summary.merge_all()

with tf.Session() as sess:

  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

  def startStates(input_size):
    return np.zeros([input_size, FLAGS.rnn_internal_size], dtype=np.float32)

  ted_data = TedDataWithLabels(FLAGS.vocab_size)
  feed_dict_valid = {X: ted_data.x_valid, Y: ted_data.y_valid, L: ted_data.x_valid_l,
                     initial_states: startStates(ted_data.x_valid.shape[0])}
  feed_dict_test  = {X: ted_data.x_test,  Y: ted_data.y_test,  L: ted_data.x_test_l,
                     initial_states: startStates(ted_data.x_test.shape[0])}

  glove = Glove(FLAGS.embedding_dim)
  embedding_matrix = np.zeros([FLAGS.vocab_size, FLAGS.embedding_dim])
  for word, index in ted_data.vocabulary().items():
    embedding_matrix[index+1] = glove.get(word, np.zeros(FLAGS.embedding_dim))

  sess.run(tf.global_variables_initializer(), feed_dict={init_embedding: embedding_matrix})

  try:
    step = 0
    for epoch in range(FLAGS.epochs):
      for batch, (x_batch, x_batch_l, y_batch) in enumerate(ted_data.training_batches(FLAGS.batch_size)):

        # TBTT
        states = startStates(x_batch.shape[0])
        for offset in range(0, np.max(x_batch_l), 10):
          feed_dict = {X: x_batch[:,offset:offset+10], L: np.maximum(0,x_batch_l - offset),
                       Y: y_batch, initial_states: states, keep_prob: FLAGS.dropout}
          states_r, ne, _ = sess.run([final_states_, non_empty, train_step], feed_dict=feed_dict)
          states[np.where(ne)] = states_r
        step += 1

        train_sum, train_acc = sess.run([merged, accuracy],
          feed_dict={X: x_batch, Y: y_batch, L: x_batch_l, initial_states: startStates(x_batch.shape[0])})
        train_writer.add_summary(train_sum, step)

        valid_summary, valid_acc = sess.run([merged, accuracy], feed_dict=feed_dict_valid)
        validation_writer.add_summary(valid_summary, step)

        print('Accuracy at Epoch {:d}.{:02d}: {:2.2f}% - {:2.2f}%'.format(epoch+1, batch+1, train_acc*100, valid_acc*100))

    raise KeyboardInterrupt

  except KeyboardInterrupt:
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict_test)
    test_writer.add_summary(summary, step)
    print('\nFinal test accuracy: {:2.2f}%'.format(acc*100))

    train_writer.close()
    validation_writer.close()
    test_writer.close()
