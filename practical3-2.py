import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from data import TedDataSeq

class Model(object):
  def __init__(self, is_training, config):
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps

    X = tf.placeholder(tf.int32, [config.batch_size, None], name="X")
    Y_ = tf.one_hot(tf.placeholder(tf.int32, [config.batch_size, None]), config.vocab_size, name="Y_")

    cell = rnn.GRUCell(config.hidden_size)
    # if is_training and config.keep_prob < 1:
    #   cell = rnn.DropoutWrapper(cell, input_keep_prob=config.keep_prob)

    cells = rnn.MultiRNNCell([cell]*config.num_layers, state_is_tuple=False)
    # if is_training and config.keep_prob < 1:
    #   cells = rnn.DropoutWrapper(cells, output_keep_prob=config.keep_prob)

    Yr, H = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)

    # naming
    H = tf.identity(H, name='H')

    Yflat = tf.reshape(Yr, [-1, config.hidden_size])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, config.vocab_size)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Yflat_ = tf.reshape(Y_, [-1, config.vocab_size])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [config.batch_size, -1])      # [ BATCHSIZE, SEQLEN ]
    Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [config.batch_size, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
    self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)


    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

    # self.state = np.zeros([config.batch_size, config.hidden_size*config.num_layers])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

  def set_lr(self, session, lr_value):
      session.run(tf.assign(self.lr, lr_value))

  def run_epoch(session, data):
    """Runs the model on the given data."""
    # start_time = time.time()
    # costs = 0.0
    # iters = 0
    for x, y in data.training_batches():
      print(x.shape,y.shape)
      print("FDKFJDFHDI:FJDKFJDF")
      cost, state, _ = session.run([cost, self.train_step],
                                   {X: x, Y_: y})
    #   costs += cost
    #   iters += model.num_steps

    #   if self.is_training and step % (epoch_size // 10) == 10:
    #         print("%.3f perplexity: %.3f speed: %.0f wps" %
    #           (step * 1.0 / epoch_size, np.exp(costs / iters),
    #            iters * model.batch_size / (time.time() - start_time)))

    # return np.exp(costs / iters)

class Config(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 0.5
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def main(_):
    config = Config()
    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    data = TedDataSeq(config.vocab_size)

    with tf.Graph().as_default(), tf.Session() as session:
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

      with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = Model(is_training=True, config=config)

      with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = Model(is_training=False, config=config)
        mtest = Model(is_training=False, config=eval_config)

      tf.global_variables_initializer().run()

      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        model.set_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
        train_perplexity = model.run_epoch(session, data)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
      print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
