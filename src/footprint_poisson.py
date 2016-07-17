#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import os, sys
import re
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#######################
import tensorflow as tf
from tflearn import rtflearn, vardict, batch_norm


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
       var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

class footprint_poisson(rtflearn):
    def _create_network(self):
        self.vars = vardict()
        self.train_time = tf.placeholder(tf.bool, name='train_time')
        self.vars.x = tf.placeholder("float", shape=[None, 1, self.xlen, 2], name = "x")
        self.vars.y = tf.placeholder("float", shape=[None, self.xlen], name = "y")

        self.vars.x = batch_norm(self.vars.x, self.train_time)
        #tfs, gts = tf.unpack(tf.transpose(self.vars.x), num=2)

        # Create Model
        with tf.variable_scope('conv1') as scope:
            conv1_channels = 64
            kernel = _variable_with_weight_decay('weights', shape=[1, 5, 2, conv1_channels],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(self.vars.x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [conv1_channels], 
                                     initializer=tf.constant_initializer(0.1))
            #biases = _variable_on_cpu('biases', [conv1_channels],
            #                          tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[1, 5, 64, 64],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            biases = tf.get_variable('biases', [64],
                                     initializer=tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv2)
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 1, 3, 1],
                               strides=[1, 1, 2, 1], padding='SAME', name='pool2')
        # local3
        print("pool2", pool2.get_shape())
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            #reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            #dim = reshape.get_shape()[1].value
            dim =  np.prod(np.array([int(x) for x in pool2.get_shape()[1:]]))
            reshape = tf.reshape(pool2, [-1, dim])
            weights = _variable_with_weight_decay('weights', shape=[dim, self.xlen],
                                                  stddev=0.04, wd=0.004)
            #biases = _variable_on_cpu('biases', [self.xlen], tf.constant_initializer(0.1))
            biases = tf.get_variable('biases', [self.xlen],
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            _activation_summary(local3)

        self.vars.y_predicted = local3
        #self.vars.y_predicted = tf.reshape(self.vars.y_predicted, [-1, 1])

        #self.vars.y_predicted = gts * 1e-2
        self.saver = tf.train.Saver()
        return self.vars.y_predicted

    def _ydiff(self):
        print( "y_predicted", self.vars.y_predicted.get_shape() )
        print( "y", self.vars.y.get_shape())
        return self.vars.y_predicted - self.vars.y

    def _create_loss(self):
        # Minimize the squared errors
        print("loss")
        epsilon = 1e-5
        y_pred_pos = tf.nn.relu( self.vars.y_predicted)
        y_pred_neg_penalty = tf.reduce_mean( tf.nn.relu( - self.vars.y_predicted), name = "neg_penalty")

        poisson_loss = tf.reduce_mean(tf.abs(y_pred_pos) - self.vars.y * tf.log(1 + tf.abs(y_pred_pos)),
                                      name = "poisson_loss")
        tf.scalar_summary("poisson_loss", poisson_loss )
        tf.scalar_summary( "neg_penalty", y_pred_neg_penalty )

        tot_loss = poisson_loss + self.parameters["neg_penalty_const"] * y_pred_neg_penalty

        l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.y, 2))
        tf.scalar_summary( "loss" , tot_loss )
        #tf.scalar_summary( "y[0]" , self.vars.y_predicted[9] )
        #tf.scalar_summary( "y_hat[0]" , self.vars.y[9,0] )
        tf.scalar_summary( "l2_loss" , l2_loss )
        "R2"
        _, y_var = tf.nn.moments(self.vars.y, [0,1])
        rsq =  1 - l2_loss / y_var
        tf.scalar_summary( "R2", rsq)
        return  tot_loss

    def fit(self, train_X=None, train_Y=None,
            test_X= None, test_Y = None,
            train_xy_loader = None,
            test_xy_loader = None,
            load = True,
            epochs = None):
        if epochs:
            self.epochs = epochs
        self.last_ckpt_num = 0
        self.train = True
        #self.X = train_X
        #self.xlen = train_X.shape[1]
        self.r2_progress = []
        self.train_summary = []
        self.test_summary = []
        #yvar = train_Y.var()
        #print("variance(y) = ", yvar, file = sys.stderr)
        # n_samples = y.shape[0]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            if not ("keep_prob" in self.vars or hasattr( self.vars, "keep_prob") ):
                self.dropout = 0.0
            tot_loss = self._create_loss()
            train_op = self.optimizer( self.learning_rate).minimize(tot_loss)
            # Merge all the summaries and write them out
            summary_op = tf.merge_all_summaries()

            # Initializing the variables
            init = tf.initialize_all_variables()
            " training per se"
            train_batch_getter = train_xy_loader( self.BATCH_SIZE)
            #test_xy_loader

            # Launch the graph        
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            with tf.Session(config= sess_config) as sess:
                sess.run(init)
                if load:
                    try:
                        self._load_(sess)
                    except IOError as ex:
                        print(ex, file = sys.stderr)
                else:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                # write summaries out
                summary_writer = tf.train.SummaryWriter( self.logdir, sess.graph)
                summary_proto = tf.Summary()
                # Fit all training data
                print("training epochs: %u ... %u, saving each %u' epoch" % \
                        (self.last_ckpt_num, self.last_ckpt_num + self.epochs, self.display_step),
                        file = sys.stderr)
                for macro_epoch in tqdm(range( self.last_ckpt_num//self.display_step ,
                                         (self.last_ckpt_num + self.epochs)//  self.display_step )):
                    "do minibatches"
                    for subepoch in tqdm(range(self.display_step)):
                        for (_x_, _y_) in train_batch_getter:
                            if len(_y_.shape) == 1:
                                _y_ = np.reshape(_y_, [-1, 1])
                            #print("x", _x_.shape )
                            #print("y", _y_.shape )
                            #print(".", end="\n", file=sys.stderr)
                            if self.dropout:
                                feed_dict={ self.vars.x: _x_, self.vars.y: _y_,
                                            self.vars.keep_prob : self.dropout}
                            else:
                                feed_dict={ self.vars.x: _x_, self.vars.y: _y_ ,
                                            self.train_time: True}
                            #print("feed_dict", feed_dict)
                            sess.run(train_op, feed_dict = feed_dict)
                    epoch = macro_epoch * self.display_step
                    # Display logs once in `display_step` epochs

                    _sets_ = {"train": train_xy_loader}
                    summaries = {}
                    summaries_plainstr = []
                    if test_xy_loader is not None:
                        _sets_["test"] = test_xy_loader

                    for _set_, _xy_   in _sets_.items():
                        print("set:", _set_)
                        for (_x_, _y_) in train_batch_getter:
                            if len(_y_.shape) == 1:
                                _y_ = np.reshape(_y_, [-1, 1])

                            feed_dict={self.vars.x: _x_,
                                       self.vars.y: _y_, self.train_time: False}
                            if self.dropout:
                                feed_dict[ self.vars.keep_prob ] = self.dropout

                            summary_str = sess.run(summary_op, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, epoch)
                            summary_d = summary_dict(summary_str, summary_proto)
                            summaries[_set_] = summary_d
                            print("---set:", _set_)

                            #summary_d["epoch"] = epoch
                            #self.r2_progress.append( (epoch, summary_d["R2"]))
                            summaries_plainstr.append(  "\t".join(["",_set_] +
                                ["{:s}: {:.4f}".format(k,v) if type(v) is float else \
                                 "{:s}: {:s}".format(k,v) for k,v in summary_d.items() ]) )

                    self.train_summary.append( summaries["train"] )
                    if  "test" in summaries:
                        self.test_summary.append( summaries["test"] )

                    logstr = "Epoch: {:4d}\t".format(epoch) +\
                               "\n"+ "\n".join(summaries_plainstr)
                    print(logstr, file = sys.stderr )
                    self.saver.save(sess, self.checkpoint_dir + '/' +'model.ckpt',
                       global_step=  epoch)
                    self.last_ckpt_num = epoch
                        #0print("\tb1",  self.parameters.b1.name , self.parameters.b1.eval()[0][0] , sep = "\t")
                        #print( "W=", sess.run(W1))  # "b=", sess.run(b1)
                print("Optimization Finished!", file = sys.stderr)
#                 print("cost = ", sess.run( tot_loss , feed_dict={self.vars.x: train_X, self.vars.y: np.reshape(train_Y, [-1, 1]) }) )
#                 print("W1 = ", sess.run(self.parameters.W1), )
#                 print("b1 = ", sess.run(self.parameters.b1) )
        return self


if __name__ == "__main__":

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    FLAGS.batch_size = 20

    # define flags (note that Fomoro will not pass any flags by default)
    flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
    flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

    # define artifact directories where results from the session can be saved
    model_path = os.environ.get('MODEL_PATH', 'models/')
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
    summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

    "paths to the data sets"
    pivotdir = "../data/"
    dbdir = "../data/"

    #infile = pivotdir+ "IGTB1077.batf_disc1.offsets_1000_1.pivot.tab"
    #nrows = None
    #ydf = pd.read_table(infile, index_col=[0,1], nrows = nrows)

    dbpath = dbdir + "batf_disc1.offsets_1000_1.pivot.db"
    import sqlite3
    conn = sqlite3.connect(dbpath)

    from match_dna_atac import get_aligned_batch, get_loader
    batchloader = get_loader(conn,)

    trainsamples = 4000

    tfl = footprint_poisson(ALPHA = 2e-6,
            BATCH_SIZE = 2**8,
            dropout = False, xlen = 2001,
            display_step = 20,
            )
    tfl.parameters["neg_penalty_const"] = 0.01
    tfl.fit( train_xy_loader= batchloader)

