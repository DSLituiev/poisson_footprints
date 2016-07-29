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
from tflearn import rtflearn, vardict,  summary_dict, batch_norm
#from tensorflow.contrib.layers import batch_norm


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def diff(x, axis=-1):
    xsh = x.get_shape()
    xsh = np.r_[[int(c) for c in xsh]]
    begin = np.zeros(len(xsh),dtype=np.int32)
    begin[axis] = 1
    y = tf.slice(x, begin=begin, size=[-1,]*len(xsh)) -\
        tf.slice(x, [0,]*len(xsh), xsh-begin)
    return y

def softmax(target, axis, name=None):
    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    with tf.op_scope([target], name, 'softmax'):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
    return softmax

def poisson_loss(y, log_y_predicted, pad=0):
  y_pred_pos = tf.exp(log_y_predicted)
  vector_loss = y_pred_pos - y * log_y_predicted + tf.lgamma(y+1)
  b = tf.shape(vector_loss,)[0]
  h = tf.shape(vector_loss,)[1]
  #w = tf.shape(vector_loss,)[2]
  #d = tf.shape(vector_loss,)[3]

  if pad!=0:
      #mask = np.ones(tf.shape(vector_loss,))
      #mask[:,:,pad:-pad-1,:] = 0
      #mask = tf.constant(mask, trainable=False)
      vector_loss = tf.slice(vector_loss,[0,pad],[b,h-pad])

  poisson_loss = tf.reduce_mean(y_pred_pos - y * log_y_predicted,
                                name = "poisson_loss")
  return poisson_loss

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


from tensorflow.python import control_flow_ops

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

def smooth_filter(name, shape, stddev, wd, axis=1):
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
    #var = _variable_on_cpu(name, shape,
    #                       tf.truncated_normal_initializer(stddev=stddev))
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
            #tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        sqd = tf.reduce_mean( diff(var,axis=axis)**2 )/2
        weight_decay = tf.mul(sqd, wd, name='smoothness_loss')
        tf.add_to_collection('losses', weight_decay)
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
    #var = _variable_on_cpu(name, shape,
    #                       tf.truncated_normal_initializer(stddev=stddev))
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
            #tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

class footprint_poisson(rtflearn):
    def _create_network(self):
        #print("conv1_channels", self.conv1_channels)
        #print("conv2_channels", self.conv2_channels)
        weight_decay = self.weight_decay
        conv_wd=self.weight_decay
        self.vars = vardict()
        self.train_time = tf.placeholder(tf.bool, name='train_time')
        self.vars.x = tf.placeholder("float", shape=[None, 1, self.xlen, self.xdepth], name = "x")
        self.vars.y = tf.placeholder("float", shape=[None, self.xlen], name = "y")

        self.vars.x = batch_norm(self.vars.x, is_training=self.train_time)
        print("x placeholder", self.vars.x.get_shape())
        # Need the batch size for the transpose layers.
        batch_size = tf.shape(self.vars.x)[0]

        # Create Model
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                            shape=[1, 5, self.xdepth, self.conv1_channels],
                            stddev=1e-4, wd= conv_wd)
            conv = tf.nn.conv2d(self.vars.x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [self.conv1_channels],
                                     initializer=tf.constant_initializer(0.01))
            #biases = _variable_on_cpu('biases', [conv1_channels],
            #                          tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            print("conv1", conv1.get_shape())
            #conv1 = tf.nn.dropout(conv1, 1-self.dropout)
            if self.batch_norm:
                conv1 = batch_norm(conv1,
                    is_training=self.train_time, n_out=self.conv1_channels, scope=scope)
            conv1 = softmax(conv1,2)
            _activation_summary(conv1)
        # pool1
        #pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1],
        #                       padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                  name='norm1')
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                    shape=[1, 3, self.conv1_channels, self.conv2_channels],
                                    stddev=1e-4, wd=conv_wd)
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            biases = tf.get_variable('biases', [self.conv2_channels],
                                     initializer=tf.constant_initializer(0.01))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            print("conv2", conv2.get_shape())
            _activation_summary(conv2)
            conv2 = tf.nn.dropout(conv2, 1-self.dropout)
            if self.batch_norm:
                conv2 = batch_norm(conv2, is_training=self.train_time,
                               n_out=self.conv2_channels, scope=scope)

        with tf.variable_scope('conv3') as scope:
            kernel = smooth_filter('weights',
                                    shape=[1, 11, self.conv1_channels, self.conv2_channels],
                                    stddev=1e-4, wd=conv_wd,
                                    axis=2)
            conv3 = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            biases = tf.get_variable('biases', [self.conv2_channels],
                                     initializer=tf.constant_initializer(0.01))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            print(scope.name, conv3.get_shape())
            _activation_summary(conv3)
            conv3 = tf.nn.dropout(conv2, 1-self.dropout)
            if self.batch_norm:
                conv3 = batch_norm(conv3, is_training=self.train_time,
                               n_out=self.conv2_channels, scope=scope)


        gate_2_3 = tf.Variable(tf.constant([0.5,0.5], dtype=np.float32), name="gate_2_3")
        gate_2_3 = gate_2_3/tf.reduce_sum(gate_2_3)
        conv_2_3 = conv2*gate_2_3[0] + conv3*gate_2_3[1]
            #conv2 = softmax(conv2, 2)
        #map_sparsity = tf.add(tf.reduce_mean(tf.abs(conv2)),
        #        tf.reduce_mean((conv2)**2,)/2,
        #        name = "map0_sparsity")
        #tf.add_to_collection('losses', self.sparsity * map_sparsity)
        # norm2
        #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                  name='norm2')
        # pool2
        #pool2 = tf.nn.max_pool(norm2, ksize=[1, 1, 3, 1],
        #                       strides=[1, 1, 2, 1], padding='SAME', name='pool2')
        # tconv1 
        #print("tconv1", pool2.get_shape())
        with tf.variable_scope('tconv1') as scope:
            kernel_h = 1
            kernel_w = 5
            stride_h = 1
            stride_w = 1
            pad_h = 1
            pad_w = 1
            kernel = _variable_with_weight_decay('weights',
                            shape=[kernel_h, kernel_w, self.tconv1_channels, self.conv2_channels],
                            stddev=1e-4, wd=conv_wd)
            inpshape = tf.shape(conv2)
            #print(scope.name, "inpshape", inpshape) 
            h = ((inpshape[1] - 1) * stride_h) + kernel_h - 2 * pad_h
            w = ((inpshape[2] - 1) * stride_w) + kernel_w - 2 * pad_w
            #output_shape =  [batch_size, h, w, self.xlen]
            output_shape =  [batch_size, (inpshape[1] + stride_h - 1),
                                (inpshape[2] + stride_w - 1) , self.tconv1_channels]
            print(scope.name, output_shape)
            output_shape = tf.pack(output_shape)
            tconv1 = tf.nn.conv2d_transpose(conv2, kernel, output_shape, strides=[1,1,1,1],
                    padding='SAME', name=None)
            #tconv1 = batch_norm(tconv1, is_training=self.train_time,
            #                    n_out=self.tconv1_channels, scope=scope)

            _activation_summary(tconv1)

        #map_sparsity = tf.add(tf.reduce_mean(tf.abs(tconv1)),
        #        tf.reduce_mean((tconv1)**2,)/2, name ="map1_sparsity")
        #tf.add_to_collection('losses', self.sparsity * map_sparsity)

        with tf.variable_scope('tconv2') as scope:
            kernel_h = 1
            kernel_w = 5
            stride_h = 1
            stride_w = 1
            pad_h = 1
            pad_w = 1
            output_channels = 1
            kernel = _variable_with_weight_decay('weights',
                            shape=[kernel_h, kernel_w,
                                   output_channels, self.tconv1_channels],
                            stddev=1e-4, wd=conv_wd)
            inpshape = tf.shape(tconv1)
            h = ((inpshape[1] - 1) * stride_h) + kernel_h - 2 * pad_h
            w = ((inpshape[2] - 1) * stride_w) + kernel_w - 2 * pad_w
            #output_shape =  [batch_size, h, w, self.xlen]
            output_shape =  [batch_size, (inpshape[1] + stride_h - 1),
                                (inpshape[2] + stride_w - 1) , output_channels]
            print(scope.name, output_shape)
            output_shape = tf.pack(output_shape)
            tconv2 = tf.nn.conv2d_transpose(tconv1, kernel,
                            output_shape, strides=[1,1,1,1],
                    padding='SAME', name=None)
            #tconv2 = batch_norm(tconv2, is_training=self.train_time)#, n_out=output_channels, scope=scope)
            tconv2 = tf.reshape(tconv2, [-1, self.xlen])
            _activation_summary(tconv2)

        self.vars.y_predicted = tconv2
        #self.vars.y_predicted = tf.reshape(self.vars.y_predicted, [-1, 1])

        #self.vars.y_predicted = gts * 1e-2
        self.saver = tf.train.Saver()
        return self.vars.y_predicted

    def _ydiff(self):
        print( "y_predicted", self.vars.y_predicted.get_shape() )
        print( "y", self.vars.y.get_shape())
        return tf.exp(self.vars.y_predicted) - self.vars.y

    def _create_loss(self):
        #print("loss")
        poisson_loss_ = poisson_loss(self.vars.y, self.vars.y_predicted, pad = 8)
        tf.add_to_collection('losses', poisson_loss_)

        tf.scalar_summary("poisson_loss", poisson_loss_ )

        self._loss_ = tf.to_float(self.train_time) * tf.add_n(tf.get_collection('losses'), name='total_loss') +\
                    tf.to_float(~self.train_time)*poisson_loss_

        l2_loss = tf.reduce_mean(tf.pow( tf.exp(self.vars.y_predicted) - self.vars.y, 2))
        tf.scalar_summary( "loss" , self._loss_)
        #tf.scalar_summary( "y[0]" , self.vars.y_predicted[9] )
        #tf.scalar_summary( "y_hat[0]" , self.vars.y[9,0] )
        tf.scalar_summary( "l2_loss" , l2_loss )
        "R2"
        # _, y_var = tf.nn.moments(self.vars.y, [0,1])
        # rsq =  1 - l2_loss / y_var
        # tf.scalar_summary( "R2", rsq)
        return self._loss_

    def fit(self, train_X=None, train_Y=None,
            test_X= None, test_Y = None,
            train_xy_loader = None,
            test_xy_loader = None,
            load = True,
            epochs = None,
            performance_set_size=int(1e3)):
        if epochs:
            self.epochs = epochs
        self.last_ckpt_num = 0
        self.train = True
        #self.X = train_X
        #self.xlen = train_X.shape[1]
        self.loss=0
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
                #for macro_epoch in tqdm(range( self.last_ckpt_num//self.display_step ,
                #                         (self.last_ckpt_num + self.epochs)//  self.display_step )):
                "do minibatches"
                for epoch in tqdm(range(self.epochs)):
                    for ii, (_x_, _y_) in enumerate(train_batch_getter):
                        if len(_y_.shape) == 1:
                            _y_ = np.reshape(_y_, [-1, 1])
                        #print("x", _x_.shape )
                        #print("y", _y_.shape )
                        #print(".", end="\n", file=sys.stderr)
                        if self.dropout:
                            feed_dict={ self.vars.x: _x_, self.vars.y: _y_,
                                        self.vars.keep_prob : self.dropout,
                                        self.train_time: True}
                        else:
                            feed_dict={ self.vars.x: _x_, self.vars.y: _y_ ,
                                        self.train_time: True}
                        #print("feed_dict", feed_dict)
                        sess.run(train_op, feed_dict = feed_dict)
                    "Display logs once in `display_step` epochs"

                    train_batch_getter = train_xy_loader( self.BATCH_SIZE)
                    _sets_ = {"train": train_batch_getter}
                    #print("self.BATCH_SIZE", self.BATCH_SIZE)
                    if test_xy_loader is not None:
                        test_batch_getter = test_xy_loader( self.BATCH_SIZE)
                        _sets_["test"] = test_batch_getter

                    summaries = {}
                    summaries_plainstr = []

                    for _set_, _xy_ in _sets_.items():
                        #print("set:", _set_)
                        (_x_, _y_) = next(_xy_)
                        if len(_y_.shape) == 1:
                            _y_ = np.reshape(_y_, [-1, 1])

                        feed_dict={self.vars.x: _x_,
                                   self.vars.y: _y_, self.train_time: False}
                        if self.dropout:
                            feed_dict[ self.vars.keep_prob ] = self.dropout

                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_d = summary_dict(summary_str, summary_proto)
                        summaries[_set_] = summary_d
                        if _set_ == "test":
                            summary_writer.add_summary(summary_str, epoch)
                            self.loss +=summary_d["loss"]
                        #print("---set:", _set_)
                        #summary_d["epoch"] = epoch
                        summaries_plainstr.append(  "\t".join(["", _set_] +
                            ["{:s}: {:.4f}".format(k,v) if type(v) is float else \
                             "{:s}: {:s}".format(k,v) for k,v in summary_d.items() ]) )

                    self.train_summary.append( summaries["train"] )
                    if  "test" in summaries:
                        self.test_summary.append( summaries["test"] )

                    logstr = "Epoch: {:4d}\t".format(epoch) +\
                               "\n"+ "\n".join(summaries_plainstr)
                    print(logstr, file = sys.stderr )
                    print("="*40, file = sys.stderr )
                    self.saver.save(sess, self.checkpoint_dir + '/' +'model.ckpt',
                       global_step=  epoch)
                    self.last_ckpt_num = epoch
                print("Optimization Finished!", file = sys.stderr)
        return self

    def get_loss(self, test_xy_loader=None, minibatches=None, load=False):
        test_batch_getter = test_xy_loader( self.BATCH_SIZE)
        sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                   intra_op_parallelism_threads= self.NUM_CORES)
        loss = 0
        g = tf.Graph()
        with g.as_default():
            "fetch a placeholder of the predicted variable"
            ph_y_predicted = self._create_network()
            if not ("keep_prob" in self.vars or hasattr( self.vars, "keep_prob") ):
                self.dropout = 0.0
            self._loss_ = self._create_loss()
            summary_op = tf.merge_all_summaries()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            # Initializing the variables
            init = tf.initialize_all_variables()

            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                else:
                    sess.run(init)
                for ii, (_x_, _y_) in enumerate(test_batch_getter):
                    feed_dict={self.vars.x: _x_,
                               self.vars.y: _y_, self.train_time: False}
                    loss += sess.run(self._loss_, feed_dict=feed_dict)
                    if minibatches and (ii>=minibatches):
                        break
        return loss



if __name__ == "__main__":
    import sqlite3

    flags = tf.app.flags
    flags.DEFINE_boolean('predict', False, 'If true, predicts')
    FLAGS = flags.FLAGS
    print(flags.FLAGS)
    FLAGS.batch_size = 128

    # define flags (note that Fomoro will not pass any flags by default)
    flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
    flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

    # define artifact directories where results from the session can be saved
    model_path = os.environ.get('MODEL_PATH', 'models/')
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
    summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

    "paths to the data sets"
    dbdir = "../data/"
    dbpath = dbdir + "batf_disc1_gw.db"
    conn = sqlite3.connect(dbpath)

    from match_dna_atac import get_aligned_batch, get_loader
    #from itertools import cycle
    train_batchloader = get_loader(conn, where={"chr": "chr20"}, binary=False)
    test_batchloader = get_loader(conn, where="chr = 'chr22'", binary=False)

    #sys.exit(1)
    trainsamples = 4000

    "initialize the object"
    tfl = footprint_poisson(
            sparsity = 1e-2,
            batch_norm = False,
            BATCH_SIZE = 2**7,
            dropout = 0.25,
            xlen = 2001,
            display_step = 100,
            xdepth = 4,
            weight_decay = 0.02583,
            conv1_channels = 128,
            conv2_channels = 32,
            tconv1_channels = 32,
            lr = 0.05,
            )
    print(tfl.parameters.keys())
    if not FLAGS.predict:
        tfl.fit( train_xy_loader = train_batchloader,
                test_xy_loader = test_batchloader,
                performance_set_size=1000,
                epochs=250)
        print(tfl.loss)
        print(tfl.get_loss(test_batchloader))
    else:
        print("predicting")
        testbl = test_batchloader(500)
        yhat_list = []
        y_list = []
        try:
            for nn, (xx, yy) in enumerate(testbl):
                tfl = footprint_poisson(
                        sparsity = 1e-2,
                        batch_norm = False,
                        BATCH_SIZE = 2**8,
                        dropout = 0.5,
                        xlen = 2001,
                        display_step = 100,
                        xdepth = 4,
                        weight_decay = 0.02583,
                        conv1_channels = 128,
                        conv2_channels = 32,
                        tconv1_channels = 32,
                        lr = 0.02,
                        )

                print(nn)
                yhat = tfl.predict(xx)
                yhat_list.append(yhat)
                y_list.append(yy)
                break
        finally:
            tt = np.arange(len(yhat[0])) - 1000

            valid = abs(tt) < 50
            yhat_mean = np.mean(np.mean(np.stack(yhat_list), axis=0), axis=0 )
            print("yhat_mean", yhat_mean.shape)
            yhat_var = np.mean(np.var( np.exp(np.stack(yhat_list)), axis=0 ), axis=0)
            y_mean = np.mean(np.mean( np.stack(y_list), axis=0 ), axis=0)
            print("y_mean", y_mean.shape)
            print(yhat_var.shape)
            fig, axs = plt.subplots(2)
            axs[0].plot(tt[valid], np.exp(yhat_mean[valid]), c="b", zorder=1, lw=2 )
            axs[0].plot(tt[valid], (yhat_var[valid]), c="g", zorder=2 )
            axs[0].set_ylim(np.exp(np.r_[ min(yhat_mean[valid]), max(yhat_mean[valid]) ]))
            print(min(yhat_var[valid]), max(yhat_var[valid]))
            #axs[0].set_xlim([0, np.exp(max(yhat[0]))+0.01 ])
            #for t_ in tt[ (y_mean>0) & valid]:
            #    axs[0].axvline(t_, c='r')
            #ax.scatter(tt, yy[0],c=(1,0,0,1), edgecolors="none", zorder=2 )
            axs[1].plot(tt[valid], y_mean[valid], c='r')
            #axs[1].stem(tt[valid], y_mean[valid], markerfmt='ro',linefmt='r-',
            #            edgecolors="none", zorder=2 )
            #axs[1].set_xlim([0, max(yy[0]+1)])
            fig.savefig("pred_%u.eps" % nn, format="eps")
            fig.clear()

