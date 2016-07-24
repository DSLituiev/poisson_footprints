#!/usr/bin/env python3
import sys
sys.path.append("freezethaw")
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from nn.nn import NeuralNetwork

from sklearn.metrics import log_loss
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

from optimizer_ft import FreezeThawOptimizer
from freezethaw import WarmLearner

from hyperopt import hp
import numpy as np
from functools import partial
import time

import tensorflow as tf
from footprint_poisson import footprint_poisson
from match_dna_atac import get_aligned_batch, get_loader
import sqlite3


CONV1_CHANNELS = [128, 256]
CONV2_CHANNELS = [16, 32]
TCONV1_CHANNELS = [16, 32]

def get_model(model, lr, wd,
        batch_norm,
        conv1_channels_ind, conv2_channels_ind, tconv1_channels_ind,
        BATCH_SIZE=2**8,
        beta1=0.9, beta2=0.999, epsilon=1e-08):
    common_opts = dict(
                batch_norm = np.array([False,True],dtype=bool)[batch_norm],
                epochs = 1,
                BATCH_SIZE = BATCH_SIZE,
                conv1_channels = CONV1_CHANNELS[conv1_channels_ind],
                conv2_channels = CONV2_CHANNELS[conv2_channels_ind],
                tconv1_channels = TCONV1_CHANNELS[tconv1_channels_ind],
                dropout = 0.5,
                xlen = 2001,
                display_step = 100,
                xdepth = 4,
                weight_decay = wd,
                learning_rate = lr,
                )

    #print("conv1_channels", conv1_channels)
    #print("conv2_channels", conv2_channels)
    if model == 0:
         return footprint_poisson( **common_opts )
    elif model==1:
         opt = partial(tf.train.AdamOptimizer, beta1=beta1, beta2=beta2, epsilon=epsilon)
         return footprint_poisson( **common_opts )
    else:
        assert False, "Unknown model %s" % model

if __name__ == "__main__":
    logfile = "hyperparam.log"
    try:
        os.remove(logfile)
    except Exception as ee:
        print(ee)
        pass

    print("# Loading data")
    mnist = fetch_mldata('MNIST original')

    dbdir = "../data/"
    dbpath = dbdir + "batf_disc1.offsets_1000_1.pivot.db"
    conn = sqlite3.connect(dbpath)
    train_batchloader = get_loader(conn, where={"chr": "chr21"}, binary=False)
    test_batchloader = get_loader(conn, where="chr = 'chr22'", binary=False)
    #X, X_val, y, y_val = train_test_split(mnist.data, mnist.target)

    common_opts = [hp.loguniform('lr', np.log(0.05), np.log(0.3)),
                      hp.loguniform('wd', np.log(0.001), np.log(0.03)),
                      hp.choice('batch_norm', [False, True]),
                      hp.choice('conv2_channels_ind', CONV1_CHANNELS),
                      hp.choice('conv1_channels_ind', CONV2_CHANNELS),
                      hp.choice('tconv1_channels_ind', TCONV1_CHANNELS)]

    adam_opts = [ hp.loguniform("epsilon", -12, -6),
                  hp.uniform('beta1', 0, 1-1e-4),
                  hp.uniform('beta2', 0, 1-1e-4),
                ]

    space = hp.choice('model',
        [
            ('adagrad', common_opts),
            #('adam', common_opts + adam_opts)
        ])

    print("# Creating model")
    ft = FreezeThawOptimizer(space, get_model, 600)

    for i in range(1000):
        print("# Iteration %s" % (i + 1))
        idx, model = ft.choose_model()

        started = time.time()
        model.fit(train_xy_loader = train_batchloader,
                    test_xy_loader = test_batchloader,
                    performance_set_size=1000,
                    load=False)
        "implement `mo.loss(X, y)` instead"

        score = -model.loss
        ft.update_model(idx, [time.time() - started], [score])
        ft.log(logfile, mode = "a")


