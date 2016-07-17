import numpy as np
import sqlite3
import pandas as pd

def blob_to_binary_dna(x):
    return np.asarray(np.unpackbits(np.fromstring(x, dtype=np.uint8),).reshape(2,-1),
                        dtype=bool)

def blob_to_int_counts(x):
    return np.asarray(np.fromstring(x, dtype=np.int64), dtype=np.int64)


def get_bin_seqs(conn):
    curs = conn.cursor()
    curs.execute("SELECT * from IGTB1077_seq WHERE chr='chr21'")
    while True:
        dna_row = curs.fetchone()
        if dna_row is None:
            raise StopIteration
        #chrs, pos, binseq, count_str  = 
        yield dna_row

def get_seq_batch(conn, size = 1, align = None):
    xx = []
    yy = []
    for nn, (chrs, pos, binseq, count_str) in enumerate(get_bin_seqs(conn)):

        x_ = blob_to_binary_dna(binseq)
        y_ = blob_to_int_counts(count_str)
        if align is not None:
            x_, y_ = align(x_, y_)
        xx.append(x_)
        yy.append(y_)
        if ((nn+1) % size) == 0:
            yield (np.transpose(
                       np.expand_dims(np.stack(xx), axis=0),
                       (1,0,3,2)),
                  np.stack(yy) )
            #yield np.stack(xx), np.stack(yy)
            xx = []
            yy = []

def align_shapes(X, y):
    start = X.shape[-1] //2 - y.shape[-1] //2
    end = X.shape[-1] - start + 1
    return X[:,start:end], y


def get_aligned_batch(conn, size = 10):
    for X, y in get_seq_batch(conn, size = size, align = align_shapes):
        yield (X, y)

from functools import partial

def get_loader(conn):
    return partial(get_aligned_batch, conn)



if __name__ == "__main__":
    #pivotdir = "/ye/yelabstore/cd4.atac.seq/all.samples/intersected_tss/batf_disc1.offsets_1000_1/pivot/"
    # dbdir = "/netapp/home/dlituiev/ataccounts/dna_fasta_extraction/"
    pivotdir = "/Users/dlituiev/repos/qtl_atac_rna/tfmotifs/deeplearn/data/"
    dbdir = "/Users/dlituiev/repos/qtl_atac_rna/tfmotifs/deeplearn/data/"

    #infile = pivotdir+ "IGTB1077.batf_disc1.offsets_1000_1.pivot.tab"
    #ydf = pd.read_table(infile, index_col=[0,1])

    dbpath = dbdir + "batf_disc1.offsets_1000_1.pivot.db"
    conn = sqlite3.connect(dbpath)

    batchloader = get_loader(conn,)
    bl = batchloader(20)
    x,y = next(bl)
    print("x", x.shape)
    print("y", y.shape)
