import sys
import sqlite3
import numpy as np
import pandas as pd

def blob_to_binary_dna(x):
    return np.asarray(np.unpackbits(np.fromstring(x, dtype=np.uint8),).reshape(2,-1),
                        dtype=bool)

def blob_to_quaternary_dna(x):
    y2 = blob_to_binary_dna(x)
    y4 = np.c_[
                y2[0] & y2[1],
                (~y2[0]) & y2[1],
                y2[0] & (~y2[1]),
                (~y2[0]) & (~y2[1]),
                ].T
    return y4

def blob_to_int_counts(x):
    return np.asarray(np.fromstring(x, dtype=np.int64), dtype=np.int64)

DNADICT = dict(A = np.r_[True, False, False, False],
    T = np.r_[False, True, False, False],
    G = np.r_[False, False, True, False],
    C = np.r_[False, False, False, True],
    )
DNALIST = ["A", "T", "G", "C"]

def int_to_nucleotide(x):
    return DNALIST[x]

def onehot4_to_dna(x):
    comparisondim = np.where(np.array(x.shape) == 4)[0][0]
    ind =  np.argmax(x, axis=comparisondim)
    xstr = np.vectorize(int_to_nucleotide)(ind)
    return xstr

def count_rows(conn, tablename="batf_seq_dna_atac"):
    qry = "SELECT MAX(ROWID) FROM {0}".format(tablename)
    curs = conn.cursor()
    curs.execute(qry)
    return curs.fetchone()[0]

def get_bin_seq_random(conn, tablename="batf_seq_dna_atac", fraction = 1/2, num=1):
    """set negative fraction to pick elements from the end of the table"""
    nrows = count_rows(conn, tablename)
    total = int(nrows * abs(fraction))
    if fraction > 0:
        start = 0
    else:
        start = int(nrows * (1+fraction))
    # x = random % total + start
    qry="""SELECT * FROM {tablename}
        WHERE ROWID IN
            (SELECT ABS(RANDOM()) % {total} + {start}
            FROM {tablename} );""".\
            format(tablename=tablename,
                   start=start, total=total)
    #LIMIT {num}
    curs = conn.cursor()
    curs.execute(qry)
    while True:
        dna_row = curs.fetchone()
        if dna_row is None:
            raise StopIteration
        #chrs, pos, binseq, count_str  =
        yield dna_row

def get_seq_batch_random(conn, size = 1, align = None,
                         tablename="batf_seq_dna_atac", fraction = 1/2,
                         binary=True):
    xx = []
    yy = []
    fetcher = get_bin_seq_random(conn, tablename=tablename, fraction = fraction, num=1)
    decoder = blob_to_binary_dna if binary else blob_to_quaternary_dna
    for nn, (chrs, pos, binseq, count_str) in \
            enumerate(fetcher):
        #print(chrs, pos, file=sys.stderr)
        x_ = decoder(binseq)
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
            xx = []
            yy = []


def get_bin_seqs(conn, where=None, tablename="batf_seq_dna_atac"):
    if where not in ("", None):
        if type(where) is str:
            where = " WHERE " + where
        elif type(where) is dict:
            where = " WHERE " +\
                    " AND ".join(["{0} = '{1}'".format(kk, vv) for kk,vv in where.items()])
    else:
        where = ""
    curs = conn.cursor()
    curs.execute(" ".join(["SELECT * from", tablename, where]))
    while True:
        dna_row = curs.fetchone()
        if dna_row is None:
            raise StopIteration
        #chrs, pos, binseq, count_str  = 
        yield dna_row

def get_seq_batch(conn, size = 1, align = None, where=None, binary=True):
    xx = []
    yy = []
    decoder = blob_to_binary_dna if binary else blob_to_quaternary_dna
    for nn, (chrs, pos, binseq, count_str) in \
            enumerate(get_bin_seqs(conn, where=where)):
        #print(chrs, pos)
        x_ = decoder(binseq)
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

from functools import partial

def get_loader(conn, where=None, binary=True, fraction=0.0,
                                         tablename="batf_seq_dna_atac",):
    #out = partial(get_aligned_batch, conn, where=where, binary=binary)
    if (where is None) and (fraction != 0.0):
        out = partial(get_seq_batch_random, conn, align=align_shapes,
                      fraction=fraction, binary=binary)
    else:
        out = partial(get_seq_batch, conn, align=align_shapes,
                      where=where, binary=binary)
    out.__repr__ = """loader:
    sql:    {0}
    binary: {1}""".format(where, binary)
    return out


if __name__ == "__main__":
    pivotdir = "../data/"
    dbdir = "../data/"

    #infile = pivotdir+ "IGTB1077.batf_disc1.offsets_1000_1.pivot.tab"
    #ydf = pd.read_table(infile, index_col=[0,1])

    dbpath = dbdir + "batf_disc1.offsets_1000_1.pivot.db"
    conn = sqlite3.connect(dbpath)

    batchloader = get_loader(conn, binary=False)
    bl = batchloader(20)
    x,y = next(bl)
    print("x", x.shape)
    print("y", y.shape)

    print("x", x[0][0].T.shape)
    print("x", x[0][0].T[:30])

    #onehot4_to_dna = np.vectorize(onehot4_to_dna)
    #print(onehot4_to_dna(x[0]))
    print( "".join(onehot4_to_dna(x[0][0])) )

