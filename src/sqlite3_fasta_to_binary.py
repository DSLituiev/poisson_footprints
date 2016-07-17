#!/usr/bin/env python3
"/netapp/home/dlituiev/ataccounts/dna_fasta_extraction"
import numpy as np
import pandas as pd
import pickle
import warnings
#import tables
import pickle
import sys

def pickle_loader(filename):
    """an iterator"""
    with open(filename, 'rb') as f:
        while 1:
            try:
                yield pickle.load(f)
            except EOFError:
                raise StopIteration

def splitkey(key):
    chrs, pos = key.split(":")
    pos = sum(map(int, pos.split("-")))//2
    return chrs, pos

def dna_to_binary(fasta):
        purine = (np.asarray(list(fasta.lower()), dtype = "S1") == b"a") | \
        (np.asarray(list(fasta.lower()), dtype = "S1") == b"g")
        a_or_t = (np.asarray(list(fasta.lower()), dtype = "S1") == b"a") | \
        (np.asarray(list(fasta.lower()), dtype = "S1") == b"t")
        dnabin =  np.vstack([purine, a_or_t])
        return pd.DataFrame(dnabin, index = ["purin", "at"])

def bool_array_to_sqlite_blob(x):
    return sqlite3.Binary(np.packbits(x).tostring())
#####################################################
infile="/netapp/home/dlituiev/references/regions/batf_disc1.range.1024.bed.fasta"

#picklepath = "batf_data.pickle"
"initialize storage"
dfbinseq = pd.DataFrame([], 
                        columns = ["key", "seq_binary"],
                       )
#dfbinseq.centre = dfbinseq.centre.astype(int)
dfbinseq.set_index("key")

tablename = "batf_bin_seq"
#st.put(tablename, dfbinseq, format="table")

Xlist = [] 
#    open(picklepath,'ab') as st, \
import sqlite3
dbname = "bin_seq.db"

#    open(picklepath,'ab') as st, \
with sqlite3.connect(dbname) as conn, \
     warnings.catch_warnings(), \
     open(infile) as fi:
    curs = conn.cursor()
    curs.execute("CREATE TABLE IF NOT EXISTS %s (chr TEXT, pos INT, bin_seq BOOL, PRIMARY KEY(chr, pos))" % tablename )
    insert_qry = "INSERT INTO %s (chr, pos, bin_seq) VALUES (?,?,?)"  % tablename
    #warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)
    for nn, line in enumerate(fi):
        key, fasta = line.rstrip("\n").split("\t")
        chrs, pos =  splitkey(key)
        dnabin = dna_to_binary(fasta)
        #key = "%s:%u" % (chrs, pos)
        key = "/".join([tablename , chrs, str(pos)])
        print((dnabin) )
        sys.exit(1)
        #print(np.packbits(dnabin).shape )
        #print(np.packbits(dnabin).tostring() )

        curs.execute(insert_qry,
            (chrs, pos,  bool_array_to_sqlite_blob(dnabin) )
                    )
        #st.dump( ( key, dnabin ) )
        #Xlist.append( (key, dnabin) )

        if 0 == ( (nn+1) % 1000 ):
            print("line %u"  % (nn+1) )
            #break
            #dfbinseq = pd.DataFrame(Xlist, 
            #            columns = ["key", "seq_binary"])

            #dfbinseq.set_index("key")
            #st.append(tablename, dfbinseq)
            #del dfbinseq
            #Xlist = []

    #dfbinseq.to_hdf(hdfpath, "batf_bin_seq")
