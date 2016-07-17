#!/usr/bin/env python3
"/netapp/home/dlituiev/ataccounts/dna_fasta_extraction"
import numpy as np
import pandas as pd
import warnings
import tables

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
#####################################################
infile="/netapp/home/dlituiev/references/regions/batf_disc1.range.1024.bed.fasta"

hdfpath = "batf_data.h5"
"initialize storage"
dfbinseq = pd.DataFrame([], 
                        columns = ["key", "seq_binary"],
                       )
#dfbinseq.centre = dfbinseq.centre.astype(int)
dfbinseq.set_index("key")

tablename = "batf_bin_seq"
#st.put(tablename, dfbinseq, format="table")

Xlist = [] 
with warnings.catch_warnings(), pd.HDFStore(hdfpath) as st, open(infile) as fi:
    warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)
    for nn, line in enumerate(fi):
        key, fasta = line.rstrip("\n").split("\t")
        chrs, pos =  splitkey(key)
        dnabin = dna_to_binary(fasta)
        #key = "%s:%u" % (chrs, pos)
        st.put( "/".join([tablename , chrs, str(pos)]),
                dnabin )
        #Xlist.append( (key, dnabin) )

        if 0 == ( (nn+1) % 500 ):
            print("line %u"  % (nn+1) )
            #dfbinseq = pd.DataFrame(Xlist, 
            #            columns = ["key", "seq_binary"])

            #dfbinseq.set_index("key")
            #st.append(tablename, dfbinseq)
            #del dfbinseq
            #Xlist = []

    #dfbinseq.to_hdf(hdfpath, "batf_bin_seq")
