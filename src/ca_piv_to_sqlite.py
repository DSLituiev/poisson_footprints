#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pandas as pd
import sqlite3

def curs_fetch_gen(curs):
    while True:
        row = curs.fetchone()
        if row is None:
            raise StopIteration
        #chrs, pos, binseq, count_str  =
        yield row

def get_pivot(datalist, r=1000):
    ds = np.zeros( 2*r +1, dtype=int )
    distance = np.r_[ [x[-2] for x in datalist] ]
    counts = [x[-1] for x in datalist]
    ds[distance + r] = counts
    #ds = pd.Series([x[-1] for x in datalist], index = [x[-2] for x in datalist])
    #ds.sort_index(inplace=True)
    return ds

def write_atac_str(curs,
    tablename,
    chrs, distance, counts,
    qry = "INSERT INTO %s VALUES(?,?,?)"
    ):
    curs.execute(qry % tablename,
                (chrs, distance, sqlite3.Binary(counts)) )

if __name__ == "__main__":
    #dbname = "batf_disc1.offsets_1000_1.pivot.db"
    #tablename = "IGTB1077"

    dbname = "../data/batf_disc1_gw.db"
    tablename = "batf_disc1_gw_atac_pivot"

    conn = sqlite3.connect(dbname)
    curs = conn.cursor()

    schema= "CREATE TABLE IF NOT EXISTS %s (chr TXT, pos INT, count_str BLOB, PRIMARY KEY(chr, pos));" % tablename
    curs.execute(schema)


    radius = 1000
    nn = 0
    commit_each = int(1e4)
    curs_get = conn.cursor()
    try:
        for chrs in ("chr%s" % (1+x) for x in range(22)):
            pos=-1
            poslist = []
            qry = """SELECT chr, centre, distance, count FROM batf_disc1_gw
                    WHERE chr='{}'
                    ORDER BY chr, centre""".format(chrs)
            curs_get.execute(qry)
            datalist = []
            print(chrs)
            for datatuple in curs_fetch_gen(curs_get):
                #print(datatuple)
                if datatuple[1] != pos and len(datalist)>0:
                    if chrs=="chr3":
                        print("pos", pos)
                    if pos in poslist:
                        print("confilicting pos", pos)
                    poslist.append(pos)
                    nn += 1
                    #key = chrs + "%u" % pos
                    position_count_line = get_pivot(datalist, r=radius)
                    count_str = position_count_line.tostring()
                    write_atac_str(curs, tablename, chrs, pos, count_str)
                    datalist = []
                    #if nn % commit_each == 0:
                    #    print(nn)
                    #    conn.commit()
                datalist.append( datatuple )
                pos = datatuple[1]
            conn.commit()
    finally:
        conn.commit()

    #pivotdir = "/Users/dlituiev/repos/qtl_atac_rna/tfmotifs/deeplearn/data/"
    #infile = pivotdir+ "IGTB1077.batf_disc1.offsets_1000_1.pivot.tab"
    #
    #print("Reading")
    #ydf = pd.read_table(infile, index_col=[0,1],)
    #print("transforming")
    #ydf.index.names = ["chr", "pos"]
    #ydf = ydf.apply( lambda x : (x.as_matrix().tostring()), axis = 1)
    #ydf = ydf.map( lambda x : sqlite3.Binary(x))
    #ydf = pd.DataFrame(ydf, columns =[ "count_str"])
    #print(ydf[:5])
    #print("Saving")
    #ydf.to_sql(tablename, conn, if_exists="append",
    #        dtype={"chr":"TEXT", "pos":"INT", "count_str":"BLOB"})
    #

    print("DONE")
