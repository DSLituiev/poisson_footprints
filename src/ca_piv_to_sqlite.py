import pandas as pd
import numpy as np
import sqlite3

dbname = "batf_disc1.offsets_1000_1.pivot.db"
tablename = "IGTB1077"

conn = sqlite3.connect(dbname)
curs = conn.cursor()

schema= "CREATE TABLE IF NOT EXISTS %s (chr TXT, pos INT, count_str BLOB);" % tablename
curs.execute(schema)

pivotdir = "/Users/dlituiev/repos/qtl_atac_rna/tfmotifs/deeplearn/data/"
infile = pivotdir+ "IGTB1077.batf_disc1.offsets_1000_1.pivot.tab"

print("Reading")
ydf = pd.read_table(infile, index_col=[0,1],)
print("transforming")
ydf.index.names = ["chr", "pos"]
ydf = ydf.apply( lambda x : (x.as_matrix().tostring()), axis = 1)
ydf = ydf.map( lambda x : sqlite3.Binary(x))
ydf = pd.DataFrame(ydf, columns =[ "count_str"])
print(ydf[:5])
print("Saving")
ydf.to_sql(tablename, conn, if_exists="append",
        dtype={"chr":"TEXT", "pos":"INT", "count_str":"BLOB"})


print("DONE")
