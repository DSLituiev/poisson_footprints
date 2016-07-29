
ATTACH DATABASE "bin_seq.db" AS bs;

--CREATE TABLE  [batf_seq_dna_atac](chr TEXT,
--  pos INT,
--  bin_seq BLOB,
--  count_str BLOB,
--  PRIMARY KEY(chr, pos) );
CREATE TABLE  [batf_seq_dna_atac] AS
SELECT aa.chr AS chr, aa.pos AS pos, aa.bin_seq AS bin_seq, bb.count_str AS count_str
FROM bs.batf_bin_seq AS aa INNER JOIN batf_disc1_gw_atac_pivot AS bb ON aa.chr=bb.chr AND aa.pos=bb.pos;

CREATE UNIQUE INDEX idx_batf_seq_dna_atac ON batf_seq_dna_atac(chr, pos);
