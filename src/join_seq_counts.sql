
ATTACH DATABASE "bin_seq.db" AS bs;

CREATE TABLE  [IGTB1077_seq] AS
SELECT aa.chr AS chr, aa.pos AS pos, aa.bin_seq AS bin_seq, bb.count_str AS count_str
FROM bs.batf_bin_seq AS aa INNER JOIN IGTB1077 AS bb ON aa.chr=bb.chr AND aa.pos=bb.pos;
