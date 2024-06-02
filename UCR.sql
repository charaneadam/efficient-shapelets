\c shapelets
DROP TABLE IF EXISTS metadata;
CREATE TABLE metadata (id INT, type VARCHAR(25), name VARCHAR(100), train INT, test INT, class INT, length INT);
copy metadata from '/tmp/clean.csv' delimiter ' ' CSV HEADER;
