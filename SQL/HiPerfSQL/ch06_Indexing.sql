
-- CREATING A HASH
-- MySQL supports multiple ways to create a hash:

-- To create a hash using crc32:
-- Note: CRC32 is universal; so if you encode a text using CRC32 on different machines, they will return the same result.
SELECT crc32("This is a test.");


-------------------------------------------------------------------------------
CREATE TABLE cookbook.pseudohash (
id int unsigned NOT NULL auto_increment,
url varchar(255) NOT NULL,
url_crc int unsigned NOT NULL DEFAULT 0,
PRIMARY KEY(id)
);


-- Now create a trigger to update the hash value any time the URL value gets updated.
DELIMITER //

CREATE TRIGGER cookbook.pseudohash_crc_ins BEFORE INSERT ON cookbook.pseudohash FOR EACH ROW BEGIN
SET NEW.url_crc=crc32(NEW.url);
END;
//

CREATE TRIGGER cookbook.pseudohash_crc_upd BEFORE UPDATE ON cookbook.pseudohash FOR EACH ROW BEGIN
SET NEW.url_crc=crc32(NEW.url);
END;
//

DELIMITER ;

-- Verify that the trigger works
INSERT INTO cookbook.pseudohash (url) VALUES ('http://www.mysql.com');
SELECT * FROM cookbook.pseudohash;

UPDATE cookbook.pseudohash SET url='http://www.mysql.com/' WHERE id=1;
SELECT * FROM cookbook.pseudohash;

-------------------------------------------------------------------------------

-- (b) USING SHA1() FUNCTION
SELECT SHA1("This is a test.");

-- (c) USING MD5() FUNCTION
SELECT MD5("This is a test.");


-- (c) IMPLEMENT YOUR OWN 64bit HASH FUNCTION
-- One way to do that is to use just part of the result returned by MD5()
-- The goal is not to use a very long value for hash; long values increase the size of the index and negatively impact its performance, 
-- especially if the value is long enough that the indexing function converts it to a string instead of storing it as an integer.
SELECT MD5('http://www.mysql.com/') AS HASH64;
SELECT CONV(RIGHT(MD5('http://www.mysql.com/'), 16), 16, 10) AS HASH64;


-------------------------------------------------------------------------------
-- BENCHMARKING A QUERY
-- To benchmark a query, we need to first understand its statistics.
-- To get the statistics of a query, use the EXPLAIN syntax: put a EXPLAIN clause before the query and examine the results.

EXPLAIN SELECT * FROM cookbook.driver_log WHERE rec_id=1;

-- Particularly, pay attention to columns 'keys' and 'rows'.
-- Column 'keys' of EXPLAIN shows which keys were chosen to find the selected rows. 
-- When optimizing the index, knowing which key the database engine used is useful. We normally want one of the indexes to be used as key.
-- The column 'rows' says how many rows were eventually scanned one-by-one to find the desired result. The lower this number, the better the performance.

EXPLAIN SELECT * FROM cookbook.driver_log WHERE name='Ben';


-------------------------------------------------------------------------------
-- A nice summary of indexing can be found here:
-- http://stackoverflow.com/questions/3567981/how-do-mysql-indexes-work

-- What are the lead nodes?
-- http://use-the-index-luke.com/sql/anatomy/the-leaf-nodes

-- Indexes explained with diagrams:
-- http://blog.jcole.us/2013/01/10/btree-index-structures-in-innodb/

-- How to exploit MySQL index optimizations?
-- http://www.xaprb.com/blog/2006/07/04/how-to-exploit-mysql-index-optimizations/

-------------------------------------------------------------------------------









