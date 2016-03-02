
-- PREVENTUNG DUPLICATES FROM OCCURING

-- Use a PRIMARY KEY or UNIQUE index.
-- A primary key field does not accept NULL values.

CREATE TABLE cookbook.person
(
last_name CHAR(20) NOT NULL,
first_name CHAR(20) NOT NULL,
address CHAR(40),
PRIMARY KEY (last_name, first_name)
);


-- Another way to enforce uniqueness is to add a UNIQUE index rather than a PRIMARY KEY to a table.
-- UNIQUE index is similar to PRIMARY KEY, but a UNIQUE index can be created on
-- columns that permit NULL values. You can separately add a NOT NULL constraint to the field.
-- If a UNIQUE index does happen to permit NULL values, NULL is the one value that can occur multiple times. 
-- This is because it is impossible to know what the original value of the NULL record is.


DROP TABLE cookbook.person;

CREATE TABLE cookbook.person
(
last_name CHAR(20) NOT NULL,
first_name CHAR(20) NOT NULL,
address CHAR(40),
UNIQUE (last_name, first_name)
);

-- If you want to allow duplicate names, but still want to distinguish different people from another, 
-- each person must be assigned some sort of unique identifier, which becomes the value that distinguishes one row from another.
DROP TABLE cookbook.person;

CREATE TABLE cookbook.person
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
last_name CHAR(20),
first_name CHAR(20),
address CHAR(40),
PRIMARY KEY (id)
);


-------------------------------------------------------------------------------
-- DEALING WITH DUPLICATES WHEN LOADING ROWS INTO A TABLE
-- One approach is to just ignore the error. 
-- Another is to use an INSERT IGNORE, REPLACE,
-- or INSERT … ON DUPLICATE KEY UPDATE statement, 
-- each of which modifies MySQL’s duplicate-handling behavior.


---------------------------------------
-- MySQL provides three single-query solutions to the problem of handling duplicate rows.
-- (a) INSERT IGNORE: To keep the original row when a duplicate occurs, use INSERT IGNORE.
-- (b) REPLACE: To replace the original row with the new one when a duplicate occurs, use REPLACE rather than INSERT. 
--     If the row is new, it’s inserted just as with INSERT. If it’s a duplicate, the new row replaces the old one:
-- (c) INSERT … ON DUPLICATE KEY UPDATE: To modify columns of an existing row when a duplicate occurs, use INSERT … ON DUPLICATE KEY UPDATE. 
--     If the row is new, it’s inserted. If it’s a duplicate, the ON DUPLICATE KEY UPDATE clause indicates how to modify the existing row in the table.
-- INSERT IGNORE is more efficient than REPLACE because it doesn’t actually insert duplicates. 
-- When you know the values of all the fields in a new records, use REPLACE. 
-- When you dont know the values of all fields, but only know the unique index values and the values that need to be updated, use INSERT ON DUPLICATE KEY UPDATE.

CREATE TABLE cookbook.poll_vote
(
poll_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
candidate_id INT UNSIGNED,
vote_count INT UNSIGNED,
PRIMARY KEY (poll_id, candidate_id)
);

-- Here, for all votes except the first, you don’t know what the vote count should be.
-- Therefore, we cannot use REPLACE or INSERT IGNORE (because both of these methods need specific values for all fields).
-- We can use INSERT ... ON DUPLICATE KEY UPDATE, since we can increment the value of the vote (whatever it might be) by 1.
SELECT * FROM poll_vote;

INSERT INTO poll_vote (poll_id,candidate_id,vote_count) 
VALUES(14,3,1) 
ON DUPLICATE KEY UPDATE vote_count = vote_count + 1;

SELECT * FROM poll_vote;

INSERT INTO poll_vote (poll_id,candidate_id,vote_count) 
VALUES(14,3,1)
ON DUPLICATE KEY UPDATE vote_count = vote_count + 1;

SELECT * FROM poll_vote;


-------------------------------------------------------------------------------
-- FINDING DUPLICATES IN A TABLE
SELECT COUNT(*) AS rows FROM cookbook.catalog_list;

SELECT 
  COUNT(DISTINCT last_name, first_name) AS 'distinct names',
  COUNT(DISTINCT last_name, first_name) / COUNT(*) AS 'unique_ratio',
  COUNT(*) - COUNT(DISTINCT last_name, first_name) AS 'duplicate names',
  1 - (COUNT(DISTINCT last_name, first_name) / COUNT(*)) AS 'duplicate_ratio'
FROM 
  cookbook.catalog_list;


-- To show which values are duplicated:
-- In general, to identify sets of values that are duplicated, do the following:
-- 1. Determine which columns contain the values that may be duplicated.
-- 2. List those columns in the column selection list, along with COUNT(*).
-- 3. List the columns in the GROUP BY clause as well.
-- 4. Add a HAVING clause that eliminates unique values by requiring group counts to be greater than one.

SELECT 
  COUNT(*), last_name, first_name
FROM 
  cookbook.catalog_list
GROUP BY 
  last_name, first_name
HAVING 
  COUNT(*) > 1;


-- To see the original rows containing the duplicate names, 
-- join the summary information to the table from which it’s generated.
-- To join the summary table to the original table, either use a temporary table, or a sub-query.
CREATE TABLE tmp
SELECT 
  COUNT(*) AS count, last_name, first_name 
FROM 
  cookbook.catalog_list
GROUP BY 
  last_name, first_name 
HAVING 
  count > 1;


SELECT 
  catalog_list.*
FROM 
  cookbook.tmp 
  INNER JOIN catalog_list 
  USING (last_name, first_name)
ORDER BY 
  last_name, first_name;


-- Using sub-query:
SELECT 
  catalog_list.*
FROM 
  (SELECT 
  COUNT(*) AS count, last_name, first_name 
FROM 
  cookbook.catalog_list
GROUP BY 
  last_name, first_name 
HAVING 
  count > 1)  counts 
  INNER JOIN catalog_list 
  USING (last_name, first_name)
ORDER BY 
  last_name, first_name;


-------------------------------------------------------------------------------
-- DELETE DUPLICATE RECORDS
-- Two methods:
-- (a) Select the unique rows from the table into a second table, then use it to replace the original one. 
-- (b) Use DELETE … LIMIT n to remove all but one instance of a specific set of duplicate rows.

---------------------------------------
-- Method (a): remove duplicate rows by copying the unique rows into a new table, then replacing the original table with the unique rows from the temporary table.
-- If a row is considered to duplicate another only if the entire row is the same, this method is preferable.

DROP TABLE cookbook.tmp;

CREATE TABLE tmp LIKE catalog_list;

INSERT INTO cookbook.tmp SELECT DISTINCT * FROM catalog_list;

DROP TABLE catalog_list;
RENAME TABLE tmp TO catalog_list;

SELECT * FROM cookbook.catalog_list;

-- For tables that contain duplicate NULL values, this method removes those duplicates. 
-- It does not prevent the occurrence of duplicates in the future


---------------------------------------
-- Method (b) 
-- If duplicates are defined only with respect to a subset of the columns in the table, 
-- create a new table that has a unique index for those columns, 
-- select rows into it using INSERT IGNORE, 
-- and replace the original table with the new one.
DROP TABLE cookbook.tmp;

-- Create a temporary table to hold unique values from the original table
CREATE TABLE cookbook.tmp LIKE catalog_list;

-- unique index prevents rows with duplicate key values from being inserted into tmp
ALTER TABLE cookbook.tmp ADD PRIMARY KEY (last_name, first_name);

-- IGNORE tells MySQL not to stop with an error if a duplicate is found.
INSERT IGNORE INTO cookbook.tmp SELECT * FROM catalog_list;

-- verify that the temporary table has the unique values
SELECT * FROM cookbook.tmp ORDER BY last_name, first_name;

-- replace the original table with the new (temporary) table
DROP TABLE cookbook.catalog_list;
RENAME TABLE cookbook.tmp TO catalog_list;

-- verify the results
SELECT * FROM cookbook.catalog_list ORDER BY last_name, first_name;


---------------------------------------
-- You can use LIMIT to DELETE only a subset of the rows.





