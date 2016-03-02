
-- SEQUENCES

-- CREATING A SEQUENCE COLUMN AND VALUES

CREATE TABLE cookbook.insect
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id),
name VARCHAR(30) NOT NULL, # type of insect
date DATE NOT NULL, # date collected
origin VARCHAR(30) NOT NULL # where collected
);

-- There are two ways to generate new AUTO_INCREMENT values in the id column. 
-- One is to explicitly set the id column to NULL.
-- Alternatively, omit the id column from the INSERT statement entirely. MySQL permits creating rows without explicitly specifying values for columns that have a default value.

-- Using the first method:
INSERT INTO 
  cookbook.insect (id,name,date,origin) 
VALUES
  (NULL,'housefly','2014-09-10','kitchen'),
  (NULL,'millipede','2014-09-10','driveway'),
  (NULL,'grasshopper','2014-09-10','front yard'),
  (NULL,'stink bug','2014-09-10','front yard');

-- Using the second method:
INSERT INTO 
  cookbook.insect (name,date,origin) 
VALUES
  ('cabbage butterfly','2014-09-10','garden'),
  ('ant','2014-09-10','back yard'),
  ('ant','2014-09-10','back yard'),
  ('termite','2014-09-10','kitchen woodwork');


SELECT * FROM cookbook.insect ORDER BY id;

-- If you explicitly set an auto-increment column to a non-NULL value, one of two things happen:
-- (a) If the value is already present in the table, an error occurs if the column cannot contain duplicates.
-- (b) If the value is not present in the table, MySQL inserts the row using that value. 
--     In addition, if the value is larger than the current sequence counter, the table’s counter
--     is reset to the value plus one. 


-------------------------------------------------------------------------------
-- CHOOSING DEFINITION FOR A SEQUENCE COLUMN
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id)

-- For an auto-increment column, use the following conventions:
-- (a) The column should be one of the integer types: TINYINT, SMALLINT, MEDIUMINT, INT, or BIGINT

-- (b) UNSIGNED prohibits negative column values and doubles the range of values you can use on the positive side.
-- The following table lists the maximum unsigned value for different INT types:

-- TINYINT    |   255
-- SMALLINT   |   65,535
-- MEDIUMINT  |   16,777,215
-- INT        |   4,294,967,295
-- BIGINT     |   18,446,744,073,709,551,615

-- (c) AUTO_INCREMENT columns cannot contain NULL values, so id is declared as NOT NULL.
-- MySQL automatically defines AUTO_INCREMENT columns as NOT NULL if you forget.

-- (d) AUTO_INCREMENT columns must be indexed. Normally, because a sequence column
-- exists to provide unique identifiers, you use a PRIMARY KEY or UNIQUE index to
-- enforce uniqueness.
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
UNIQUE (id)


---------------------------------------
-- If you are running out of sequence numbers for a column, 
-- check whether you can make the column UNSIGNED or change it to use a larger integer type.
-- To change to UNSIGNED which doubles the space:
ALTER TABLE tbl_name MODIFY id MEDIUMINT UNSIGNED NOT NULL AUTO_INCREMENT;
-- To change INT type:
ALTER TABLE tbl_name MODIFY id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT;


---------------------------------------
-- WHAT HAPPENS TO AUT-INCREMENT AFTER DELETING RECORDS?
DELETE FROM cookbook.insect WHERE id IN (2,8,7);

-- If you remove a row in the middle of the id sequence, nothing changes. The next auto-assigned values will not be affected (MySQL will not fill in the holes).
-- If a record from the top of the id is deleted, it depends on the SQL engine whether the next auto-increment ID will reuse the value that we just deleted, or if it will continue the id after the original high point.
-- For example, InnoDB and MyISM do not reuse values.

-- If a table uses an engine that differs in value-reuse behavior from the behavior you require, 
-- use ALTER TABLE to change the table to a more appropriate engine.
ALTER TABLE tbl_name ENGINE = InnoDB;

-- To see which engine a table uses:
SELECT ENGINE FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'cookbook' AND TABLE_NAME = 'insect';

-- To empty a table and reset the sequence counter (even for engines that normally do not
-- reuse values), use TRUNCATE TABLE:
TRUNCATE TABLE tbl_name;

---------------------------------------
-- To get the ID of the last record you just inserted, you have two options:
-- (a) use the LAST_INSERT_ID() function on the server side.
-- (b) use the function that comes with your APIs have utilities to get this ID, without making a SQL call.

-- Option a
-- When you generate an AUTO_INCREMENT value, LAST_INSERT_ID() returns that specific value, 
-- even when other clients generate new rows in the same table in the meantime.

SELECT * FROM cookbook.insect ORDER BY id;
SELECT LAST_INSERT_ID();

INSERT INTO cookbook.insect (name,date,origin)
VALUES('moth','2014-09-14','windowsill');

SELECT * FROM cookbook.insect WHERE id = LAST_INSERT_ID();


-- Option b:
-- The Connector/Python driver for DB API provides a lastrowid cursor object attribute
-- that returns the most recent AUTO_INCREMENT value:
-- seq = cursor.lastrowid

-- The SQL function LAST_INSERT_ID() operates on the server side,
-- while the API-provided functions operate on the client side.
-- The server-side function is more resilient: as long as you do not issue a new statement that involved an auto-increment field, LAST_INSERT_ID() will return the correct value.
-- However, the client-side functions are not resilient, and are sometimes inconsistent when other statements are issued after adding a recrods with auto-increment field.
-- The client-side function keeps the value of the last inserted ID for the active session only; so if you disconnect and connect again, it will lose the value (will return zero).
-- Also, if you issue other statements (even if they do not involve auto-increment fields), the client-side function will still add to the last inserted ID value.


---------------------------------------
-- RENUMBERING AN EXISTING SEQUENCE
-- Generally, it is not a good idea to requence the group. 
-- you should not resequence a column containing values that are referenced by another table.

-- But, if you still want to add an ID column,
-- the following example shows how to renumber the id values in the insect
-- table using this technique:
-- Note that we used the 'FIRST' keyword to put the column in the beginning of the table.

ALTER TABLE cookbook.insect DROP id;

ALTER TABLE cookbook.insect
ADD id INT UNSIGNED NOT NULL AUTO_INCREMENT FIRST,
ADD PRIMARY KEY (id);


-- In the above code, we had separate processes for deleting the column and adding the new column.
-- This could create some inconsistencies if another session tries to reference the table between the
-- delete row and insert row commands.
-- To prevent this from happening, perform both operations with a single ALTER TABLE statement
ALTER TABLE cookbook.insect
DROP id,
ADD id INT UNSIGNED NOT NULL AUTO_INCREMENT FIRST;

---------------------------------------
-- To reuse the values that have been deleted but avoid resequencing the column
-- Use ALTER TABLE to reset the sequence counter.
-- New sequence numbers will begin with the value one larger than the current maximum in the table.

-- The following command causes MySQL to reset the sequence counter down as far as it can for new rows:
ALTER TABLE cookbook.insect AUTO_INCREMENT = 1;

-- Note that this reuses only values deleted from the top of the sequence. It does not eliminate the gaps.

---------------------------------------
-- ENSURING THAT ROWS ARE RENUMBERED IN A PARTICULAR ORDER
-- If you decide to resequence a column (note that resequencing an existing column is not generally recommended),
-- there is no way to do this in place. You will need to copy the column to a new table, using an ORDER BY constraint to 
-- put the rows in the sequence you would like, and let MySQL add the sequence values.

-- Follow either of the two sequences below:
-- (1) Create an empty clone of the original table.
-- (2) Copy all columns except the auto-increment column from the original table to the clone using the syntax:
-- INSERT INTO <table_name> SELECT ... FROM <original_table> ORDER BY ...
-- (3) Drop the original table and rename the clone table to the original table name.
-- NOTE: If the tavle is a large MyISAM table and has multiple indexes, it is more efficient to create the table initially with no indexes except the one on the auto-increment column.
--       Then copy the original rable into the new table and use ALTER TABLE to add the remaining indexes afterward.

-- Alternative procedure:
-- (1) Create a new table that contains all the columns of the original table except the auto-increment column.
-- (2) Use INSERT INTO <clone_table> SELECT ... FROM <original_table> to copy all columns except the autom-increment column to the cloned table.
-- (3) Use TRUNCATE TABLE <original_table> on the original table to empty int. This also resets the sequence counter to 1.
-- (4) Copy rows from the new table back to the original table, using an ORDER BY clause. MySQL will take care of assigning sequence values to auto-increment column.


---------------------------------------
-- SEQUENCING AN UNSEQUENCED TABLE
-- Add an AUTO_INCREMENT column using ALTER TABLE; MySQL creates the column and numbers its rows.
-- NOTE: remember to add a PRIMARY KEY or a UNIQUE constraint to it. MySQL will index the auto-increment column by default.

ALTER TABLE <table_name>
ADD id INT NOT NULL AUTO_INCREMENT,
ADD PRIMARY KEY (id);

-- By default, ALTER TABLE adds new columns to the end of the table. 
-- To place a column at a specific position, use FIRST or AFTER at the end of the ADD clause.

ALTER TABLE t
ADD id INT NOT NULL AUTO_INCREMENT FIRST,
ADD PRIMARY KEY (id);

ALTER TABLE t
ADD id INT NOT NULL AUTO_INCREMENT AFTER name,
ADD PRIMARY KEY (id);


---------------------------------------
-- MANAGING MULTIPLE AUTO_INCREMENT COLUMNS SIMULTANEOUSLY
-- the LAST_INSERT_ID() server-side sequence value function
-- is set each time a statement generates an AUTO_INCREMENT value, 
-- whereas client-side sequence indicators may be reset for every statement (either involving an auto_increment field or not).
-- Therefore, if you issue a statement involving an auto_increment field, you need to save the last ID either from client side or server side to use it later.

-- To save the value on the server side:
INSERT INTO cookbook.insect (name,date,origin)
VALUES('roach','2014-09-16','kitchen');

-- Save it in a variable
SELECT @saved_id := LAST_INSERT_ID();

-- Access the saved value
SELECT @saved_id;

---------------------------------------
-- USING SEQUENCE GENERATORS AS COUNTERS
-- If you are only interested in counting events, you do not need to create a table row for each sequence value.
-- Instead, use a sequence-generation mechanism that uses a single row per counter.

-- To count a trivial event, use a trivial table with a single column, and one row per type of event that you are counting.

CREATE TABLE cookbook.booksales
(
title VARCHAR(60) NOT NULL, # book title
copies INT UNSIGNED NOT NULL, # number of copies sold
PRIMARY KEY (title)
);

-- To maintain the table, you can first create a line per book, and the update it:
INSERT INTO cookbook.booksales (title,copies) VALUES('The Greater Trumps',0);
UPDATE cookbook.booksales SET copies = copies + 1 WHERE title = 'The Greater Trumps';

-- Or, you can use the INSERT ... ON DUPLICATE UPDATE ... syntax to insert or update depending on the situation.
INSERT INTO cookbook.booksales (title,copies)
VALUES('The Greater Trumps', 1)
ON DUPLICATE KEY UPDATE copies = copies + 1;

-- We can examine the count at any point in time using the following query
SELECT * FROM cookbook.booksales;
-- But there might be another transaction between the time of our transaction and the time we call this function.
-- So it is not quite accurate to associate this count with the last transaction that we performed, unless we lock the table between our transaction and this query.
-- Another way is to call LAST_INSERT_ID with an expression argument:
INSERT INTO cookbook.booksales (title, copies)
VALUES('The Greater Trumps',LAST_INSERT_ID(1))
ON DUPLICATE KEY UPDATE copies = LAST_INSERT_ID(copies+1);

-- MySQL treats the expression argument like an AUTO_INCREMENT value, 
-- so that you can invoke LAST_INSERT_ID() later with no argument to retrieve the value:
SELECT LAST_INSERT_ID();

-- If you are using Python, you do not need to issue the SELECT statement. 
-- Using Connector/Python, update a count and get the new value using the lastrowid attribute:
-- Note that you still need to use the LAST_INSERT_ID(expression) syntax as shown above.
-- cursor = conn.cursor()
-- cursor.execute(SQL)
-- count = cursor.lastrowid
-- cursor.close()
-- conn.commit()

---------------------------------------
-- GENERATING REPEATING SEQUENCES
-- Generate a sequence and use it to produce cyclic elements with division and modulo operations.
-- Just use simple math and modulo functions in addition to the auto-increment field.




















---------------------------------------









