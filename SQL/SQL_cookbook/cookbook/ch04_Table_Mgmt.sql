
-- There are three ways to define a new table:
-- (a) by explicit column definitions
-- (b) by cloning an existing table
-- (c) by creating the table on the fly using a query result set

-------------------------------------------------------------------------------
-- CLONING A TABLE
-- Use CREATE TABLE … LIKE to clone the table structure.
-- To also copy some or all of the rows from the original table to the new one, use INSERT INTO … SELECT.

-- The structure of the new table is the same as that of the original table, with a few exceptions:
-- CREATE TABLE … LIKE does not copy foreign key definitions, and it doesn’t
-- copy any DATA DIRECTORY or INDEX DIRECTORY table options that the table might use.

-- The new table is empty. If you also want the contents to be the same as the original table,
-- copy the rows using an INSERT INTO … SELECT statement:

CREATE TABLE new_table LIKE original_table;
INSERT INTO new_table SELECT * FROM original_table;


CREATE TABLE cookbook.mail2 LIKE cookbook.mail;
INSERT INTO cookbook.mail2 SELECT * FROM cookbook.mail WHERE srcuser = 'barb';


-------------------------------------------------------------------------------
-- SAVE A QUERY RESULT IN A TABLE
-- If the table exists, retrieve rows into it using INSERT INTO … SELECT.
-- If the table does not exist, create it on the fly using CREATE TABLE … SELECT.


-- The number of columns to be inserted must match the number of selected columns,
-- with the correspondence between columns based on position rather than name
INSERT INTO dst_tbl (i, s) SELECT val, name FROM src_tbl;


-- The select statement can be any query, as long as the query results match the selected columns in INSERT statement.
INSERT INTO dst_tbl (i, s) 
SELECT COUNT(*), name
FROM src_tbl 
GROUP BY name;


-- If the destination table does not exist:
-- use CREATE TABLE … SELECT to create the destination table directly from the result of the SELECT
CREATE TABLE dst_tbl SELECT * FROM src_tbl;

-- We can mix & match new columns and columns from existing tables to create a new table:
-- The following statement creates id as an AUTO_INCREMENT column in dst_tbl and adds
-- columns a, b, and c from src_tbl:
CREATE TABLE dst_tbl
(
id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (id)
)
SELECT a, b, c FROM src_tbl;


-- Note that in this syntax, some meta-data from the original table is lost.
-- To make the destination table an exact copy of the source table, use the cloning technique.
-- To include indexes in the destination table, you can also specify them explicitly. For example, if
-- src_tbl has a PRIMARY KEY on the id column, and a multiple-column index on
-- state and city, specify them for dst_tbl as well:
CREATE TABLE dst_tbl (PRIMARY KEY (id), INDEX(state,city))
SELECT * FROM src_tbl;

-- Column attributes such as AUTO_INCREMENT and a column’s default value are not
-- copied to the destination table. To preserve these attributes, create the table, then
-- use ALTER TABLE to apply the appropriate modifications to the column definition.

CREATE TABLE dst_tbl (PRIMARY KEY (id)) SELECT * FROM src_tbl;
ALTER TABLE dst_tbl MODIFY id INT UNSIGNED NOT NULL AUTO_INCREMENT;


-------------------------------------------------------------------------------
-- CREATING TEMPORARY TABLES
-- When you create a table using TEMPORARY keyword, MySQL will automatically remove it when the session is over.

-- Similar to table creation, there are three main ways to define a new temporary table:
-- (a) create the table by defining its column specs
CREATE TEMPORARY TABLE tbl_name (...column definitions...);
-- (b) create the table by cloning an existing table
CREATE TEMPORARY TABLE new_table LIKE original_table;
-- (c) create the table by a query result set
CREATE TEMPORARY TABLE tbl_name SELECT ... ;


-- A temporary table can have the same name as a permanent table. In this case, the temporary
-- table “hides” the permanent table for the duration of its existence.
-- This can be useful for making a copy of a table that you can modify without affecting the original by mistake.


-------------------------------------------------------------------------------
-- GENERATING UNIQUE TABLE NAME

-- If you create a TEMPORARY table, it doesn’t matter whether a permanent table with that name exists.
-- If you cannot or do not want to use a TEMPORARY table, incorporate into the name some value guaranteed to be unique per invocation.
-- To achieve this, incorporate into the table name the process ID PID of the session. 
-- PID is guaranteed to be unique among the set of currently executing processes.
-- PID can be grabbed from within the client application, such as Python. The os module in Python has a function getpid() for this purpose.

-- Another way to get a unique identifier for a session is to use a connection ID. We can get the connection ID from within MySQL:
SELECT CONNECTION_ID();

SELECT @tbl_name := CONCAT('tmp_tbl_', CONNECTION_ID());

SELECT @stmt := CONCAT('DROP TABLE IF EXISTS ', @tbl_name);
PREPARE stmt FROM @stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SELECT @stmt := CONCAT('CREATE TABLE ', @tbl_name, ' (i INT)');
PREPARE stmt FROM @stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


-------------------------------------------------------------------------------
-- CHECKING OR CHANGING A TABLE STORAGE ENGINE
-- To determine a table’s storage engine, you can use any of several statements. 
-- For example, check INFORMATION_SCHEMA or use the SHOW TABLE STATUS or SHOW CREATE TABLE statement
-- To change the table’s engine, use ALTER TABLE with an ENGINE clause.

-- Different table engines have different capabilities. 
-- For example, the InnoDB engine supports transactions, whereas MyISAM does not.

SELECT ENGINE FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'cookbook' AND TABLE_NAME = 'mail';

SHOW TABLE STATUS LIKE 'mail';

SHOW CREATE TABLE cookbook.mail;

-- To change the table engine
ALTER TABLE cookbook.mail ENGINE = MyISAM;


-------------------------------------------------------------------------------
-- COPYING A TABLE USING mysqldump
-- To copy a table or tables, either among the databases managed by a MySQL server, or from one server to another,
-- use the mysqldump program.

-- The mysqldump program makes a backup file that can be reloaded to re-create the original table or tables:
-- From the Terminal, run:
-- $ mysqldump cookbook mail > mail.sql

-- The output file mail.sql consists of a CREATE TABLE statement to create the mail table
-- and a set of INSERT statements to insert its rows.

-- To recreate the table using the dump file, run the following command in Terminal:
-- $ mysql cookbook < mail.sql

-- mysqldump can also be used to copy tables within the same mysql server and across different servers.

---------------------------------------
-- Copy a table within the same MySQL server, to a different database:
-- $ mysqldump cookbook mail > mail.sql
-- $ mysql other_db < mail.sql

-- Copy all tables in a database to a different database, do not specify a table name after the database name,
-- When you name no tables after the database name, mysqldump dumps them all.
-- $ mysqldump cookbook > cookbook.sql
-- $ mysql other_db < cookbook.sql


-- To perform a table-copying operation without an intermediary file, use a pipe to connect
-- the mysqldump and mysql commands:
-- $ mysqldump cookbook mail | mysql other_db
-- $ mysqldump cookbook | mysql other_db


-- To copy the mail table from the cookbook database on the local host 
-- to the other_db database on the host otherhost.example.com
-- One way to do this is to dump the output into a file:
-- $ mysqldump cookbook mail > mail.sql
-- Then copy mail.sql to other-host.example.com, and run the following command there
-- to load the table into that MySQL server’s other_db database:
-- $ mysql other_db < mail.sql

-- To accomplish this without an intermediary file, use a pipe to send the output of mysqldump
-- directly over the network to the remote MySQL server.
-- Note that this requires that you are able to connect to both servers from your local host.
-- $ mysqldump cookbook mail | mysql -h other-host.example.com other_db


-- If you cannot connect directly to the remote server using mysql from your local host,
-- send the dump output into a pipe that uses ssh to invoke mysql remotely on otherhost.example.com
-- $ mysqldump cookbook mail | ssh other-host.example.com mysql other_db





