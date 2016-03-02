
-- Information about statement results
-- Information about databases and tables
-- Information about the MySQL server

-- A more portable source of metadata is INFORMATION_SCHEMA, a database that contains
-- information about databases, tables, columns, character sets, and so forth.

-------------------------------------------------------------------------------
-- DETERMINING THE NUMBER OF ROWS AFFECTED BY A STATEMENT
-- For statements that affect rows (UPDATE, DELETE, INSERT, REPLACE), each API provides
-- a way to determine the number of rows involved.

-- Python’s DB API makes the rows-changed count available as the value of the statement
-- cursor’s rowcount attribute:

-- cursor = conn.cursor()
-- cursor.execute(stmt)
-- print("Number of rows affected: %d" % cursor.rowcount)
-- cursor.close()

-------------------------------------------------------------------------------
-- OBTAINING RESULT SET META DATA
-- to know things about the result set, such as the column names and data types, or how many rows and columns there are.
-- See my tutorial codes in Python tutorials folder.

-- To get the row count for a result set, access the cursor’s rowcount attribute.

-- To determine the column count without fetching any rows by using cursor.description.
-- This is a tuple containing one element per column in the result set, 
-- so its length tells you how many columns are in the set.

-- To determine whether a SQL statement returned a result set or not, you can check the column count.
-- Check the column count in the metadata. There is no result set if the count is zero.
-- A column count of zero indicates that the statement was an INSERT, UPDATE, or some other statement that returns no result set.

-- In Python, the value of cursor.description is None for statements that produce no result set.


-------------------------------------------------------------------------------
-- Use INFORMATION_SCHEMA to get this information. The SCHEMATA table contains a row
-- for each database, and the TABLES table contains a row for each table in each database

SELECT SCHEMA_NAME 
FROM INFORMATION_SCHEMA.SCHEMATA;


---------------------------------------
-- Get table names for a database
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'cookbook';

-- Get table names for the default database
-- If no database has been selected, DATABASE() returns NULL and no rows match.
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = DATABASE();


-------------------------------------------------------------------------------
-- ACCESSING TABLE COLUMN DEFINITIONS
-- MySQL provides several ways to find out about a table’s structure:
--  (a) Retrieve the information from INFORMATION_SCHEMA. The COLUMNS table contains the column definitions.
--  (b) Use a SHOW COLUMNS statement.
--  (c) Use the SHOW CREATE TABLE statement or the mysqldump command-line program to obtain a CREATE TABLE statement that displays the table’s structure.

-- Example: 
CREATE TABLE cookbook.item
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
name CHAR(20),
colors ENUM('chartreuse','mauve','lime green','puce') DEFAULT 'puce',
PRIMARY KEY (id)
);


-- To obtain information about a single column in a table
SELECT * 
FROM 
  INFORMATION_SCHEMA.COLUMNS
WHERE 
  TABLE_SCHEMA = 'cookbook' 
  AND TABLE_NAME = 'item'
  AND COLUMN_NAME = 'colors';


-- Here are some COLUMNS table columns likely to be of most use:
-- (a) COLUMN_NAME: The column name.
-- (b) ORDINAL_POSITION: The position of the column within the table definition.
-- (c) COLUMN_DEFAULT: The column’s default value.
-- (d) IS_NULLABLE: YES or NO to indicate whether the column can contain NULL values.
-- (e) DATA_TYPE, COLUMN_TYPE: Data type information. DATA_TYPE is the data-type keyword
-- and COLUMN_TYPE contains additional information such as type attributes.
-- (f) CHARACTER_SET_NAME, COLLATION_NAME: The character set and collation for string
-- columns. They are NULL for nonstring columns.
-- (g) COLUMN_KEY: Information about whether the column is indexed.


---------------------------------------
-- Another way to obtain table structure information from MySQL is from the CREATE
-- TABLE statement that defines the table
SHOW CREATE TABLE item;
*************************** 1. row ***************************
Table: item
Create Table: CREATE TABLE `item` (
`id` int(10) unsigned NOT NULL AUTO_INCREMENT,
`name` char(20) DEFAULT NULL,
`colors` enum('chartreuse','mauve','lime green','puce') DEFAULT 'puce',
PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1

SHOW CREATE TABLE;


-------------------------------------------------------------------------------
-- GETTING ENUM AND SET COLUMN INFORMATION
-- Obtain the column definition from the table metadata, then extract the member list from the definition.

SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'cookbook' AND TABLE_NAME = 'item' AND COLUMN_NAME = 'colors';


-------------------------------------------------------------------------------
-- GETTING SERVER METADATA
-- Both SHOW statements permit a GLOBAL or SESSION
-- keyword to select global server values or values specific to your session, and a LIKE
-- 'pattern' clause for limiting the results to variable names matching the pattern:

SELECT VERSION();
SELECT DATABASE();
SELECT USER();
SELECT CURRENT_USER();

SHOW SESSION STATUS;
SHOW SESSION ariables;

-------------------------------------------------------------------------------
-- WRITING APPLICATIONS THAT ADAPT TO THE MYSQL PATENT





