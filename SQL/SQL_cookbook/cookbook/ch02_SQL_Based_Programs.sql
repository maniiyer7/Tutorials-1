
-- Chapter summary: how to use MySQL from within the context of a general-purpose programming language.

-- Different programming languages connect to SQL using different APIs and interfaces.
-- For example, Java connects to SQL using JDBC, while
-- Python connects to SQL using DB API.

-- Basic capabilities that any client API should provide include:
-- connecting to MySQL server, selecting a database, and disconnecting
-- checking for errors
-- executing SQL statemens and retrieving results
-- handling special characters and NULL values in statements
-- identifying NULL values in result sets


-------------------------------------------------------------------------------
-- MySQL CLIENT API ARCHITECTURE
--  two-level architecture:
-- (a) The upper level provides database-independent methods that implement database
-- access in a portable way that’s the same whether you use MySQL, PostgreSQL, Oracle, or whatever.
-- (b) The lower level consists of a set of drivers, each of which implements the details for a single database system.

-- The APIs are object-oriented. The operation that connects to the MySQL server returns an object that enables you 
-- to process statements in an object-oriented manner.

-------------------------------------------------------------------------------
-- CONNECTING, SELECTING A DATABASE, DISCONNECTING

-------------------------------------------------------------------------------
-- EXECUTING STATEMENTS AND RETRIEVING RESULTS
DROP TABLE cookbook.profile;

CREATE TABLE cookbook.profile
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
name VARCHAR(20) NOT NULL,
birth DATE,
color ENUM('blue','red','green','brown','black','white'),
foods SET('lutefisk','burrito','curry','eggroll','fadge','pizza'),
cats INT,
PRIMARY KEY (id)
);

-- SQL statements can be grouped into two broad categories, depending on whether they
-- return a result set (a set of rows):
-- (a) Statements that return no result set, such as INSERT, DELETE, or UPDATE. 
-- As a general rule, statements of this type generally change the database in some way. 
-- There are some exceptions, such as USE db_name, which changes the default (current) database
-- for your session without making any changes to the database itself.
-- (b) Statements that return a result set, such as SELECT, SHOW, EXPLAIN, or DESCRIBE. I
-- refer to such statements generically as SELECT statements, but you should understand
-- that category to include any statement that returns rows.


-- When sending the string variable that containts the query,
-- No terminator is necessary because the end of the statement string terminates it.


-------------------------------------------------------------------------------
-- HANDLING SPECIAL VALUES AND NULL VALUES IN STATEMENTS
-- Use your API’s placeholder mechanism or quoting function to make data safe for insertion.
-- This is also useful when you are  constructing statements using data 
-- obtained from external sources and want to prevent SQL injection attacks.

-- There are two methods for dealing with special characters such as quotes
-- and backslashes, and with special values such as NULL:
-- (a) Use placeholders in the statement string to refer to data values symbolically, then
-- bind the data values to the placeholders when you execute the statement.
-- (b) Use a quoting function (if your API provides one) for converting data values to a
-- safe form that is suitable for use in statement strings.

-- Placeholders enable you to avoid writing data values literally in SQL statements.
-- Two common parameter markers are ? and %s.
-- To use a placeholder, rewrite the INSERT statement to use placeholders like this:

INSERT INTO profile (name,birth,color,foods,cats)
VALUES(?,?,?,?,?);

INSERT INTO profile (name,birth,color,foods,cats)
VALUES(%s,%s,%s,%s,%s);

-- Then pass the statement string to the database server and supply the data values separately.
-- The API binds the values to the placeholders to replace them, resulting in a statement that contains the data values.

-- One benefit of placeholders is that parameter-binding operations automatically handle
-- escaping of characters such as quotes and backslashes.
-- A second benefit of placeholders is that you can “prepare” a statement in advance, then
-- reuse it by binding different values to it each time it’s executed

-- You cannot bind an array of data values to a single placeholder. Each value must be bound to a separate placeholder.

-- Using placeholders with Python:
-- The Connector/Python module implements placeholders using %s format specifiers in
-- the SQL statement string. (To place a literal % character into the statement, use %% in the
-- statement string.) To use placeholders, invoke the execute() method with two arguments:
-- a statement string containing format specifiers and a sequence containing the
-- values to bind to the statement string. Use None to bind a NULL value to a placeholder.
-- The Connector/Python placeholder mechanism provides quotes around data values as
-- necessary when they are bound to the statement string, so don’t put quotes around the
-- %s format specifiers in the string.
-- The values passed to the placeholders should be in a tuple. 
-- So, if you have only a single value val to bind to a placeholder, write it as a sequence using the syntax (val,).


-------------------------------------------------------------------------------
-- HANDLING SPECIAL CHARACTERS IN IDENTIFIERS
-- To make an identifier safe for insertion into an SQL statement, quote it by enclosing it within backticks.
CREATE TABLE cookbook.`some table` (i INT);
-- In MySQL, backticks are always permitted for identifier quoting. The double-quote
-- character is permitted as well, if the ANSI_QUOTES SQL mode is enabled.

-- To check which identifier character quoting characters are permitted:
SELECT @@sql_mode;
-- If the result includes ANSI_QUOTES, MySQL interprets 'abc' as a string and "abc" as an identifier. 
-- In such case, be careful to use single quotes for strings and double-quotes for identifiers.

SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'cookbook' AND TABLE_NAME = 'profile';


-------------------------------------------------------------------------------
-- IDENTIFYING NULL VALUES IN RESULTS SETS
-- First, fabricate some null values in the table to use in the examples below.
INSERT INTO cookbook.profile (name) VALUES('Amabel');
SELECT * FROM cookbook.profile WHERE name = 'Amabel';


-------------------------------------------------------------------------------
-- TECHNIQUES FOR OBTAINING CONNECTION PARAMETERS
-- Connection parameters such as user name, password, etc.
-- See the chapter for more details.

















