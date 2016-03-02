
-- There are several different types of stored database programs:
-- (a) stored functions and procedures (collectively called stored routines)
--     A stored function or procedure object encapsulates the code for performing an
--     operation, enabling you to invoke the object easily by name rather than repeat all
--     its code each time it’s needed.
--     Functions return values and can be used in expressions.
--     Procedures return no value. They can be used to update some values in a table or to produce a results set that is sent to a client program.

-- (b) triggers
--     A trigger is an object that activates when a table is modified by an INSERT, UPDATE, or DELETE statement.
--     Triggers can be used to check values before they are inserted into a table, 
--     or specify that any row deleted from a table should be logged to another table that serves as a journal of data changes.

-- (c) scheduled events
--     Scheduled events are similar to cron jobs in Unix. They go off at a certain time/frequency.

-- Stored programs are database objects that are user-defined but stored on the server side for later execution

CREATE PROCEDURE cookbook.show_version()
SELECT VERSION() AS 'MySQL Version';


CREATE PROCEDURE cookbook.show_part_of_day()
BEGIN
DECLARE cur_time, day_part TEXT;
SET cur_time = CURTIME();
IF cur_time < '12:00:00' THEN
SET day_part = 'morning';
ELSEIF cur_time = '12:00:00' THEN
SET day_part = 'noon';
ELSE
SET day_part = 'afternoon or night';
END IF;
SELECT cur_time, day_part;
END;


CALL cookbook.show_version();
CALL cookbook.show_part_of_day();


-- Previlede requirements for running stored routines:
-- (a) To create or execute the routine, you must have the CREATE ROUTINE or EXECUTE privilege, respectively.
-- (b) If binary logging is enabled for your MySQL server, as is common practice, there
--     are additional requirements for creating stored functions (but not stored procedures).
--     (i) You must have the SUPER privilege, and you must declare either that the function
--     is deterministic or does not modify data by using one of the DETERMINISTIC, NO SQL, or READS SQL DATA characteristics.
--     (ii) Alternatively, if you enable the log_bin_trust_function_creators system
--     variable, the server waives both of the preceding requirements.
-- (c) To create a trigger, you must have the TRIGGER privilege for the table associated with the trigger.
-- (d) To create a scheduled event, you must have the EVENT privilege for the database in which the event is created.


-------------------------------------------------------------------------------
-- CREATING COMPOUND STATEMENT OBJECTS

-- Stored routine is composed of several statements, enclosed in a BEGING ... END block.
-- Each statement within a compound statement must be terminated by a ; character.
-- That requirement causes a problem if you use the mysql client to define an object that uses
-- compound statements because mysql itself interprets ; to determine statement boundaries.
-- The solution is to redefine mysql’s statement delimiter while you define a
-- compound-statement object.

delimiter $$

CREATE FUNCTION cookbook.avg_mail_size(user VARCHAR(8))
RETURNS FLOAT READS SQL DATA
BEGIN
DECLARE avg FLOAT;
IF user IS NULL
THEN # average message size over all users
SET avg = (SELECT AVG(size) FROM cookbook.mail);
ELSE # average message size for given user
SET avg = (SELECT AVG(size) FROM cookbook.mail WHERE srcuser = user);
END IF;
RETURN avg;
END;
$$

delimiter ;

SELECT avg_mail_size(NULL), avg_mail_size('barb');


-------------------------------------------------------------------------------
-- USING STORED FUNCTIONS TO ENCAPSULATE CALCULATIONS

-- The function below looks up and return the sales tax for a given US state from the table sales_tax_rates.
-- The function handles states not listed using a CONTINUE handler for NOT FOUND, which sets the value to zero.

-- Note how we assigned value to variables: SELECT ... INTO

CREATE FUNCTION cookbook.sales_tax_rate(state_code CHAR(2))
RETURNS DECIMAL(3,2) READS SQL DATA
BEGIN
DECLARE rate DECIMAL(3,2);
DECLARE CONTINUE HANDLER FOR NOT FOUND SET rate = 0;
SELECT tax_rate INTO rate FROM cookbook.sales_tax_rate WHERE state = state_code;
RETURN rate;
END;


-- We run a function by SELECT statement.
-- We run a procedure by CALL statement.
SELECT cookbook.sales_tax_rate('CA');
SELECT cookbook.sales_tax_rate('VT'), cookbook.sales_tax_rate('NY');


-------------------------------------------------------------------------------
-- USING STORED PROCEDURES TO RETURN MULTIPLE VALUES

-- An operation produces two or more values, but a stored function can return only a single value.

-- Use a stored procedure that has OUT or INOUT parameters, 
-- and pass user-defined variables for those parameters when you invoke the procedure

-- A procedure does not “return” a value the way a function does, 
-- but it can assign values to those parameters so that the
-- user-defined variables have the desired values when the procedure returns.

-- a stored procedure parameter can be any of three types:
-- (a) An IN parameter is for input only. This is the default if you specify no type.
-- (b) An INOUT parameter is used to pass a value in, and can also pass a value out.
-- (c) An OUT parameter is used to pass a value out.

CREATE PROCEDURE cookbook.mail_sender_stats(IN user VARCHAR(8),
                                            OUT messages INT,
                                            OUT total_size INT,
                                            OUT avg_size INT)
BEGIN
-- Use IFNULL() to return 0 for SUM() and AVG() in case there are
-- no rows for the user (those functions return NULL in that case).
SELECT COUNT(*), IFNULL(SUM(size),0), IFNULL(AVG(size),0)
INTO messages, total_size, avg_size
FROM cookbook.mail 
WHERE srcuser = user;
END;


CALL mail_sender_stats('barb',@messages,@total_size,@avg_size);
SELECT @messages, @total_size, @avg_size;


-------------------------------------------------------------------------------
-- A table contains a column for which the initial value is not constant
-- a BEFORE INSERT trigger would let you perform dynamic column initialization by calculating the default value.
-- Nore: An AFTER INSERT trigger can examine column values for a new row, 
-- but by the time the trigger activates, it’s too late to change the values.

CREATE TABLE cookbook.cust_invoice
(
id INT NOT NULL AUTO_INCREMENT,
state CHAR(2), 
amount DECIMAL(10,2), 
tax_rate DECIMAL(3,2), 
PRIMARY KEY (id)
);

-- To initialize the sales tax_rate column, use a trigger
CREATE TRIGGER cookbook.bi_cust_invoice BEFORE INSERT ON cust_invoice
FOR EACH ROW SET NEW.tax_rate = sales_tax_rate(NEW.state);

-- Verify that the trigger works
INSERT INTO cookbook.cust_invoice (state,amount) VALUES('NY',100);

SELECT * FROM cookbook.cust_invoice WHERE id = LAST_INSERT_ID();


-------------------------------------------------------------------------------
-- USING TRIGGERS TO SIMULATE FUNCTION-BASED INDEX
-- If you have an index on a column, and want to use that column in WHERE statement,
-- the index would help improve the performance. However, if you want to use a function
-- in your WHERE statement, such as the query below, MySQL can no longer the index.
-- The result in diminished performance of the query.
-- SELECT * FROM expdata WHERE LOG10(value) < 2;

-- To work around this problem, and use a function of a column as a criteria to use records,
-- use a secondary column and triggers to simulate a function-based index. More specifically:
-- 1. Define a secondary column to store the function values and index that column.
-- 2. Define triggers that keep the secondary column up to date when the original column is initialized or modified.
-- 3. Refer directly to the secondary column in queries so that the optimizer can use the index on it for efficient lookups.

CREATE TABLE cookbook.expdata
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
value FLOAT,
log10_value FLOAT, 
INDEX (value), 
INDEX (log10_value) 
);

-- Create an INSERT trigger to initialize the log10_value value from value for new rows,
-- and an UPDATE trigger to keep log10_value up to date when value changes.
CREATE TRIGGER cookbook.bi_expdata 
BEFORE INSERT ON expdata
FOR EACH ROW SET NEW.log10_value = LOG10(NEW.value);

CREATE TRIGGER cookbook.bu_expdata 
BEFORE UPDATE ON expdata
FOR EACH ROW SET NEW.log10_value = LOG10(NEW.value);

-- test the implementation
INSERT INTO cookbook.expdata (value) VALUES (.01),(.1),(1),(10),(100);
SELECT * FROM cookbook.expdata;

UPDATE cookbook.expdata SET value = value * 10;
SELECT * FROM cookbook.expdata;

-- Now that we have a secondary column that stores LOG values of the column of interest, and
-- we have an insert and an update trigger to automatically update the secondary column,
-- the SELECT query shown earlier can be rewritten:
-- SELECT * FROM expdata WHERE log10_value < 2;


-------------------------------------------------------------------------------
-- SIMULATING TIMESTAMP PROPERTIES FOR OTHER DATA AND TIME TYPES
-- Most importantly, auto-initialization and auto-update properties
-- Based on desired functionality, use an INSERT and/or UPDATE trigger.

CREATE TABLE cookbook.ts_emulate (data CHAR(10), d DATE, t TIME);

-- Now create BEFORE INSERT triggers for inserting new records.
CREATE TRIGGER cookbook.bi_ts_emulate 
BEFORE INSERT ON ts_emulate
FOR EACH ROW SET NEW.d = CURDATE(), NEW.t = CURTIME();

-- For BEFORE UPDATE trigger, an IF statement is required here to emulate the TIMESTAMP
-- property that an update occurs only if the data value in the row actually changes from its current value:
CREATE TRIGGER cookbook.bu_ts_emulate 
BEFORE UPDATE ON ts_emulate
FOR EACH ROW 
IF NEW.data <> OLD.data THEN
SET NEW.d = CURDATE(), NEW.t = CURTIME();
END IF;

-- test the implementation:
INSERT INTO cookbook.ts_emulate (data) VALUES('cat');
INSERT INTO cookbook.ts_emulate (data) VALUES('dog');
SELECT * FROM cookbook.ts_emulate;

UPDATE cookbook.ts_emulate SET data = 'axolotl' WHERE data = 'cat';
SELECT * FROM cookbook.ts_emulate;

UPDATE cookbook.ts_emulate SET data = data;
SELECT * FROM cookbook.ts_emulate;


-------------------------------------------------------------------------------
-- USING TRIGGERS TO LOG CHANGES TO A TABLE
-- We can write triggers to “catch” table changes and write them to a separate log table.

-- Supposed in the following table, we want to keep a record of the different bids as they change.
CREATE TABLE cookbook.auction
(
id INT UNSIGNED NOT NULL AUTO_INCREMENT,
ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
item VARCHAR(30) NOT NULL,
bid DECIMAL(10,2) NOT NULL,
PRIMARY KEY (id)
);

-- To maintain a journal that shows all changes to auctions as they progress from creation
-- to removal, set up another table that serves to record a history of changes to the auctions
CREATE TABLE cookbook.auction_log
(
action ENUM('create','update','delete'),
id INT UNSIGNED NOT NULL,
ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
item VARCHAR(30) NOT NULL,
bid DECIMAL(10,2) NOT NULL,
INDEX (id)
);

-- To log all different types of changes to the auction table, we create 
-- three different triggers, each logging one of the following changes: insert, update, delete.
-- For this application, AFTER triggers are used because they activate only after successful
-- changes to the auction table

CREATE TRIGGER cookbook.ai_auction 
AFTER INSERT ON auction
FOR EACH ROW
INSERT INTO auction_log (action,id,ts,item,bid)
VALUES('create',NEW.id,NOW(),NEW.item,NEW.bid);

CREATE TRIGGER cookbook.au_auction 
AFTER UPDATE ON auction
FOR EACH ROW
INSERT INTO auction_log (action,id,ts,item,bid)
VALUES('update',NEW.id,NOW(),NEW.item,NEW.bid);

CREATE TRIGGER cookbook.ad_auction 
AFTER DELETE ON auction
FOR EACH ROW
INSERT INTO auction_log (action,id,ts,item,bid)
VALUES('delete',OLD.id,OLD.ts,OLD.item,OLD.bid);

-- The INSERT and UPDATE triggers use NEW.col_name to access the new values being stored in rows. 
-- The DELETE trigger uses OLD.col_name to access the existing values from the deleted row.

-- The INSERT and UPDATE triggers use NEW.col_name to access the new values being stored
-- in rows. The DELETE trigger uses OLD.col_name to access the existing values from the
-- deleted row. 
-- The INSERT and UPDATE triggers use NOW() to get the row-modification
-- times; the ts column is initialized automatically to the current date and time, but NEW.ts
-- will not contain that value.

-- test the implementation
INSERT INTO cookbook.auction (item,bid) VALUES('chintz pillows',5.00);
SELECT LAST_INSERT_ID();

-- change the values a few times and finally delete it to test the logging
UPDATE cookbook.auction SET bid = 7.50 WHERE id = 3;
UPDATE cookbook.auction SET bid = 8.25 WHERE id = 3;
UPDATE cookbook.auction SET bid = 8.65 WHERE id = 3;
UPDATE cookbook.auction SET bid = 9.25 WHERE id = 3;
DELETE FROM cookbook.auction WHERE id = 1;

SELECT * FROM cookbook.auction_log WHERE id = 2 ORDER BY ts;


-------------------------------------------------------------------------------
-- USING EVENTS TO SCHEDULE DATABASE ACTIONS

-- Create a dummy table to store some continuous logging data.
CREATE TABLE cookbook.mark_log
(
ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
message VARCHAR(100)
);


-- Our logging event will write a string to a new row. To set it up, use a CREATE EVENT statement:
CREATE EVENT cookbook.mark_insert
ON SCHEDULE EVERY 5 MINUTE
DO INSERT INTO cookbook.mark_log (message) VALUES('-- MARK --');

SELECT * FROM cookbook.mark_log;

-- Check the scheduler status by examining the value of the event_scheduler system variable:
SHOW VARIABLES LIKE 'event_scheduler';

-- To enable the scheduler interactively if it’s not running, execute the following statement
SET GLOBAL event_scheduler = 1;

-- That statement enables the scheduler, but only until the server shuts down. To start the
-- scheduler each time the server starts, enable the system variable in your my.cnf option file:
--   [mysqld]
--   event_scheduler=1

-- There are several ways that you can affect event execution to prevent
-- the table from growing forever:
-- (optin a) Drop the event:
DROP EVENT cookbook.mark_insert;

-- (option b) Disable event execution:
ALTER EVENT cookbook.mark_insert DISABLE;
-- And if you want to enble it again later on:
ALTER EVENT cookbook.mark_insert ENABLE;

-- (optiob c) Let the event continue to run, but set up another event that “expires” old mark_log rows.
CREATE EVENT cookbook.mark_expire
ON SCHEDULE EVERY 1 DAY
DO DELETE FROM cookbook.mark_log WHERE ts < NOW() - INTERVAL 2 DAY;


-------------------------------------------------------------------------------
-- WRITING HELPER ROUTINES FOR EXECUTING DYNAMIC SQL
-- Using a prepared SQL statement involves three steps: preparation, execution, and deal-location.
SELECT @tbl_name := 'test_table';
SELECT @val := 2;

-- A common pattern for executing commands is:
SELECT @stmt := CONCAT('CREATE TABLE cookbook.',@tbl_name,' (i INT);');
PREPARE stmt FROM @stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SELECT @stmt := CONCAT('INSERT INTO ',@tbl_name,' (i) VALUES(',@val,')');
PREPARE cookbook.stmt FROM @stmt;
EXECUTE cookbook.stmt;
DEALLOCATE PREPARE cookbook.stmt;

-- But it is so easy to streamline it,
-- We summarize the prepare, execute, and deallocate processes.
DELETE PROCEDURE cookbook.exec_stmt;
DEALLOCATE PROCEDURE cookbook.exec_stmt;

DROP TABLE cookbook.test_table;

CREATE PROCEDURE cookbook.exec_stmt(stmt_str TEXT)
BEGIN
  SELECT @_stmt_str := stmt_str;
  PREPARE stmt FROM @_stmt_str;
  EXECUTE stmt;
  DEALLOCATE PREPARE stmt;
END;

CALL cookbook.exec_stmt(CONCAT('CREATE TABLE cookbook.',@tbl_name,' (i INT)'));

SELECT * FROM cookbook.test_table;

---------------------------------------
-- To work with SQL queries where a source of the query might be external data: 
-- The QUOTE() function is available for quoting data values.
-- There is no corresponding function for identifiers, but it’s easy to write one that
-- doubles internal backticks and adds a backtick at the beginning and end.
CREATE FUNCTION cookbook.quote_identifier(id TEXT)
RETURNS TEXT DETERMINISTIC
RETURN CONCAT('`',REPLACE(id,'`','``'),'`')

-- Revising the preceding example to ensure the safety of data values and identifiers, we have:
SELECT @tbl_name := cookbook.quote_identifier(@tbl_name);
SELECT @val := QUOTE(@val);

DROP TABLE cookbook.test_table;
CALL cookbook.exec_stmt(CONCAT('CREATE TABLE cookbook.',@tbl_name,' (i INT)'));
CALL cookbook.exec_stmt(CONCAT('INSERT INTO cookbook.',@tbl_name,' (i) VALUES(',@val,')'));

SELECT * FROM cookbook.test_table;

-------------------------------------------------------------------------------
-- HANDLING ERRORS WITHIN STORED PROGRAMS
-- Stored programs can also produce their own errors or warnings to signal that something has gone wrong.

---------------------------------------
-- To implement a 'no more data' condition in a SELECT statement,
-- use a cursor-based fetch loop in conjunction with a condition handler that 
-- catches end-of-data condition. This technique has 4 essential components:

-- (a) A cursor associated with a SELECT statement that reads rows. Open the cursor to
-- start reading, and close it to stop.
-- (b) A condition handler that activates when the cursor reaches the end of the result set
-- and raises an end-of-data condition (NOT FOUND). 
-- (c) A variable that indicates loop termination. Initialize the variable to FALSE, then set
-- it to TRUE within the condition handler when the end-of-data condition occurs.
-- (d) A loop that uses the cursor to fetch each row and exits when the loop-termination variable becomes TRUE.

-- Example: a fetch loop that processes the states table row by row to calculate the total US population:
CREATE PROCEDURE cookbook.us_population()
BEGIN
  DECLARE done BOOLEAN DEFAULT FALSE;
  DECLARE state_pop, total_pop BIGINT DEFAULT 0;
  DECLARE cur CURSOR FOR SELECT pop FROM cookbook.states;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
  OPEN cur;
  fetch_loop: LOOP
    FETCH cur INTO state_pop;
    IF done THEN
      LEAVE fetch_loop;
    END IF;
    SET total_pop = total_pop + state_pop;
  END LOOP;
  CLOSE cur;
  SELECT total_pop AS 'Total U.S. Population';
END;

CALL cookbook.us_population();

-- check the calcs
SELECT SUM(pop) FROM cookbook.states;


---------------------------------------
-- CATCHING AND IGNORING ERRORS
CREATE PROCEDURE cookbook.drop_user(user TEXT, host TEXT)
BEGIN
  DECLARE account TEXT;
  DECLARE CONTINUE HANDLER FOR 1396
    SELECT CONCAT('Unknown user: ', account) AS Message;
  SET account = CONCAT(QUOTE(user),'@',QUOTE(host));
  CALL exec_stmt(CONCAT('DROP USER ',account));
END;

CALL cookbook.drop_user('bad-user','localhost');

-- To ignore the error completely, write the handler using an empty BEGIN … END block:
-- DECLARE CONTINUE HANDLER FOR 1396 BEGIN END;

---------------------------------------
-- RAISING ERRORS AND WARNINGS
-- For example, if we want to issue a warning every time a division by zero happens (instead of issuing an error and stopping the program),
-- we can set the ERROR_FOR_DIVISION_BY_ZERO mode. However, this only works for data-modification
-- situations such as INSERT
SET sql_mode = 'ERROR_FOR_DIVISION_BY_ZERO,STRICT_ALL_TABLES';
SELECT 1/0;

-- If we want to handle division by zero in any context,
-- wrtie a function that performs the division, but checks for 
-- divisor first and uses SIGNAL to issue a warning when a division by zero wants to happen

-- The SIGNAL statement specifies a SQLSTATE value plus an optional SET clause you can
-- use to assign values to error attributes. MYSQL_ERRNO corresponds to the MySQL-specific
-- error code, and MESSAGE_TEXT is a string of your choice

CREATE FUNCTION cookbook.divide(numerator FLOAT, divisor FLOAT)
RETURNS FLOAT DETERMINISTIC
BEGIN
  IF divisor = 0 THEN
  SIGNAL SQLSTATE '22012'
  SET MYSQL_ERRNO = 1365, MESSAGE_TEXT = 'unexpected 0 divisor';
  END IF;
  RETURN numerator / divisor;
END;

SELECT cookbook.divide(1,0);


-- SIGNAL can also raise warning conditions, not just errors.
-- For example, the following routine generates a warning that can be displayed with SHOW WARNINGS. 
-- SQLSTATE value 01000 and error 1642 indicate a user-defined unhandled exception.

CREATE PROCEDURE cookbook.drop_user_warn(user TEXT, host TEXT)
BEGIN
  DECLARE account TEXT;
  DECLARE CONTINUE HANDLER FOR 1396
  BEGIN
    DECLARE msg TEXT;
    SET msg = CONCAT('Unknown user: ', account);
    SIGNAL SQLSTATE '01000' SET MYSQL_ERRNO = 1642, MESSAGE_TEXT = msg;
  END;
  SET account = CONCAT(QUOTE(user),'@',QUOTE(host));
  CALL exec_stmt(CONCAT('DROP USER ',account));
END;

CALL cookbook.drop_user_warn('bad-user','localhost');

SHOW cookbook.WARNINGS;


-------------------------------------------------------------------------------
-- USING TRIGGERS TO PREPROCESS OR REJECT DATA
-- If you don’t want to write the validation logic for every INSERT, 
-- centralize the input-testing logic into a BEFORE INSERT trigger.
-- You can either Reject bad data by raising a signal, or preprocess values and modify them.

CREATE TABLE cookbook.contact_info
(
id INT NOT NULL AUTO_INCREMENT,
name VARCHAR(30), # state of residence
state CHAR(2), # state of residence
email VARCHAR(50), # email address
url VARCHAR(255), # web address
PRIMARY KEY (id)
);

-- Suppose we want to impose the following rule set on the new records added to the table above:
-- (a) state must be a two-letter string, only if presemt in the states table.
-- (b) Email address must contain @
-- (c) For URLs, strip the hettp:// to save space

CREATE TRIGGER cookbook.bi_contact_info BEFORE INSERT ON contact_info
FOR EACH ROW
BEGIN
  IF (SELECT COUNT(*) FROM cookbook.states WHERE abbrev = NEW.state) = 0 THEN
  SIGNAL SQLSTATE 'HY000'
  SET MYSQL_ERRNO = 1525, MESSAGE_TEXT = 'invalid state code';
  END IF;
  IF INSTR(NEW.email,'@') = 0 THEN
  SIGNAL SQLSTATE 'HY000'
  SET MYSQL_ERRNO = 1525, MESSAGE_TEXT = 'invalid email address';
  END IF;
  SET NEW.url = TRIM(LEADING 'http://' FROM NEW.url);
END;

-- test the implementation
INSERT INTO cookbook.contact_info (name,state,email,url)
VALUES('Jen','NY','jen@example.com','http://www.example.com');

INSERT INTO cookbook.contact_info (name,state,email,url)
VALUES('Jen','XX','jen@example.com','http://www.example.com');

INSERT INTO cookbook.contact_info (name,state,email,url)
VALUES('Jen','NY','jen','http://www.example.com');

SELECT * FROM cookbook.contact_info;













