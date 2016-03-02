
-- to use transactions, you must use a transaction-safe engine. Currently, the transactional engines include InnoDB and NDB.
SELECT ENGINE FROM INFORMATION_SCHEMA.ENGINES
WHERE SUPPORT IN ('YES','DEFAULT') AND TRANSACTIONS='YES';

-- MyISAM tables are nontransactional and trying to use them for transactions will yield incorrect results because they do not support rollback.

-- to create a table that uses a given engine, add an ENGINE = tbl_engine clause to your CREATE TABLE statement:
CREATE TABLE cookbook.t (i INT) ENGINE = InnoDB;

-------------------------------------------------------------------------------
-- To use transactions in MySQL, you need to enable transactions.
-- There are two ways to enable transactions in MySQL:

-- (option a) turn off transactions on a case-by-case basis.
-- Execute a START TRANSACTION (or BEGIN) statement to suspend auto-commit mode,
-- then execute the statements that make up the transaction. 
-- If the statements succeed, record their effect in the database and 
-- terminate the transaction by executing a COMMIT statement:

DROP TABLE cookbook.t;
CREATE TABLE cookbook.t (i INT) ENGINE = InnoDB;

/*sqldev:stmt*/START TRANSACTION;
INSERT INTO cookbook.t (i) VALUES(1);
INSERT INTO cookbook.t (i) VALUES(2);
/*sqldev:stmt*/COMMIT;

SELECT * FROM cookbook.t;

-- If an error occurs, don’t use COMMIT. Instead, cancel the transaction by executing a ROLLBACK statement.
DROP TABLE cookbook.t;
CREATE TABLE cookbook.t (i INT) ENGINE = InnoDB;

/*sqldev:stmt*/START TRANSACTION;
INSERT INTO cookbook.t (i) VALUES(1);
INSERT INTO cookbook.t (x) VALUES(2);

/*sqldev:stmt*/ROLLBACK;

SELECT * FROM cookbook.t;


-- (option b) turn off transactions for the entire session
-- Another way to group statements is to turn off auto-commit mode explicitly by
-- setting the autocommit session variable to 0. After that, each statement you execute
-- becomes part of the current transaction. To end the transaction and begin the next one, 
-- execute a COMMIT or ROLLBACK statement.

/*sqldev:stmt*/SET autocommit = 0;
INSERT INTO cookbook.t (i) VALUES(1);
INSERT INTO cookbook.t (i) VALUES(2);
/*sqldev:stmt*/COMMIT;

SELECT * FROM cookbook.t;

-- To turn auto-commit mode back on, use this statement:
/*sqldev:stmt*/SET autocommit = 1;


-------------------------------------------------------------------------------
-- USING APIs TO MANAGE SQL TRANSACTIONS
-- See the 'recipes/transactions' directory for Python examples.
-- Also see my own Python codes in pMysql.py files.

-- In Python, The DB API specification indicates that database connections
-- should begin with auto-commit mode disabled. 
-- This means that when you open a connection, a new transaction is opened by default.

-- When working with UPDATE, INSERT, DELETE statements, end each transaction with either commit() or rollback(). 
-- The commit() call occurs within a try statement, and the rollback() occurs within the
-- except clause to cancel the transaction if an error occurs.
-- When working with SELECT statements, you do not need to use commit statement.

-- Also note that you commit or rollback on the connection level, not on the cursor level.

-- A subtle point to be aware of when rolling back within languages that raise exceptions
-- is that it may be possible for the rollback itself to fail, causing another exception to be
-- raised. If you don’t deal with that, your program itself may terminate. To handle this,
-- execute the rollback within another block that has an empty exception handler. The
-- sample programs do this as necessary. This is implemented in the second (nested) try/except block in the code below.

-- Python code:
/*
try:
  cursor = conn.cursor()
  # move some money from one person to the other
  cursor.execute("UPDATE money SET amt = amt - 6 WHERE name = 'Eve'")
  cursor.execute("UPDATE money SET amt = amt + 6 WHERE name = 'Ida'")
  cursor.close()
  conn.commit()
except mysql.connector.Error as e:
  print("Transaction failed, rolling back. Error was:")
  print(e)
  try: # empty exception handler in case rollback fails
    conn.rollback()
  except:
    pass
*/
-------------------------------------------------------------------------------









