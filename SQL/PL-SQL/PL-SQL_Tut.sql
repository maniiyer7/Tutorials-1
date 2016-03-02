
-- http://stackoverflow.com/questions/9957885/how-to-assign-value-to-variable-in-expression-for-a-pl-sql-block
-- http://stackoverflow.com/questions/743772/how-to-reuse-dynamic-columns-in-an-oracle-sql-statement
-- https://docs.oracle.com/cd/B19306_01/appdev.102/b14261/dynamic.htm#i14257

--------------------------------------------------------------------------------
-- TODO: how to use current month shading to calculate estimated peer adj??
SELECT UPPER(TO_CHAR(SYSDATE,'Mon')) FROM DUAL;
SELECT CONCAT(UPPER(TO_CHAR(SYSDATE,'Mon')), '_SHADING') FROM DUAL;
---------------------------------------

---------------------------------------
SELECT MAR_SHADING FROM FACT_SOLAR_FACILITY_ARRAY WHERE ROWNUM < 10;
---------------------------------------

---------------------------------------
-- Using procedure
-- TODO: looks like this procedure is working. I just need to convert it to a function so that it returns the values.
CREATE OR REPLACE PROCEDURE get_current_shading (
  asset_key NUMBER) IS
   shading_column VARCHAR2(30);
   sql_stmt  VARCHAR2(400);
BEGIN
  SELECT CONCAT(UPPER(TO_CHAR(SYSDATE,'Mon')), '_SHADING') INTO shading_column FROM DUAL;
  sql_stmt := 'SELECT ' || shading_column || ' FROM FACT_SOLAR_FACILITY_ARRAY WHERE d_asset_fkey = :1';
  EXECUTE IMMEDIATE sql_stmt USING asset_key;
END get_current_shading;
/

exec get_current_shading(1);
---------------------------------------

---------------------------------------
-- Using simple script
-- TODO: this also seems to work, but it does not return anything
DECLARE
   plsql_block       VARCHAR2(500);
   shading_column    VARCHAR2(20);
BEGIN
-- note the semi-colons (;) inside the quotes '...'
  SELECT CONCAT(UPPER(TO_CHAR(SYSDATE,'Mon')), '_SHADING') INTO shading_column FROM DUAL;
  plsql_block := 'SELECT ' || shading_column || ' FROM FACT_SOLAR_FACILITY_ARRAY WHERE d_asset_fkey = :1';
  EXECUTE IMMEDIATE plsql_block USING 1;
END;
/
---------------------------------------


-------------------------------------------------------------------------------
-- Return results from functions
DECLARE
  TYPE cursor_ref IS REF CURSOR;
  c1 cursor_ref;
  TYPE shading_tab IS TABLE OF NUMBER;
  shade_est_tab shading_tab;
  rows_fetched NUMBER;
  plsql_block       VARCHAR2(500);
  shading_column    VARCHAR2(20);
BEGIN
  OPEN c1 FOR 'SELECT MAR_SHADING FROM FACT_SOLAR_FACILITY_ARRAY';
  FETCH c1 BULK COLLECT INTO shade_est_tab;
  rows_fetched := c1%ROWCOUNT;
  DBMS_OUTPUT.PUT_LINE('Number of shading estimates fetched: ' || TO_CHAR(rows_fetched));
END;
/

select * from shade_est_tab;

---------------------------------------
-- Return value from PL/SQL function
CREATE OR REPLACE FUNCTION get_current_shading_func (asset_key NUMBER) RETURN VARCHAR AS
   shading_column VARCHAR2(30);
   sql_stmt  VARCHAR2(400);
BEGIN
  SELECT CONCAT(UPPER(TO_CHAR(SYSDATE,'Mon')), '_SHADING') INTO shading_column FROM DUAL;
  sql_stmt := 'SELECT ' || shading_column || ' FROM FACT_SOLAR_FACILITY_ARRAY WHERE d_asset_fkey = :1';
  EXECUTE IMMEDIATE sql_stmt USING asset_key;
  RETURN shading_column;
END;
/

-- To call a procedure:
exec get_current_shading_func(1);

-- To call a function:
call get_current_shading_func(1);

-- To see the results of a function 
select get_current_shading_func(1) from dual;
---------------------------------------

-------------------------------------------------------------------------------
-- Return table from PL/SQL function
-- http://www.adp-gmbh.ch/ora/plsql/coll/return_table.html

-- First, we need to create a new object type that contains the fields that are going to be returned:
create or replace type t_col as object (
  i number,
  n varchar2(30)
);
/

-- Then, out of this new type, a nested table type must be created.
create or replace type t_nested_table as table of t_col;
/

-- Now, we're ready to actually create the function.
-- This function adds one line at at time to the table, then fills in the line with a value that we assign to it.
-- It still does not return the results of a query; it just allows us to fill in the table by adding the records one-by-one (in this case, manually).
create or replace function return_table return t_nested_table as
  v_ret   t_nested_table;
  begin
    v_ret  := t_nested_table();
  
    v_ret.extend;
    v_ret(v_ret.count) := t_col(1, 'one');
  
    v_ret.extend;
    v_ret(v_ret.count) := t_col(2, 'two');
  
    v_ret.extend;
    v_ret(v_ret.count) := t_col(3, 'three');
  
    return v_ret;
  end return_table;
/

select * from table(return_table);
---------------------------------------


---------------------------------------
-- Example: using the table object types created above to create a summary of user objects in the SQL session
create or replace function return_objects(
                  p_min_id in number,
                  p_max_id in number
                )
                return t_nested_table as
  v_ret   t_nested_table;
  begin
    select 
    cast(
    multiset(
      select 
        object_id, object_name
      from 
        user_objects
      where
        object_id between p_min_id and p_max_id) 
        as t_nested_table)
      into
        v_ret
      from 
        dual;
  
    return v_ret;
    
  end return_objects;
/

select * from table(return_objects(1,100000));
-------------------------------------------------------------------------------

-- To return the whole table at once you could change the SELECT to:

--    SELECT  ...
--    BULK COLLECT INTO T
--    FROM    ...

-- This is only advisable for results that aren't excessively large, since they all have to be accumulated in memory before being returned; otherwise consider the pipelined function as suggested by Charles, or returning a REF CURSOR.

-------------------------------------------------------------------------------
-- USING PIPELINED FUNCTIONS
-- http://stackoverflow.com/questions/2829880/create-an-oracle-function-that-returns-a-table
-- TODO: this only returns one row. How to return the entire table??
CREATE OR REPLACE PACKAGE test AS

    TYPE measure_record IS RECORD(
       object_name VARCHAR2(50), 
       object_id NUMBER);

    TYPE measure_table IS TABLE OF measure_record;

    FUNCTION get_ups(foo NUMBER)
        RETURN measure_table
        PIPELINED;
END;


CREATE OR REPLACE PACKAGE BODY test AS

    FUNCTION get_ups(foo number)
        RETURN measure_table
        PIPELINED IS

        rec            measure_record;

    BEGIN
        SELECT object_name, object_id
          INTO rec
          FROM user_objects 
          WHERE ROWNUM = 1;

        -- you would usually have a cursor and a loop here. Then you can remove the 'WHERE ROWNUM = 1' constraint and pipe all rows back.
        PIPE ROW (rec);

        RETURN;
    END get_ups;
END;

SELECT * 
  FROM table(test.get_ups(0));
-------------------------------------------------------------------------------


---------------------------------------
-- USING SELECT ... CNULK COLLECT INTO
-- TODO: this throws error
CREATE OR REPLACE PACKAGE test AS

    TYPE measure_record IS RECORD(
       object_name VARCHAR2(50), 
       object_id NUMBER);

    TYPE measure_table IS TABLE OF measure_record;

    FUNCTION get_ups(foo NUMBER)
        RETURN measure_table;
END;


CREATE OR REPLACE PACKAGE BODY test AS

    FUNCTION get_ups(foo number)
        RETURN measure_table
        IS

        rec measure_record;

    BEGIN
        SELECT object_name, object_id
          BULK COLLECT INTO rec
          FROM user_objects 
          ;

        RETURN rec;
    END get_ups;
END;

SELECT * 
  FROM table(test.get_ups(0));
---------------------------------------


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- USING 'EXECUTE IMMEDIATE' COMMAND IN PL/SQL
-- The main argument to EXECUTE IMMEDIATE is the string containing the SQL statement to execute. 
-- You can build up the string using concatenation, or use a predefined string.

-- Bind variables (or bind arguments) are the placeholder variables within a EXECUTE IMMEDIATE statement.
-- We refer to them using :<var name> in the body of the EXECUTE IMMEDIATE block;
-- then, we assign values to them in the USING block, which immediately follows the EXECUTE IMMEDIATE command.
-- We can use variable name, or an ordinal number that corresponds to the order of the bind variable.

-- Except for multi-row queries, the dynamic string can contain any SQL statement or any PL/SQL block. 
-- The string can also contain placeholders, arbitrary names preceded by a colon, for bind arguments. 
-- In this case, you specify which PL/SQL variables correspond to the placeholders with the INTO, USING, and RETURNING INTO clauses.

-- When constructing a single SQL statement in a dynamic string, do not include a semicolon (;) at the end inside the quotation mark. 
-- When constructing a PL/SQL anonymous block, include the semicolon at the end of each PL/SQL statement and at the end of the anonymous block; 
-- there will be a semicolon immediately before the end of the string literal, and another following the closing single quotation mark.

-- You can only use placeholders in places where you can substitute variables in the SQL statement, 
-- such as conditional tests in WHERE clauses. You cannot use placeholders for the names of schema objects.

-- Used only for single-row queries, the INTO clause specifies the variables or record into which column values are retrieved. 
-- For each value retrieved by the query, there must be a corresponding, type-compatible variable or field in the INTO clause.

-- Used only for DML statements that have a RETURNING clause (without a BULK COLLECT clause), 
-- the RETURNING INTO clause specifies the variables into which column values are returned. 
-- For each value returned by the DML statement, there must be a corresponding, type-compatible variable in the RETURNING INTO clause.

-- You can place all bind arguments in the USING clause. The default parameter mode is IN.
-- For DML statements that have a RETURNING clause, you can place OUT arguments in the RETURNING INTO clause without specifying the parameter mode.
-- If you use both the USING clause and the RETURNING INTO clause, the USING clause can contain only IN arguments.

-- At run time, bind arguments replace corresponding placeholders in the dynamic string. 
-- Every placeholder must be associated with a bind argument in the USING clause and/or RETURNING INTO clause. 
-- You can use numeric, character, and string literals as bind arguments, but you cannot use Boolean literals (TRUE, FALSE, and NULL).
-- To pass nulls to the dynamic string, you must use a workaround.
---------------------------------------


---------------------------------------
-- Example of dynamic SQL in a procedure
-- Note the use of 'IS' keyword after the function header
-- Note: placeholder are specified using the convention :<name>
-- The placeholders can be simply a number, in which case the number
-- refers to the relative position of the argument passed to the proceure in USING clause.
-- We can also use explicit names for the placeholders (see the next example).
CREATE OR REPLACE PROCEDURE raise_emp_salary (
  column_value NUMBER, emp_column VARCHAR2, amount NUMBER) IS
   v_column VARCHAR2(30);
   sql_stmt  VARCHAR2(200);
BEGIN
-- determine if a valid column name has been given as input
  SELECT COLUMN_NAME INTO v_column FROM USER_TAB_COLS 
    WHERE TABLE_NAME = 'EMPLOYEES' AND COLUMN_NAME = emp_column;
  sql_stmt := 'UPDATE employees SET salary = salary + :1 WHERE ' 
               || v_column || ' = :2';
  EXECUTE IMMEDIATE sql_stmt USING amount, column_value;
  IF SQL%ROWCOUNT > 0 THEN
    DBMS_OUTPUT.PUT_LINE('Salaries have been updated for: ' || emp_column 
                        || ' = ' || column_value);
  END IF;
  EXCEPTION
  WHEN NO_DATA_FOUND THEN
    DBMS_OUTPUT.PUT_LINE ('Invalid Column: ' || emp_column);
END raise_emp_salary;
/
---------------------------------------

---------------------------------------
-- Example of dynamic SQL PL/SQL block using one-line procedures
-- In this example, we use specific names to refer to placeholders.
DECLARE
   plsql_block       VARCHAR2(500);
BEGIN
-- note the semi-colons (;) inside the quotes '...'
  plsql_block := 'BEGIN raise_emp_salary(:cvalue, :cname, :amt); END;';
  EXECUTE IMMEDIATE plsql_block 
    USING 110, 'DEPARTMENT_ID', 10;
  EXECUTE IMMEDIATE 'BEGIN raise_emp_salary(:cvalue, :cname, :amt); END;'
    USING 112, 'EMPLOYEE_ID', 10;
END;
/
---------------------------------------

---------------------------------------
-- Example of dynamix SQL PL/SQL block using placeholders
-- Use EXECUTE IMMEDIATELY for purposes other than just running the query.
DECLARE
   sql_stmt          VARCHAR2(200);
   v_column          VARCHAR2(30) := 'DEPARTMENT_ID';
   dept_id           NUMBER(4) := 46;
   dept_name         VARCHAR2(30) := 'Special Projects';
   mgr_id            NUMBER(6) := 200;
   loc_id            NUMBER(4) := 1700;
BEGIN
-- note that there is no semi-colon (;) inside the quotes '...'
  EXECUTE IMMEDIATE 'CREATE TABLE bonus (id NUMBER, amt NUMBER)';
  sql_stmt := 'INSERT INTO departments VALUES (:1, :2, :3, :4)';
  EXECUTE IMMEDIATE sql_stmt USING dept_id, dept_name, mgr_id, loc_id;
  EXECUTE IMMEDIATE 'DELETE FROM departments WHERE ' || v_column || ' = :num'
      USING dept_id;
  EXECUTE IMMEDIATE 'ALTER SESSION SET SQL_TRACE TRUE';
  EXECUTE IMMEDIATE 'DROP TABLE bonus';
END;
/
---------------------------------------

---------------------------------------
-- Dynamic SQL Procedure that Accepts Table Name and WHERE Clause 
-- This procedure has both input parameters, and derived parameters.
-- Input parameters are declared in the function header.
-- Other derived parameters are declared at the very beginning of the procedure. Note that we did not use DECLARE keyword.
-- To calculate derived variables:
--   If they are calculated using a query, use the SELECT ... INTO ... syntax.
--   If they are calculated differently, use the := syntax.
-- After building a query using derived and passed-on parameters, run the query using EXECUTE IMMADIATE <query> format.

CREATE TABLE employees_temp AS SELECT * FROM employees;

CREATE OR REPLACE PROCEDURE delete_rows (
   table_name IN VARCHAR2,
   condition  IN VARCHAR2 DEFAULT NULL) AS
   where_clause  VARCHAR2(100) := ' WHERE ' || condition;
   v_table      VARCHAR2(30);
BEGIN
-- first make sure that the table actually exists; if not, raise an exception
  SELECT OBJECT_NAME INTO v_table FROM USER_OBJECTS
    WHERE OBJECT_NAME = UPPER(table_name) AND OBJECT_TYPE = 'TABLE';
   IF condition IS NULL THEN where_clause := NULL; END IF;
   EXECUTE IMMEDIATE 'DELETE FROM ' || v_table || where_clause;
  EXCEPTION
  WHEN NO_DATA_FOUND THEN
    DBMS_OUTPUT.PUT_LINE ('Invalid table: ' || table_name);
END;
/

BEGIN
  delete_rows('employees_temp', 'employee_id = 111');
END;
/
---------------------------------------

---------------------------------------
-- With the USING clause, the mode defaults to IN, so you do not need to specify a parameter mode for input bind arguments.
-- With the RETURNING INTO clause, the mode is OUT, so you cannot specify a parameter mode for output bind arguments.
-- You must specify the parameter mode in more complicated cases, such as this one where you call a procedure from a dynamic PL/SQL block:
CREATE PROCEDURE create_dept (
   deptid IN OUT NUMBER,
   dname  IN VARCHAR2,
   mgrid  IN NUMBER,
   locid  IN NUMBER) AS
BEGIN
   SELECT departments_seq.NEXTVAL INTO deptid FROM dual;
   INSERT INTO departments VALUES (deptid, dname, mgrid, locid);
END;
/
---------------------------------------

---------------------------------------
-- Example: Using IN OUT Bind Arguments to Specify Substitutions
-- If you dont want to create functions, you can use the script format to perform calculations.
-- In this case, you just need to run the job.
DECLARE
   plsql_block VARCHAR2(500);
   new_deptid  NUMBER(4);
   new_dname   VARCHAR2(30) := 'Advertising';
   new_mgrid   NUMBER(6) := 200;
   new_locid   NUMBER(4) := 1700;
BEGIN
   plsql_block := 'BEGIN create_dept(:a, :b, :c, :d); END;';
   EXECUTE IMMEDIATE plsql_block
      USING IN OUT new_deptid, new_dname, new_mgrid, new_locid;
END;
/
---------------------------------------

-------------------------------------------------------------------------------
-- Bulk Dynamic SQL in Pl/SQL 
-- Bulk SQL passes entire collections back and forth, not just individual elements. 
-- This technique improves performance by minimizing the number of context switches between the PL/SQL and SQL engines. 
-- You can use a single statement instead of a loop that issues a SQL statement in every iteration.

-- Bulk dynamic SQL statements use the following commands:
-- BULK FETCH statement
-- BULK EXECUTE IMMEDIATE statement
-- FORALL statement
-- COLLECT INTO clause
-- RETURNING INTO clause
-- %BULK_ROWCOUNT cursor attribute

-- Bulk binding lets Oracle bind a variable in a SQL statement to a collection of values. 
-- The collection type can be any PL/SQL collection type: index-by table, nested table, or varray. 
-- The collection elements must have a SQL datatype such as CHAR, DATE, or NUMBER. 
-- Three statements support dynamic bulk binds: EXECUTE IMMEDIATE, FETCH, and FORALL.

-- You can use the BULK COLLECT INTO clause with the EXECUTE IMMEDIATE statement to store values from each column of a query's result set in a separate collection.
-- You can use the RETURNING BULK COLLECT INTO clause with the EXECUTE IMMEDIATE statement to store the results of an INSERT, UPDATE, or DELETE statement in a set of collections.
-- You can use the BULK COLLECT INTO clause with the FETCH statement to store values from each column of a cursor in a separate collection.
-- You can put an EXECUTE IMMEDIATE statement with the RETURNING BULK COLLECT INTO inside a FORALL statement. You can store the results of all the INSERT, UPDATE, or DELETE statements in a set of collections.

-- You can pass subscripted collection elements to the EXECUTE IMMEDIATE statement through the USING clause. 
-- You cannot concatenate the subscripted elements directly into the string argument to EXECUTE IMMEDIATE; 
-- for example, you cannot build a collection of table names and write a FORALL statement where each iteration applies to a different table.


---------------------------------------
-- Exmaple of  Dynamic SQL with BULK COLLECT INTO Clause.
-- You can bind define variables in a dynamic query using the BULK COLLECT INTO clause. 
-- You can then use that clause in a bulk FETCH or bulk EXECUTE IMMEDIATE statement.
-- In this example, we used both FETCH and EXECUTE IMMEDIATE to demonstrate their application.

-- Note how we first declared a cursor type, and then created a cursor object from it.
-- Note how we defined two table types (one for storing numbers and one for storing strings), and then created separate instances of them.
-- Note when using the cursor, we first opened the cursor within the query (the cursor is bind to a specific query). Then we fetched that cursor and used BULK COLLECT INTO to save the results.

-- You use three statements to process a dynamic multi-row query: OPEN-FOR, FETCH, and CLOSE. 
-- First, you OPEN a cursor variable FOR a multi-row query. 
-- Then, you FETCH rows from the result set one at a time. 
-- When all the rows are processed, you CLOSE the cursor variable.

DECLARE
   TYPE EmpCurTyp IS REF CURSOR;
   TYPE NumList IS TABLE OF NUMBER;
   TYPE NameList IS TABLE OF VARCHAR2(25);
   emp_cv EmpCurTyp;
   empids NumList;
   enames NameList;
   sals   NumList;
BEGIN
   OPEN emp_cv FOR 'SELECT employee_id, last_name FROM employees';
   FETCH emp_cv 
      BULK COLLECT INTO empids, enames;
   CLOSE emp_cv;
   EXECUTE IMMEDIATE 'SELECT salary FROM employees'
      BULK COLLECT INTO sals;
END;
/
---------------------------------------
select * from user_objects;
---------------------------------------
-- Only INSERT, UPDATE, and DELETE statements can have output bind variables. 
-- You bulk-bind them with the RETURNING BULK COLLECT INTO clause of EXECUTE IMMEDIATE.

-- Note that we can assign a value to a variable in the DECLARE statement block.
DECLARE
   TYPE NameList IS TABLE OF VARCHAR2(15);
   enames    NameList;
   bonus_amt NUMBER := 50;
   sql_stmt  VARCHAR(200);
BEGIN
   sql_stmt := 'UPDATE employees SET salary = salary + :1 
                RETURNING last_name INTO :2';
   EXECUTE IMMEDIATE sql_stmt
      USING bonus_amt RETURNING BULK COLLECT INTO enames;
END;
/
---------------------------------------

---------------------------------------
-- To bind the input variables in a SQL statement, you can use the FORALL statement and USING clause.
-- The SQL statement cannot be a query.
-- FORALL is the same as 'for' statements (creating loops).
-- Also note how we create List data types. PL/SQL does not have a List data type out of the box; but 
-- we can easily define a List data type as shown below.

DECLARE
   TYPE NumList IS TABLE OF NUMBER;
   TYPE NameList IS TABLE OF VARCHAR2(15);
   empids NumList;
   enames NameList;
BEGIN
   empids := NumList(101,102,103,104,105);
   FORALL i IN 1..5
      EXECUTE IMMEDIATE
        'UPDATE employees SET salary = salary * 1.04 WHERE employee_id = :1
         RETURNING last_name INTO :2'
         USING empids(i) RETURNING BULK COLLECT INTO enames;
END;
/
---------------------------------------

---------------------------------------
-- When building up a single SQL statement in a string, do not include any semicolon at the end.
-- When building up a PL/SQL anonymous block, include the semicolon at the end of each PL/SQL statement and at the end of the anonymous block. 
BEGIN
   EXECUTE IMMEDIATE 'BEGIN DBMS_OUTPUT.PUT_LINE(''semicolons''); END;';
END;
/
---------------------------------------

---------------------------------------
-- When you build up dynamic SQL statements, use specify the bind variables and USING clause, instead of manually building the SQL string using dynamic column name.
-- If you use dynamic column name instead of bind variables, Oracle opens a different cursor for each distinct value of the field (in the example below, emp_id). 
-- This can lead to resource contention and poor performance as each statement is parsed and cached.

-- BAD performance. AVOID:
CREATE PROCEDURE fire_employee (emp_id NUMBER) AS
BEGIN
   EXECUTE IMMEDIATE
      'DELETE FROM employees WHERE employee_id = ' || TO_CHAR(emp_id);
END;
/

-- GOOD performance. USE:
CREATE PROCEDURE fire_employee (emp_id NUMBER) AS
BEGIN
   EXECUTE IMMEDIATE
      'DELETE FROM employees WHERE employee_id = :id' USING emp_id;
END;
/
---------------------------------------

---------------------------------------
-- Suppose you need a procedure that accepts the name of any database table, then drops that table from your schema. 
-- You must build a string with a statement that includes the object names, then use EXECUTE IMMEDIATE to execute the statement.
-- In such case, use concatenation to build the string, rather than trying to pass the table name as a bind variable through the USING clause. 
-- Since the variable has only one name (the name of the table you want to delete), this does not impact performance.

CREATE TABLE employees_temp AS SELECT last_name FROM employees;

CREATE PROCEDURE drop_table (table_name IN VARCHAR2) AS
BEGIN
  EXECUTE IMMEDIATE 'DROP TABLE ' || table_name;
END;
/
---------------------------------------

---------------------------------------
-- If you need to call a procedure whose name is unknown until runtime, you can pass a parameter identifying the procedure. 
-- For example, the following procedure can call another procedure (drop_table) by specifying the procedure name when executed.
-- Note that in this case, we need to define the DROP_TABLE procedure separately (see above).
CREATE PROCEDURE run_proc (proc_name IN VARCHAR2, table_name IN VARCHAR2) AS
BEGIN
   EXECUTE IMMEDIATE 'CALL "' || proc_name || '" ( :table_name )' using table_name;
END;
/

-- Now to use the table above:
BEGIN 
  run_proc('DROP_TABLE', 'employees_temp'); 
END;
/
---------------------------------------

---------------------------------------
-- Placeholders in a dynamic SQL statement are associated with bind arguments in the USING clause by position, not by name. 
-- Therefore, if you specify a sequence of placeholders like :a, :a, :b, :b, you must include four items in the USING clause. 

-- Note that in this example, we do not have a BEGIN/END block. Thereofore, this is a dynamic SQL statement, and not a PL/SQL block.
-- However, as the next example shows, when we use BEGIN/END block (ie, use a PL/SQL block), the rules change.
sql_stmt := 'INSERT INTO payroll VALUES (:x, :x, :y, :x)';
EXECUTE IMMEDIATE sql_stmt USING a, a, b, a;


-- However, If the dynamic statement represents a PL/SQL block, the rules for duplicate placeholders are different. 
-- In that case, each unique placeholder maps to a single item in the USING clause. 
-- If the same placeholder appears two or more times, all references to that name correspond to one bind argument in the USING clause. 
-- In the example below, all references to the placeholder x are associated with the first bind argument a, and the second unique placeholder y is associated with the second bind argument b.

CREATE PROCEDURE calc_stats(w NUMBER, x NUMBER, y NUMBER, z NUMBER) IS
BEGIN
  DBMS_OUTPUT.PUT_LINE(w + x + y + z);
END;
/

DECLARE
   a NUMBER := 4;
   b NUMBER := 7;
   plsql_block VARCHAR2(100);
BEGIN
   plsql_block := 'BEGIN calc_stats(:x, :x, :y, :x); END;';
   EXECUTE IMMEDIATE plsql_block USING a, b;
END;
/
---------------------------------------

-------------------------------------------------------------------------------
-- USING CURSOR ATTRIBUTES WITH DYNAMIC SQL
-- The SQL cursor attributes %FOUND, %ISOPEN, %NOTFOUND, and %ROWCOUNT work when you issue an INSERT, UPDATE, DELETE, or single-row SELECT statement in dynamic SQL:
BEGIN
  EXECUTE IMMEDIATE 'DELETE FROM employees WHERE employee_id > 1000';
  DBMS_OUTPUT.PUT_LINE('Number of employees deleted: ' || TO_CHAR(SQL%ROWCOUNT));
END;
/
---------------------------------------

---------------------------------------
-- When appended to a cursor variable name, the cursor attributes return information about the execution of a multi-row query:
-- Note how we defined a REF CURSOR type to create a reference to the cursor. The instance of the reference is defined in the next line.
-- Also note how we used the cursor object: first OPEN the cursor; then FETCH the results; you can CLOSE the cursor after done (in this case we did not).
DECLARE
  TYPE cursor_ref IS REF CURSOR;
  c1 cursor_ref;
  TYPE emp_tab IS TABLE OF employees%ROWTYPE;
  rec_tab emp_tab;
  rows_fetched NUMBER;
BEGIN
  OPEN c1 FOR 'SELECT * FROM employees';
  FETCH c1 BULK COLLECT INTO rec_tab;
  rows_fetched := c1%ROWCOUNT;
  DBMS_OUTPUT.PUT_LINE('Number of employees fetched: ' || TO_CHAR(rows_fetched));
END;
/
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- PASSING NULL TO DYNAMIC SQL
-- The literal NULL is not allowed in the USING clause. To work around this restriction, replace the keyword NULL with an uninitialized variable.
CREATE TABLE employees_temp AS SELECT * FROM EMPLOYEES;
DECLARE
   a_null CHAR(1); -- set to NULL automatically at run time
BEGIN
   EXECUTE IMMEDIATE 'UPDATE employees_temp SET commission_pct = :x' USING a_null;
END;
/
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- USING DATABASE LINKS WITH DYNAMIC PL/SQL
-- PL/SQL subprograms can execute dynamic SQL statements that use database links to refer to objects on remote databases.

CREATE PROCEDURE delete_dept (db_link VARCHAR2, dept_id INTEGER) IS
BEGIN
   EXECUTE IMMEDIATE 'DELETE FROM departments@' || db_link ||
      ' WHERE department_id = :num' USING dept_id;
END;
/

-- delete department id 41 in the departments table on the remote DB hr_db
CALL delete_dept('hr_db', 41); 
---------------------------------------

---------------------------------------
-- The targets of remote procedure calls (RPCs) can contain dynamic SQL statements. 
-- For example, suppose the following standalone function, which returns the number of rows in a table, resides on the hr_db database in London:
CREATE FUNCTION row_count (tab_name VARCHAR2) RETURN NUMBER AS
   rows NUMBER;
BEGIN
   EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM ' || tab_name INTO rows;
   RETURN rows;
END;
/

-- From an anonymous block, you might call the function remotely, as follows:
DECLARE
   emp_count INTEGER;
BEGIN
   emp_count := row_count@hr_db('employees');
   DBMS_OUTPUT.PUT_LINE(emp_count);
END;
/
---------------------------------------

-------------------------------------------------------------------------------
-- Using Dynamic SQL With PL/SQL Records and Collections

-- You can fetch rows from the result set of a dynamic multi-row query into a record:
DECLARE
   TYPE EmpCurTyp IS REF CURSOR;
   emp_cv   EmpCurTyp;
   emp_rec  employees%ROWTYPE;
   sql_stmt VARCHAR2(200);
   v_job   VARCHAR2(10) := 'ST_CLERK';
BEGIN
   sql_stmt := 'SELECT * FROM employees WHERE job_id = :j';
   OPEN emp_cv FOR sql_stmt USING v_job;
   LOOP
     FETCH emp_cv INTO emp_rec;
     EXIT WHEN emp_cv%NOTFOUND;
     DBMS_OUTPUT.PUT_LINE('Name: ' || emp_rec.last_name || ' Job Id: ' ||
                           emp_rec.job_id);
   END LOOP;
   CLOSE emp_cv;
END;
/
---------------------------------------

-- https://docs.oracle.com/cd/B19306_01/appdev.102/b14261/dynamic.htm#i14257
-- https://docs.oracle.com/cd/B19306_01/appdev.102/b14261/sqloperations.htm#i45288














