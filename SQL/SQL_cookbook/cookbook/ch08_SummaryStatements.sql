-- Chapter 8: Generating Summaries


-------------------------------------------------------------------------------
-- FINDING COUNT OF SPECIFIC RECORDS

-- To count all records in a table or a subset of table returned by a query, use SELECT COUNT(*). 
-- Without a WHERE clause, the statement counts all the rows in the table,
SELECT COUNT(*) FROM cookbook.driver_log;

-- For InnoDB tables, you may want to avoid it because it can be slow for large tables. 
-- If an approximate row count is good enough, avoid a full scan by extracting the
-- TABLE_ROWS value from the INFORMATION_SCHEMA database
SELECT 
  TABLE_ROWS 
FROM 
  INFORMATION_SCHEMA.TABLES
WHERE 
  TABLE_SCHEMA = 'cookbook' 
  AND TABLE_NAME = 'states';

---------------------------------------
-- Specify constraints to calculate the count of particular records
SELECT 
  COUNT(*) 
FROM 
  cookbook.states
WHERE 
  statehood BETWEEN '1800-01-01' AND '1899-12-31';

---------------------------------------
-- If you use an expression in the COUNT() clause, it returns the number of non-NULL values
-- This can be helpful in writing summary queries while specifying more complex criteria for being included in the summation.
SELECT
  COUNT(IF(DAYOFWEEK(trav_date)=7,1,NULL)) AS 'Saturday trips',
  COUNT(IF(DAYOFWEEK(trav_date)=1,1,NULL)) AS 'Sunday trips'
FROM 
  cookbook.driver_log;


-------------------------------------------------------------------------------
-- CALCULATING SUMMARY STATISTICS
-- MIN() and MAX() can be applied to the data or to functions of data
SELECT
  MIN(name) AS first,
  MAX(name) AS last,
  MIN(CHAR_LENGTH(name)) AS shortest,
  MAX(CHAR_LENGTH(name)) AS longest
FROM 
  cookbook.states;

-- SUM() and AVG()
SELECT
  SUM(size) AS 'total traffic',
  AVG(size) AS 'average message size'
FROM 
  cookbook.mail;

-- Finding total time elapsed
SELECT 
  SUM(TIME_TO_SEC(t1)) AS 'total seconds',
  SEC_TO_TIME(SUM(TIME_TO_SEC(t1))) AS 'total time'
FROM 
  cookbook.time_val;


-------------------------------------------------------------------------------
-- To determine the number of different drivers, use COUNT(DISTINCT):
SELECT COUNT(DISTINCT name) FROM cookbook.driver_log;


-- COUNT(DISTINCT) ignores NULL values. To count NULL as one of the values in the set if
-- it’s present, use one of the following expressions
SELECT COUNT(DISTINCT val) + IF(COUNT(IF(val IS NULL,1,NULL))=0,0,1) FROM ...;

COUNT(DISTINCT val) + IF(SUM(ISNULL(val))=0,0,1) FROM ...;

COUNT(DISTINCT val) + (SUM(ISNULL(val))<>0) FROM ...;


-- When used with multiple columns, DISTINCT shows the different combinations of values
-- in the columns and COUNT(DISTINCT) counts the number of combinations
SELECT DISTINCT srcuser, dstuser FROM cookbook.mail
ORDER BY srcuser, dstuser;

-- Create views to prevent typing the same long query multiple times
CREATE VIEW cookbook.trip_summary_view AS
SELECT
  COUNT(IF(DAYOFWEEK(trav_date) IN (1,7),1,NULL)) AS weekend_trips,
  COUNT(IF(DAYOFWEEK(trav_date) IN (1,7),NULL,1)) AS weekday_trips
FROM 
  cookbook.driver_log;

SELECT * FROM cookbook.trip_summary_view;


-------------------------------------------------------------------------------
-- USING SUMMARY STATS TO CHOOSE ROWS
-- Aggregate functions such as MIN() and MAX() cannot be used in WHERE clauses, 
-- which require expressions that apply to individual rows.
-- SQL uses the WHERE clause to determine which rows to select, but the value of an aggregate function
-- is known only after selecting the rows from which the function’s value is determined.

-- One way to select rows based on the results of a summary statement is to use user-defined variables.
SELECT @max := (SELECT MAX(pop) FROM cookbook.states);

SELECT 
  pop AS 'highest population', 
  name 
FROM 
  cookbook.states 
WHERE 
  pop = @max;


---------------------------------------
-- The other strategy to select rows based on the results of a summary statement is to use sub-query.
SELECT 
  pop AS 'highest population', 
  name 
FROM 
  cookbook.states
WHERE 
  pop = (SELECT MAX(pop) FROM cookbook.states);

-- Note that a WHERE statement can include a derived values, but it cannot include summary values (or aliases).
-- As a result, we can use a derived value together with a sub-query in the WHERE statement:
SELECT 
  bname, cnum, vnum, vtext 
FROM 
  cookbook.kjv
WHERE 
  CHAR_LENGTH(vtext) = (SELECT MIN(CHAR_LENGTH(vtext)) FROM cookbook.kjv);

SELECT MIN(CHAR_LENGTH(vtext)) FROM cookbook.kjv;


---------------------------------------
-- Yet another way to select other columns from rows containing 
-- a minimum or maximum value is to use a join.
-- Select the value into another table, then join it to the original table
-- to select the row that matches the value.
DROP TABLE cookbook.tmp;

CREATE TEMPORARY TABLE cookbook.tmp 
SELECT MAX(pop) as maxpop FROM cookbook.states;

SELECT 
  states.* 
FROM 
  cookbook.states 
  INNER JOIN cookbook.tmp
    ON cookbook.states.pop = cookbook.tmp.maxpop;


-------------------------------------------------------------------------------
-- COMPARING STRINGS
-- To compare case-insensitive strings in case-sensitive fashion, order the values using a case-sensitive collation:
SELECT
  MIN(str_col COLLATE latin1_general_cs) AS min,
  MAX(str_col COLLATE latin1_general_cs) AS max
FROM tbl;

-- To compare case-sensitive strings in case-insensitive fashion, order the values using a case-insensitive collation:
SELECT
  MIN(str_col COLLATE latin1_swedish_ci) AS min,
  MAX(str_col COLLATE latin1_swedish_ci) AS max
FROM tbl;

-- compare values that have all been converted to the same lettercase, which makes lettercase irrelevant.
-- note that this will also change the results
SELECT
  MIN(UPPER(str_col)) AS min,
  MAX(UPPER(str_col)) AS max
FROM tbl;

-- To compare binary strings using a case-insensitive ordering, convert
-- them to non-binary strings and apply an appropriate collation:
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
-- SUMMARY STATISTICS BY SUB-GROUPS 
-- For a summary of each subgroup of a set of rows, not an overall summary value, use a GROUP BY statement
SELECT srcuser, COUNT(*) 
FROM cookbook.mail 
GROUP BY srcuser;

-- We can GROUP BY a column, and get summary stats for other columns
SELECT 
  srcuser,
  SUM(size) AS 'total bytes',
  AVG(size) AS 'bytes per message'
FROM 
  cookbook.mail 
GROUP BY 
  srcuser;

-- To create more fine-grained groupings, use two grouping columns. 
-- This produces a result with nested groups (groups within groups):
SELECT 
  srcuser, srchost, COUNT(srcuser) 
FROM 
  cookbook.mail
GROUP BY 
  srcuser, srchost;

-- you can similarly use the other summarizing functions, such as MAX()
SELECT srcuser, dstuser, MAX(size) 
FROM cookbook.mail 
GROUP BY srcuser, dstuser;


---------------------------------------
-- When you include a GROUP BY clause in a query, the only values that
-- you can meaningfully select are the grouping columns or summary values calculated from the groups.
-- You may even get a response (no error) from MySQL if you use a non-grouping column in the SELECT statement,
-- but the query results will be wrong, since 
-- SQL will return the first value that finds, not the value corresponding to the summary value.
-- If you display additional table columns, they’re not tied to the grouped
-- columns and the values displayed for them are indeterminate.

-- Example: in the query below, trav_date is not repeated in the GROUP BY statement, and it does not have a summary statistics applied to it in the SELECT statement.
-- As a result, the query result will show the correct max(miles), but for trav_date it only shows the first trav_date that it finds. For example, for Suzi, the trav_date associated with 502 miles
-- is 2014-08-02 and not 2014-07-29 as the query results suggest. 
SELECT 
  name, 
  trav_date, 
  MAX(miles) AS 'longest trip'
FROM 
  cookbook.driver_log 
GROUP BY 
  name;

SELECT * FROM cookbook.driver_log;


-- To get the max miles and the date associated with it for each driver, we need a sub-query.
-- In general, if we want to get several summary stats (and only summary stats) per group, the GROUP BY statement can help us.
-- However, if we want to mix summary stats and ordinary records per group, we need a sub-query to get the rows with summary stats, and then find the ordinary records associated with those summary stats in the wrapper query.
-- One exception to this rule is when the ordinary records are all the same (i.e., they were repeated). Then we can just include them in the SELECT statement of the GROUP BY statement (may need to put them in a summary stat function, but that is just for syntax reasons).
SELECT
  dl1.name,
  trav_date,
  miles AS 'longest trip'
FROM
  cookbook.driver_log dl1
  INNER JOIN 
    (SELECT name, MAX(miles) AS max_miles FROM cookbook.driver_log GROUP BY name) dl_max
    ON dl1.name = dl_max.name 
    AND dl1.miles = dl_max.max_miles;
    

---------------------------------------
-- The general solution is either using a sub-query or a temporary table to
-- link each driver with his/her longest drive day.

-- Step a: create the temp table
DROP TABLE cookbook.t;

CREATE TEMPORARY TABLE cookbook.t
SELECT 
  name, 
  MAX(miles) AS miles 
FROM 
  cookbook.driver_log 
GROUP BY 
  name;

SELECT * FROM cookbook.t;

-- Step b: Create summary stats, and use the temp table to connect summary stats to their respective records.
SELECT 
  d.name, 
  d.trav_date, 
  d.miles AS 'longest trip'
FROM 
  cookbook.driver_log AS d 
  INNER JOIN cookbook.t 
    USING (name, miles) 
ORDER BY name;


---------------------------------------
-- To prevent the inadvertantly wrong query results that columns not included in GROUP BY 
-- can cause (see above for explanation and an example), 
-- set the ONLY_FULL_GROUP_BY SQL mode.
-- Note 1: most Oracle implementations do this by default.
-- Note 2: This option is available in SQLPlus. Thus, I was not able to set it on my localhost.
SET sql_mode = 'ONLY_FULL_GROUP_BY';
SELECT name, trav_date, MAX(miles) AS 'longest trip'
FROM cookbook.driver_log GROUP BY name;


--------------------------------------------------------------------------------
-- HOW AGGREGATE FUNCTIONS HANDLE NULL VALUES?
-- Most aggregate functions ignore NULL values. COUNT() is different: 
-- COUNT(expr) ignores NULL instances of expr, but COUNT(*) counts rows, regardless of content.
SELECT subject, test, score 
FROM cookbook.expt 
ORDER BY subject, test;

SELECT 
  subject,
  COUNT(score) AS n,
  SUM(score) AS total,
  AVG(score) AS average,
  MIN(score) AS lowest,
  MAX(score) AS highest
FROM 
  cookbook.expt 
GROUP BY 
  subject;


---------------------------------------
-- By default, an aggregate value of a column with all NULL values will result in NULL.
-- The only exception is COUNT, which will show the actual number (which will be zero, if we select only NULL values).
SELECT 
  subject,
  COUNT(*) AS n_records,
  COUNT(score) AS n_scores,
  SUM(score) AS total,
  AVG(score) AS average,
  MIN(score) AS lowest,
  MAX(score) AS highest
FROM 
  cookbook.expt 
WHERE 
  score IS NULL 
GROUP BY 
  subject;
  
-- If you don’t want an aggregate value of NULL to display as NULL, use IFNULL() to map it appropriately
SELECT 
  subject,
  COUNT(score) AS n,
  IFNULL(SUM(score),0) AS total,
  IFNULL(AVG(score),0) AS average,
  IFNULL(MIN(score),'Unknown') AS lowest,
  IFNULL(MAX(score),'Unknown') AS highest
FROM 
  cookbook.expt 
WHERE 
  score IS NULL 
GROUP BY 
  subject;


---------------------------------------
-- The different forms of COUNT() can be very useful for counting missing values.
SELECT COUNT(*) - COUNT(score) AS missing FROM cookbook.expt;

-- Use GROUP BY to find missing items per group.
SELECT subject,
  COUNT(*) AS total,
  COUNT(score) AS 'nonmissing',
  COUNT(*) - COUNT(score) AS missing
FROM 
  cookbook.expt 
GROUP BY 
  subject;


---------------------------------------
-- HAVING
-- Use HAVING statement to calculate group summaries but
-- display results only for groups that match certain criteria.

-- Note: to understand the difference between WHERE and HAVING clauses,
-- we first need to establish the order of SQL operations.
-- WHERE specifies the initial constraints that determine which rows to select.
-- The value of aggregate functions such as COUNT() can be determined only after the rows have been selected.
-- HAVING applies to group characteristics rather than to single rows. I.e., HAVING operates on the already-selected-and-grouped set of rows.

-- Therefore, the following query will throw an error, because COUNT(*) is not available to WHERE clause
SELECT COUNT(*), name 
FROM cookbook.driver_log
WHERE COUNT(*) > 3
GROUP BY name;

-- However, the following query will work because HAVING can wait 
-- until the query has established the groups and have calculated their COUNT(*) values.
-- In this case, HAVING clause prevents the use of a sub-query.
SELECT COUNT(*), name 
FROM cookbook.driver_log
GROUP BY name
HAVING COUNT(*) > 3;

-- Since HAVING operates after the records are selected and aggregate functions have operated,
-- it can use aliases.
SELECT 
  COUNT(*) AS count, 
  name 
FROM 
  cookbook.driver_log
GROUP BY 
  name
HAVING 
  count > 3;


---------------------------------------
-- DISTINCT eliminates duplicates but doesn’t show which values actually were duplicated
-- in the original data. 
-- You can use HAVING to find unique values in situations to which
-- DISTINCT does not apply. HAVING can tell you which values were unique or nonunique.

-- Example: show the days on which only one driver was active.
SELECT 
  trav_date, COUNT(trav_date) 
FROM 
  cookbook.driver_log
GROUP BY 
  trav_date 
HAVING 
  COUNT(trav_date) = 1;


---------------------------------------
-- This pattern can also work with combination of values.
SELECT 
  srcuser, dstuser
FROM 
  cookbook.mail
GROUP BY 
  srcuser, dstuser 
HAVING 
  COUNT(*) = 1;


---------------------------------------
-- Use calculations as a basis for grouping in a GROUP BY clause. You can even use aliases in GROUP BY and ORDER BY statements.
-- GROUP BY, like ORDER BY, can refer to expressions and aliases. 
-- Remember that you cannot use calculations or aliases in a WHERE clause, because WHERE is applied to the table before the rows are selected.

-- Example: find days of the year on which more than one state joined the Union
-- Note on how to create this table: always start with the aggreate function and GROUP BY statements. 
-- Once you verify the accuracy of the aggregate functions, use a HAVING clause to add constraints to results. 
-- This is asimilar to creatng a wrapper query around the query that has GROUP BY clause.
SELECT
  MONTHNAME(statehood) AS month,
  DAYOFMONTH(statehood) AS day,
  COUNT(*) AS count
FROM 
  cookbook.states 
GROUP BY 
  month, 
  day 
HAVING 
  count > 1;


--------------------------------------------------------------------------------
-- ORGANIZING NUMERICAL VALUES INTO CATEGORICAL BUCKETS
-- Divide the numerical value by the bin width of interest, and then floor the results.

-- Thought process for the query below:
-- (a) first, we need to convert the numerical population values to categorical, hence the first line
-- (b) then, we want to find the count of states per population category. This is done as normal: 
-- add a GROUP BY clause, and an aggregated function (in this case, COUNT) to the query.

SELECT 
  (FLOOR(pop/5000000)+1)*5 AS `max population (millions)`,
  COUNT(*) AS `number of states`
FROM 
  cookbook.states 
GROUP BY 
  `max population (millions)`;

-- Note that the aliases in the preceding queries are written using backticks
-- (identifier quoting) rather than single quotes (string quoting). 
-- Quoted aliases in the GROUP BY clause must use identifier quoting. Otherwise, the alias is treated as a constant string expression and 
-- the grouping produces the wrong result. 
-- Identifier quoting clarifies to MySQL that the alias refers to an output column. 
-- The aliases in the output column list could have been written using string quoting; but
-- we also used identifier quoting for them to be consistent with the GROUP BY clause.

---------------------------------------
-- To see how repetitive the values in a column are (the cardinality of the data):
SELECT COUNT(DISTINCT pop) / COUNT(pop) FROM cookbook.states;
-- A result close to zero indicates a high degree of
-- repetition, which means the values will group into a small number of categories naturally.
-- A result of 1 or close to it indicates many unique values, with the consequence that
-- GROUP BY won’t be very efficient for grouping the values into categories.
-- In such cases, it is good practice to bin the data into buckets using the technique shown above begore using GROUP BY.


---------------------------------------
-- MIN() and MAX() find the values at the endpoints of a set of values, 
-- but to find the endpoints of a set of summary values, those functions won’t work. 
-- Their argument cannot be another aggregate function.

-- To select only the row for the driver with the most miles, the following doesn’t work
SELECT name, SUM(miles)
FROM cookbook.driver_log
GROUP BY name
HAVING SUM(miles) = MAX(SUM(miles));

-- order the rows with the largest SUM() values first and use LIMIT to select the first row:
SELECT name, SUM(miles)
FROM cookbook.driver_log
GROUP BY name
ORDER BY SUM(miles) DESC 
LIMIT 1;

--------------------------------------------------------------------------------
-- to get the most common initial letter for state names:
-- The thought process for this query is as follows:
-- (a) We are looking for initial letter of state name, so first create that using LEFT(name, 1) as letter
-- (b) We want to find the most "common", so we need to first have the count per category. 
--     Since the variable letter is already categorized, we just GROUP BY it and add a COUNT(*) to the SELECT clause.
-- (c) Since we want the "the most" common, we add an ORDER BY DESC and then a LIMIT 1 statement.

SELECT 
  LEFT(name,1) AS letter,
  COUNT(*) 
FROM 
  cookbook.states
GROUP BY 
  letter 
ORDER BY 
  COUNT(*) DESC
LIMIT 1;


-- Using a LIMIT 1 clause runs the risk to leave out multiple top results when they tie with the first one.
-- To find all most-frequent values when there
-- may be more than one, use a user-defined variable or subquery to determine the maximum count, 
-- then select those values with a count equal to the maximum:
SELECT 
  LEFT(name,1) AS letter, 
  COUNT(*) 
FROM 
  cookbook.states
GROUP BY 
  letter 
HAVING 
  COUNT(*) = 
  (SELECT COUNT(*) FROM cookbook.states
  GROUP BY LEFT(name,1) ORDER BY COUNT(*) DESC LIMIT 1);


---------------------------------------
-- To produce a summary based on date or time values,
-- use GROUP BY to place temporal values into categories of the appropriate duration
SELECT 
  trav_date,
  COUNT(*) AS 'number of drivers', 
  SUM(miles) As 'miles logged'
FROM 
  cookbook.driver_log 
GROUP BY trav_date;

-- You can change the frequency
SELECT 
  YEAR(trav_date) AS year, 
  MONTH(trav_date) AS month,
  COUNT(*) AS 'number of drivers', 
  SUM(miles) As 'miles logged'
FROM 
  cookbook.driver_log 
GROUP BY year, month;


--------------------------------------------------------------------------------
-- WORKING WITH PER GROUP AND OVERALL SUMMARY VALUES SIMULTANEOUSLY
-- I.e., working with different levels of summary detail.
-- One solution is to use two statements that retrieve different levels of summary information. 
-- The other apporach is to use a subquery to retrieve one summary value and refer to it in the outer query that refers to other summary values.
-- A third approach is to use WITHROLLUP

-- Example of using a subquery to compute total miles driven of each driver as 
-- a percentage of total miles driven of all drivers:
SELECT 
  name,
  SUM(miles) AS 'miles/driver',
  (SUM(miles)*100) / (SELECT SUM(miles) FROM cookbook.driver_log) AS 'percent of total miles'
FROM 
  cookbook.driver_log 
GROUP BY 
  name;

---------------------------------------
-- Suppose that you want to display drivers
-- who had a lower average miles per day than the group average. 
-- Calculate the overall average in a subquery, and 
-- then compare each driver’s average to the overall average using a HAVING clause:
-- Note that since we want to compare a summary value of each dirver (his/her average miles), we need to use a HAVING clause instead of WHERE clause.
SELECT 
  name, 
  AVG(miles) AS driver_avg 
FROM 
  cookbook.driver_log
GROUP BY 
  name
HAVING 
  driver_avg < 
  (SELECT AVG(miles) FROM cookbook.driver_log);


--To display different summary-level values (and not perform calculations involving 
-- one summary level against another), add WITH ROLLUP to the GROUP BY clause
SELECT 
  name, SUM(miles) AS 'miles/driver'
FROM 
  cookbook.driver_log 
GROUP BY 
  name 
WITH ROLLUP;


SELECT name, AVG(miles) AS driver_avg 
FROM cookbook.driver_log
GROUP BY name WITH ROLLUP;


---------------------------------------
-- WITH ROLLUP produces multiple summary levels if you group by more than one column.
SELECT srcuser, dstuser, COUNT(*)
FROM cookbook.mail 
GROUP BY srcuser, dstuser
WITH ROLLUP;


---------------------------------------
-- to create a report that displays a summary, together with the list of rows associated with each summary value
-- Use two statements that retrieve different levels of summary information
-- A singel query cannot achieve this. You need to use Python or separate queries to achieve this.

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-- *SUMMARY*
-- You can use an alias or calculated value in GROUP BY.
-- You cannot use an alias in WHERE, but you can include the calculation in WHERE statement.
-- You cannot use GROUP BY groups in WHERE.
-- You can use GROUP BY groups in HAVING.
-- You cannot use the result of an aggregate function as input to another aggregate function.


