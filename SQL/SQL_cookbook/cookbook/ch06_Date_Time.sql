
-- CHOOSING THE RIGHT DATE/TIME DATA TYPE
-- Start by answering the following questions:
-- (a) Do you need times only, dates only, or combined date and time values?
-- (b) What range of values do you require?
-- (c) Do you want automatic initialization of the column to the current date and time?

-- Four major date / time data types in MySQL are:
-- (a) DATE values have CCYY-MM-DD format, where CC, YY, MM, and DD represent the century,
--     year within century, month, and day parts of the date. The supported range for DATE
--     values is 1000-01-01 to 9999-12-31.

-- (b) TIME values have hh:mm:ss format, where hh, mm, and ss are the hours, minutes,
--     and seconds parts of the time. TIME values often can be thought of as time-of-day
--     values, but MySQL actually treats them as elapsed time. Thus, they may be greater
--     than 23:59:59 or even negative. (The actual range of a TIME column is -838:59:59
--     to 838:59:59.)

-- (c,d) DATETIME and TIMESTAMP are combined date-and-time values in CCYY-MM-DD hh:mm:ss format.
--     DATETIME has a supported range of 1000-01-01 00:00:00 to 9999-12-31 23:59:59, 
--     whereas TIMESTAMP values are valid only from the year 1970 partially through 2038.

-- When a client inserts a TIMESTAMP value, the server converts it from the time
-- zone associated with the client session to UTC and stores the UTC value. When
-- the client retrieves a TIMESTAMP value, the server performs the reverse operation
-- to convert the UTC value back to the client session time zone.

SELECT * FROM cookbook.time_val;
SELECT * FROM cookbook.date_val;
SELECT * FROM cookbook.datetime_val;

SELECT CURTIME(), CURTIME(2), CURTIME(6);


-- You CANNOT change the ISO format that MySQL uses for representing date values.
-- The CCYY-MM-DD format that MySQL uses for DATE values follows the ISO 8601 standard
-- for representing dates.

-- MySQL always stores dates in ISO format, a fact with implications both
-- for data entry (input) and for displaying query results (output):
-- (a) For data-entry purposes, to store values that are not in ISO format, you normally
--     must rewrite them first. If you don’t want to rewrite them, you can store them as
--     strings (for example, in a CHAR column). But then you can’t operate on them as dates.
-- (b) For display purposes, you can rewrite dates to non-ISO formats. 
--     The DATE_FORMAT() function provides a lot of flexibility for 
--     changing date values into other formats.
--     You can also use functions such as YEAR() to extract parts of dates for display.

-- One way to rewrite non-ISO values for date entry is to use the STR_TO_DATE() function,
-- which takes a string representing a temporal value and a format string that specifies the
-- “syntax” of the value.

-- For example. to insert the value May 13, 2007 into a DATE column, do this:
SELECT STR_TO_DATE('May 13, 2007','%M %d, %Y');

-- For date display, MySQL uses ISO format (CCYY-MM-DD) unless you tell it otherwise. To
-- display dates or times in other formats, use the DATE_FORMAT() or TIME_FORMAT()
-- function to rewrite them.

-- The DATE_FORMAT() function takes two arguments: a DATE, DATETIME, or TIMESTAMP
-- value, and a string describing how to display the value.

SELECT d, DATE_FORMAT(d,'%M %d, %Y') FROM cookbook.date_val;

-- %Y Four-digit year
-- %y Two-digit year
-- %M Complete month name
-- %b Month name, initial three letters
-- %m Two-digit month of year (01..12)
-- %c Month of year (1..12)
-- %d Two-digit day of month (01..31)
-- %e Day of month (1..31)
-- %W Weekday name (Sunday..Saturday)
-- %r 12-hour time with AM or PM suffix
-- %T 24-hour time
-- %H Two-digit hour
-- %i Two-digit minute
-- %s Two-digit second
-- %% Literal %

SELECT 
  dt,
  DATE_FORMAT(dt,'%c/%e/%y %r') AS format1,
  DATE_FORMAT(dt,'%M %e, %Y %T') AS format2,
  DATE_FORMAT(dt,'%Y%m%d') AS format3
FROM 
  cookbook.datetime_val;


-- TIME_FORMAT() is similar to DATE_FORMAT(). It works with TIME, DATETIME, or TIME
-- STAMP values, but understands only time-related specifiers in the format string.
SELECT 
  dt,
  TIME_FORMAT(dt, '%r') AS '12-hour time',
  TIME_FORMAT(dt, '%T') AS '24-hour time'
FROM 
  cookbook.datetime_val;


---------------------------------------
-- SETTING TIME ZONES
-- When the MySQL server starts, it examines its operating environment to determine
-- its time zone. (To use a different value, start the server with the --default-timezone
-- option.)

-- For each client that connects, the server interprets TIMESTAMP values with respect
-- to the time zone associated with the client session. When a client inserts a TIMESTAMP value, 
-- the server converts it from the client time zone to UTC and stores the
-- UTC value.

DROP TABLE cookbook.t;
CREATE TABLE cookbook.t (ts TIMESTAMP);
INSERT INTO cookbook.t (ts) VALUES('2014-06-01 12:30:00');
SELECT * FROM cookbook.t;

SELECT @@global.time_zone, @@session.time_zone;


---------------------------------------
-- SHIFTING TEMPORAL VALUES BETWEEN TIME ZONES
-- The CONVERT_TZ() function converts temporal values between time zones.
-- It takes three arguments: a date-and-time value and two time zone indicators.
-- The first time zone is the source, and the second time zone is the destination TZ.

SET @dt = '2014-11-23 09:00:00';
SELECT CURTIME();

-- it requires that you have the time zone
-- tables in the mysql database initialized with support for named time zones.
SELECT 
  CURTIME() AS UTC,
  CONVERT_TZ(CURTIME(),'US/Central','Europe/Berlin') AS Berlin,
  CONVERT_TZ(CURTIME(),'US/Central','Europe/London') AS London,
  CONVERT_TZ(CURTIME(),'US/Central','America/Edmonton') AS Edmonton,
  CONVERT_TZ(CURTIME(),'US/Central','Australia/Brisbane') AS Brisbane;


---------------------------------------
-- DETERMINING THE CURRENT DATE/TIME
-- CURDATE(), CURTIME(), or NOW() functions give the current time in the client timezone.
-- UTC_DATE(), UTC_TIME(), or UTC_TIMESTAMP() give current time values in UTC time.
-- CURRENT_DATE, CURRENT_TIME, and CURRENT_TIMESTAMP are synonyms for CURDATE(), CURTIME(), and NOW(), respectively.

SELECT CURDATE(), CURTIME(), NOW();

SELECT UTC_DATE(), UTC_TIME(), UTC_TIMESTAMP();


---------------------------------------
-- USING TIMESTAMP OR DATETIME TO TRACK ROW-MODIFICATION TIMES
-- A TIMESTAMP or DATETIME column declared with the DEFAULT CURRENT_TIMESTAMP attribute 
-- initializes automatically for new rows. Simply omit the column from
-- INSERT statements and MySQL sets it to the row-creation time.
-- A TIMESTAMP or DATETIME column declared with the ON UPDATE CURRENT_TIME
-- STAMP attribute automatically updates to the current date and time when you change
-- any other column in the row from its current value.

CREATE TABLE cookbook.tsdemo
  (
    val INT,
    ts_both TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    ts_create TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ts_update TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
  );

INSERT INTO cookbook.tsdemo (val) VALUES(5);

INSERT INTO cookbook.tsdemo (val,ts_both,ts_create,ts_update)
  VALUES(10,NULL,NULL,NULL);

SELECT * FROM cookbook.tsdemo;

-- Note that by not setting the value of the timestamps in the query,
-- we make the SQL engine to use the default.

-- Also, you can set a TIMESTAMP column to the current date and time by setting it
-- explicitly to NULL, even one that does not auto-initialize.

UPDATE cookbook.tsdemo SET val = 11 WHERE val = 10;

UPDATE cookbook.tsdemo SET val = val + 1;

SELECT * FROM cookbook.tsdemo;

-- For the first TIMESTAMP column in a table, if neither of the DEFAULT or ON UPDATE
-- attributes are specified, the column is implicitly defined with both. For DATETIME,
-- qautomatic properties never apply implicitly; only those specified explicitly.

-- You can set a TIMESTAMP column to the current date and time at any time by setting
-- it to NULL, unless it has specifically been defined to permit NULL values. Assigning
-- NULL to a DATETIME column never sets it to the current date and time.

---------------------------------------
-- EXTRACT PARTS OF DATES OR TIMES

-------------------
-- (Option a) Decomposing dates or times using component-extraction functions

SELECT dt, DATE(dt), TIME(dt) 
FROM cookbook.datetime_val;


-- Common date / time extraction functions. The time-related functions work with TIME, DATETIME, or TIMESTAMP values.
-- YEAR() Year of date
-- MONTH() Month number (1..12)
-- MONTHNAME() Month name (January..December)
-- DAYOFMONTH() Day of month (1..31)
-- DAYNAME() Day name (Sunday..Saturday)
-- DAYOFWEEK() Day of week (1..7 for Sunday..Saturday)
-- WEEKDAY() Day of week (0..6 for Monday..Sunday)
-- DAYOFYEAR() Day of year (1..366)
-- HOUR() Hour of time (0..23)
-- MINUTE() Minute of time (0..59)
-- SECOND() Second of time (0..59)
-- EXTRACT() Varies


-- DAYOFWEEK() returns values from 1 to 7, corresponding to Sunday through Saturday. 
-- WEEKDAY() returns values from 0 to 6, corresponding to Monday through Sunday:
SELECT d, DAYNAME(d), LEFT(DAYNAME(d),3), DAYOFWEEK(d), WEEKDAY(d)
FROM cookbook.date_val;


-- The keyword indicating what to extract from the value should be a unit specifier such as
-- YEAR, MONTH, DAY, HOUR, MINUTE, or SECOND.
SELECT dt, EXTRACT(DAY FROM dt), EXTRACT(HOUR FROM dt)
FROM cookbook.datetime_val;

-------------------
-- (Option b) Decomposing dates or times using formatting functions
-- The DATE_FORMAT() and TIME_FORMAT() functions reformat date and time values. 
-- By specifying appropriate format strings, you can extract individual parts of temporal values.
SELECT 
  dt,
  DATE_FORMAT(dt,'%Y') AS year,
  DATE_FORMAT(dt,'%d') AS day,
  TIME_FORMAT(dt,'%H') AS hour,
  TIME_FORMAT(dt,'%s') AS second
FROM cookbook.datetime_val;


SELECT 
  dt,
  DATE_FORMAT(dt,'%Y-%m-%d') AS 'date part',
  TIME_FORMAT(dt,'%T') AS 'time part'
FROM cookbook.datetime_val;


SELECT 
dt,
DATE_FORMAT(dt,'%M %e, %Y') AS 'descriptive date',
TIME_FORMAT(dt,'%H:%i') AS 'hours/minutes'
FROM cookbook.datetime_val;


---------------------------------------
-- SYNTHESIZING A TEMPORAL VALUE FROM ITS CONSTITUENT PARTS
-- There are three main techniques for synthesizing data/time objects from their constituent parts:
-- (a) composition functions, (b) formatting functions, and (c) string concatenation


-- (a) USING COMPONENT FUNCTIONS
-- The MAKETIME() function takes component hour, minute, and second values as arguments
-- and combines them to produce a time.
SELECT MAKETIME(10,30,58), MAKETIME(5,0,11);


-- (b) USING FORMATTING FUNCTIONS
-- For example, to produce the first day of the month in which a date falls, 
-- use DATE_FORMAT() to extract the year and month parts
-- from the date, combining them with a day part of 01:
SELECT d, DATE_FORMAT(d,'%Y-%m-01') FROM cookbook.date_val;
SELECT t1, TIME_FORMAT(t1,'%H:%i:00') FROM cookbook.time_val;


-- (c) USING STRING CONCATENATION
-- use date-part extraction functions in conjunction with CONCAT()
SELECT d, CONCAT(YEAR(d),'-',MONTH(d),'-01') FROM cookbook.date_val;

-- To ensure that the month has two digits, as required for ISO format, 
-- use LPAD() to add a leading zero as necessary:
SELECT d, CONCAT(YEAR(d),'-',LPAD(MONTH(d),2,'0'),'-01')
FROM cookbook.date_val;

-- Note that even though the concatenation technically creates a string object,
-- you can use the result as a date object.
SELECT DAYOFMONTH(fdate.formatted_date)
FROM
(SELECT 
  d, 
  CONCAT(YEAR(d),'-',LPAD(MONTH(d),2,'0'),'-01') AS formatted_date
FROM cookbook.date_val) fdate;


---------------------------------------
-- CONVERTING BETWEEN TIME AND DATE VALUES AND BASIC UNITS
-- Depending on situation and goal, use one of the following functions:
-- To convert between time values and seconds, use the TIME_TO_SEC() and SEC_TO_TIME() functions.

-- The reference (zero) time for TIME_TO_SEC() is midnight (00:00:00)
-- Note: If you pass TIME_TO_SEC() a date-and-time value, it extracts the time part and discards the date.
SELECT 
  t1,
  TIME_TO_SEC(t1) AS 'TIME to seconds',
  SEC_TO_TIME(TIME_TO_SEC(t1)) AS 'TIME to seconds to TIME',
  TIME_TO_SEC(t1)/60 AS 'TIME to minutes',
  TIME_TO_SEC(t1)/3600 AS 'TIME to hours',
  TIME_TO_SEC(t1)/(24*60*60) AS 'TIME to days'
FROM 
  cookbook.time_val;

-------------------
-- To convert between date values and days, use the TO_DAYS() and FROM_DAYS() functions.
-- TO_DAYS() converts a date to the corresponding number of days, and FROM_DAYS() does the opposite:
-- If you pass TO_DAYS() a date-and-time value, it extracts the date part and discards the time.
-- The reference for T_DAYS() function is year 0.
SELECT 
  d,
  TO_DAYS(d) AS 'DATE to days',
  FROM_DAYS(TO_DAYS(d)) AS 'DATE to days to DATE'
FROM cookbook.date_val;

SELECT TO_DAYS('0000-01-01');

-------------------
-- To convert between date-and-time values and seconds, use the UNIX_TIMESTAMP() and FROM_UNIXTIME() functions.
-- For DATETIME or TIMESTAMP values that lie within the range of the TIMESTAMP data type
-- (from the beginning of 1970 partially through 2038), 
-- the UNIX_TIMESTAMP() and FROM_UNIXTIME() functions convert to and from the number of seconds elapsed since the beginning of 1970. 
-- The “Unix epoch” begins at 1970-01-01 00:00:00 UTC. 

-- The epoch is time zero, or the reference point for measuring time in Unix systems.
-- Note that UNIX_TIMESTAMP() assumes the datetime values provided are in local timezone, so it will convert them to UTC before counting the seconds from epoch.
SELECT 
  dt,
  UNIX_TIMESTAMP(dt) AS seconds,
  FROM_UNIXTIME(UNIX_TIMESTAMP(dt)) AS timestamp
FROM cookbook.datetime_val;


-- UNIX_TIMESTAMP() can also convert dates to seconds. If the time component is not provided,
-- UNIX_TIMESTAMP() assumes a time component of 00:00:00.
SELECT
  CURDATE(),
  UNIX_TIMESTAMP(CURDATE()),
  FROM_UNIXTIME(UNIX_TIMESTAMP(CURDATE()));
  

-- SUMMARY OF FUNCTIONS:
-- To convert between time values and seconds since midnight, use TIME_TO_SEC() and SEC_TO_TIME().
-- To convert between date values and days since year 0, use TO_DAYS() and FROM_DAYS().
-- To convert between date-and-time values and seconds since the epoch, use UNIX_TIMESTAMP() and FROM_UNIXTIME().


---------------------------------------
-- CALCULATING INTERVALS BETWEEN DATES OR TIMES
-- To calculate an interval in days between two date values, use the DATEDIFF() function.
SET @d1 = '2010-01-01', @d2 = '2009-12-01';

SELECT 
  DATEDIFF('2010-01-01', '2009-12-01') AS 'd1 - d2', 
  DATEDIFF('2009-12-01', '2010-01-01') AS 'd2 - d1';

-- DATEDIFF() also works with date-and-time values, but ignores the time part. 
-- This makes it suitable for producing day intervals for DATE, DATETIME, or TIMESTAMP values.

-- To calculate an interval between TIME values as another TIME value, use the TIMEDIFF() function.
-- Some SQL engines throw error for negative time difference. This is likely something that can be fixed in engine settings.
SELECT 
  TIMEDIFF('12:00:00', '16:30:00') AS 't1 - t2', 
  TIMEDIFF('16:30:00','12:00:00') AS 't2 - t1';


-- to express a time interval in terms of its constituent hours, minutes, and seconds values, 
-- calculate time interval subparts using the HOUR(), MINUTE(), and SECOND() functions.
SELECT 
  t1, t2,
  --TIMEDIFF(t2,t1) AS 't2 - t1 as TIME',
  IF(TIMEDIFF(t2,t1) >= 0,'+','-') AS sign,
  HOUR(TIMEDIFF(t2,t1)) AS hour,
  MINUTE(TIMEDIFF(t2,t1)) AS minute,
  SECOND(TIMEDIFF(t2,t1)) AS second
FROM cookbook.time_val;


-- If you work with date or date-and-time values, the TIMESTAMPDIFF() function provides
-- another way to calculate intervals.
-- It enables you to specify the units in which intervals should be expressed.
-- Permitted unit specifiers are MICROSECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.
SET @dt1 = '1900-01-01 00:00:00', @dt2 = '1910-01-01 00:00:00';
SELECT
  TIMESTAMPDIFF(MINUTE, '1900-01-01 00:00:00', '1910-01-01 00:00:00') AS minutes,
  TIMESTAMPDIFF(HOUR, '1900-01-01 00:00:00', '1910-01-01 00:00:00') AS hours,
  TIMESTAMPDIFF(DAY, '1900-01-01 00:00:00', '1910-01-01 00:00:00') AS days,
  TIMESTAMPDIFF(WEEK, '1900-01-01 00:00:00', '1910-01-01 00:00:00') AS weeks,
  TIMESTAMPDIFF(YEAR, '1900-01-01 00:00:00', '1910-01-01 00:00:00') AS years;


-- Using basic units
SELECT 
  t1, t2,
  TIME_TO_SEC(t2) - TIME_TO_SEC(t1) AS 't2 - t1 (in seconds)',
  SEC_TO_TIME(TIME_TO_SEC(t2) - TIME_TO_SEC(t1)) AS 't2 - t1 (as TIME)'
FROM cookbook.time_val;


---------------------------------------
-- ADDING DATE / TIME VALUES

-- Different options for adding date / time values
-- (a) Use one of the temporal-addition functions.
-- (b) Use the + INTERVAL or - INTERVAL operator.
-- (c) Convert the values to basic units, and take the sum.


-------------------
-- (a) Adding temporal values using temporal-addition functions or operators
-- Use the ADDTIME() function
SET @t1 = '12:00:00', @t2 = '15:30:00';
SELECT ADDTIME('12:00:00', '15:30:00');

-- To add a time to a date or date-and-time value, use the TIMESTAMP() function
SET @d = '1984-03-01', @t = '15:30:00';
SELECT TIMESTAMP('1984-03-01', '15:30:00');

SET @dt = '1984-03-01 12:00:00', @t = '12:00:00';
SELECT TIMESTAMP('1984-03-01 12:00:00', '12:00:00');


-------------------
-- (b) Use the + INTERVAL or - INTERVAL operator.
-- MySQL also provides DATE_ADD() and DATE_SUB() functions for 
-- adding intervals to dates and subtracting intervals from dates.
-- DATE_ADD(d, INTERVAL val unit)
-- DATE_SUB(d, INTERVAL val unit)

-- The + INTERVAL and - INTERVAL operators are similar:
-- d + INTERVAL val unit
-- d - INTERVAL val unit
-- common unit specifiers are SECOND, MINUTE, HOUR, DAY, MONTH, and YEAR.

SELECT CURDATE(), DATE_ADD(CURDATE(), INTERVAL 3 DAY);
SELECT CURDATE(), DATE_SUB(CURDATE(), INTERVAL 1 WEEK);

-- For questions where you need to know both the date and the time, 
-- begin with a DATETIME or TIMESTAMP value
SELECT NOW(), DATE_ADD(NOW(), INTERVAL 60 HOUR);

SELECT NOW(), DATE_ADD(NOW(), INTERVAL '14:30' HOUR_MINUTE);
SELECT NOW(), DATE_ADD(NOW(), INTERVAL '3 4' DAY_HOUR);


-- DATE_ADD() and DATE_SUB() are interchangeable because one is the same as the other
-- with the sign of the interval value flipped.
-- E.g., the following two expressions have the same results.
DATE_ADD(d, INTERVAL -3 MONTH)
DATE_SUB(d, INTERVAL 3 MONTH)


-- Use the + INTERVAL or - INTERVAL operator to perform date interval addition or subtraction:
SELECT CURDATE(), CURDATE() + INTERVAL 1 YEAR;
SELECT NOW(), NOW() - INTERVAL '1 12' DAY_HOUR;


-- TIMESTAMPADD() is an alternative function for adding intervals to date or date-and-time values. 
-- Its arguments are similar to those for DATE_ADD(), and the following equivalence holds:
-- TIMESTAMPADD(unit,interval,d) = DATE_ADD(d,INTERVAL interval unit)


-------------------
-- (c) Convert the values to basic units, and take the sum.
SELECT 
  t1,
  SEC_TO_TIME(TIME_TO_SEC(t1) + 7200) AS 't1 plus 2 hours'
FROM 
  cookbook.time_val;

SELECT 
  t1,
  SEC_TO_TIME(TIME_TO_SEC(t1) + TIME_TO_SEC(t1)) AS 't1 plus 2 hours'
FROM 
  cookbook.time_val;

SELECT 
  t1, t2,
  TIME_TO_SEC(t1) + TIME_TO_SEC(t2) AS 't1 + t2 (in seconds)',
  SEC_TO_TIME(TIME_TO_SEC(t1) + TIME_TO_SEC(t2)) AS 't1 + t2 (as TIME)'
FROM 
  cookbook.time_val;
  
-- Note that MySQL TIME values represent elapsed time, 
-- not time of day, so they don’t reset to 0 after reaching 24 hours.
-- To produce time-of-day values, enforce a 24- hour wraparound 
-- using a modulo operation before converting the seconds value back to a TIME value.
-- To calculate he modulo (all three expressions below are equivalent)
-- MOD(s,86400)
-- s % 86400
-- s MOD 86400

SELECT t1, t2,
  MOD(TIME_TO_SEC(t1) + TIME_TO_SEC(t2), 86400)
  AS 't1 + t2 (in seconds)',
  SEC_TO_TIME(MOD(TIME_TO_SEC(t1) + TIME_TO_SEC(t2), 86400))
  AS 't1 + t2 (as TIME)'
FROM cookbook.time_val;


-- Adding to date and data/time values
SET @d = '1980-01-01';
SELECT 
  @d AS date,
  FROM_DAYS(TO_DAYS(@d) + 7) AS 'date + 1 week',
  FROM_DAYS(TO_DAYS(@d) - 7) AS 'date - 1 week';
  
-- To preserve the time, you can use UNIX_TIMESTAMP() and FROM_UNIXTIME() instead, if
-- the initial and resulting values both lie in the permitted range for TIMESTAMP values.
SET @dt = '1980-01-01 09:00:00';
SELECT @dt AS datetime,
FROM_UNIXTIME(UNIX_TIMESTAMP(@dt) + 3600) AS 'datetime + 1 hour',
FROM_UNIXTIME(UNIX_TIMESTAMP(@dt) - 3600) AS 'datetime - 1 hour';


---------------------------------------
-- CALCULATING AGE DIFFERENC
-- If we want to know the difference between two dates in a certain unit of time (e.g., years):
-- To calculate ages, use the TIMESTAMPDIFF() function. Pass it a birth date, a current date,
-- and the unit in which you want the time difference expressed:
-- TIMESTAMPDIFF(unit, first_date, second_date)

SELECT * FROM cookbook.sibling;

SELECT 
  name, birth, CURDATE() AS today,
  TIMESTAMPDIFF(YEAR, birth, CURDATE()) AS 'age in years',
  TIMESTAMPDIFF(MONTH, birth, CURDATE()) AS 'age in months'
FROM cookbook.sibling;

-- How old were Gretchen and Wilbur when Franz was born?
SELECT 
  name, birth, '1953-03-05' AS 'Franz'' birth',
  TIMESTAMPDIFF(YEAR, birth,'1953-03-05') AS 'age in years',
  TIMESTAMPDIFF(MONTH, birth,'1953-03-05') AS 'age in months'
FROM 
  cookbook.sibling 
WHERE 
  name <> 'Franz';


-- How old were sibling when another sibling was born?
SELECT 
  s1.name, s1.birth, s2.name, s2.birth, 
  TIMESTAMPDIFF(YEAR, s1.birth, s2.birth) AS 'age in years'
FROM 
  cookbook.sibling s1,
  cookbook.sibling s2
WHERE 
  s1.birth < s2.birth;


---------------------------------------
-- FIND FIRST DAY OF MONTH FOR A GIVEN DATE
-- To find the first day of the month for a given date, shift the date back by one fewer days
-- than its DAYOFMONTH() value.
-- To find the first day of month n months from now, add n months to DAYOFMONTH() of the given date.
SELECT 
  d, 
  DATE_ADD(DATE_SUB(d,INTERVAL DAYOFMONTH(d)-1 DAY),INTERVAL -1 MONTH) AS '1st of previous month',
  DATE_SUB(d, INTERVAL DAYOFMONTH(d)-1 DAY) AS '1st of month',
  DATE_ADD(DATE_SUB(d,INTERVAL DAYOFMONTH(d)-1 DAY),INTERVAL 3 MONTH)  AS '1st of 3 months from now'
FROM 
  cookbook.date_val;

-- To find the last day of month for a given date, use the function LAST_DAY():
SELECT d, LAST_DAY(d) AS 'last of month'
FROM cookbook.date_val;

SELECT 
  d,
  LAST_DAY(DATE_ADD(d,INTERVAL -1 MONTH)) AS 'last of previous month',
  LAST_DAY(DATE_ADD(d,INTERVAL 1 MONTH)) AS 'last of following month'
FROM 
  cookbook.date_val;

-- To find the length of a month in days
SELECT d, DAYOFMONTH(LAST_DAY(d)) AS 'days in month' FROM cookbook.date_val;
---------------------------------------


---------------------------------------
-- CALCULATING DATES BY SUBSTRING REPLACEMENT
-- Given a date, you want to produce another date from it when you know that the two
-- dates share some components in common.
-- Solution: use DATE_FORMAT(), CONCAT(), and other string manipulation functions to 
-- directly modify parts of the date string.
SELECT 
  d,
  DATE_FORMAT(d, '%Y-%m-01') AS '1st of month A',
  CONCAT(YEAR(d), '-', LPAD(MONTH(d),2,'0'), '-01') AS '1st of month B'
FROM 
  cookbook.date_val;
---------------------------------------


---------------------------------------
-- FINDING THE DAY OF THE WEEK FOR A DATE
-- use DAYNAME() function
SELECT CURDATE(), DAYNAME(CURDATE());

-- FINDING DATES FOR ANY WEEKDAY OF A GIVEN WEEK
-- Use DAYOFWEEK() and WEEKDAY() functions.
-- DAYOFWEEK() treats Sunday as the first day of the week and returns 1 through 7 for Sunday through Saturday.
-- WEEKDAY() treats Monday as the first day of the week and returns 0 through 6 for Monday through Sunday.


-- This could be done by a combination of the following operations:
-- 1. Shift the reference date back by its DAYOFWEEK() value, which always produces the
-- date for the Saturday preceding the week.
-- 2. Shift the Saturday date by one day to reach the Sunday date, by two days to reach
-- the Monday date, and so forth.

-- n is 1 through 7 to produce the dates for Sunday through Saturday:
-- DATE_ADD(DATE_SUB(d, INTERVAL DAYOFWEEK(d) DAY), INTERVAL n DAY)
-- or, put in one single command:
-- DATE_ADD(d, INTERVAL n-DAYOFWEEK(d) DAY)

SELECT d, DAYNAME(d) AS day,
  DATE_ADD(d,INTERVAL 1-DAYOFWEEK(d) DAY) AS Sunday,
  DATE_ADD(d,INTERVAL 7-DAYOFWEEK(d) DAY) AS Saturday
FROM 
  cookbook.date_val;

---------------------------------------

---------------------------------------
-- PERFORMING LEAP YEAR CALCS
-- We need to how to test whether a year is a leap year.
-- For a year to qualify as a leap year, it must satisfy both of these constraints:
-- The year must be divisible by four.
-- The year cannot be divisible by 100, unless it is also divisible by 400. In SQL:
-- (YEAR(d) % 4 = 0) AND ((YEAR(d) % 100 <> 0) OR (YEAR(d) % 400 = 0))

-- To compute a year’s length in SQL, compute the date of the last day of the year and pass it to DAYOFYEAR()
SELECT
  DAYOFYEAR(DATE_FORMAT('2014-04-13','%Y-12-31')) AS 'days in 2014',
  DAYOFYEAR(DATE_FORMAT('2016-04-13','%Y-12-31')) AS 'days in 2016';
---------------------------------------

---------------------------------------
-- CANONIZING NON-ISO DATE STRINGS
-- One way to standardize a close-to-ISO date is to use it in an expression that produces an ISO date result
-- Examples of such operations:
-- DATE_ADD(d,INTERVAL 0 DAY)
-- d + INTERVAL 0 DAY
-- FROM_DAYS(TO_DAYS(d))
-- STR_TO_DATE(d,'%Y-%m-%d')

SELECT
  CONCAT(YEAR(d),'-',MONTH(d),'-01') AS 'non-ISO',
  DATE_ADD(CONCAT(YEAR(d),'-',MONTH(d),'-01'),INTERVAL 0 DAY) AS 'ISO 1',
  CONCAT(YEAR(d),'-',MONTH(d),'-01') + INTERVAL 0 DAY AS 'ISO 2',
  FROM_DAYS(TO_DAYS(CONCAT(YEAR(d),'-',MONTH(d),'-01'))) AS 'ISO 3',
  STR_TO_DATE(CONCAT(YEAR(d),'-',MONTH(d),'-01'),'%Y-%m-%d') AS 'ISO 4'
FROM 
  cookbook.date_val;

---------------------------------------
-- DATE-BASED RESTRICTIONS
-- Comparing dates to one another
SELECT d FROM cookbook.date_val where d < '1900-01-01';

-- to find dates that occur later than 20 years ago, use DATE_SUB() to calculate the cutoff date:
SELECT d FROM cookbook.date_val 
WHERE d >= DATE_SUB(CURDATE(),INTERVAL 20 YEAR);

-- Optimzing the query above:
-- Note that the expression in the WHERE clause isolates the date column d on one side of
-- the comparison operator. This is usually a good idea; if the column is indexed, placing
-- it alone on one side of a comparison enables MySQL to process the statement more
-- efficiently. To illustrate, the preceding WHERE clause can be written in a way that’s logically
-- equivalent but much less efficient for MySQL to execute.
-- WHERE DATE_ADD(d,INTERVAL 20 YEAR) >= CURDATE();

-- Another example of how to isolate a date value to make use of its index:
-- WHERE YEAR(d) >= 1987 AND YEAR(d) <= 1991;
-- rewrite as:
-- WHERE d >= '1987-01-01' AND d < '1992-01-01';


-- To delete expired rows:
-- DELETE FROM mytbl WHERE create_date < DATE_SUB(NOW(),INTERVAL n DAY);

-- Comparings times:
-- For an indexed TIME column, the first method is more efficient. The second method has
-- the property that it works not only for TIME columns, but for DATETIME and TIME
-- STAMP columns as well.
--   WHERE t1 BETWEEN '09:00:00' AND '14:00:00';
--   WHERE HOUR(t1) BETWEEN 9 AND 14;

-- To find out 'who has a birthday today?'
-- WHERE MONTH(d) = MONTH(CURDATE()) AND DAYOFMONTH(d) = DAYOFMONTH(CURDATE());
-- NOTE: do not use DAYOFYEAR() as it will have wrong values for leap years.

-- Who has a birthday this month?
-- WHERE MONTH(d) = MONTH(CURDATE());

-- Who has a birthday next month?
-- The tricky part is if we are in December.
-- WHERE MONTH(d) = MONTH(DATE_ADD(CURDATE(), INTERVAL 1 MONTH));
-- WHERE MONTH(d) = MOD(MONTH(CURDATE()), 12)+1;



---------------------------------------
---------------------------------------
-- SUMMARY: the following functions can be used to handle dates:

-- To get current time (in local machine's TZ), use:
CURDATE(), CURTIME(), or NOW()
UTC_DATE(), UTC_TIME(), UTC_TIMESTAMP()
CURRENT_DATE, CURRENT_TIME, and CURRENT_TIMESTAMP are synonyms for CURDATE(), CURTIME(), and NOW(), respectively.


DATE_FORMAT()  -- changes the way that a date object is displayed
TIME_FORMAT()  -- changes the way that a time object is displayed
  -- %Y Four-digit year
  -- %y Two-digit year
  -- %M Complete month name
  -- %b Month name, initial three letters
  -- %m Two-digit month of year (01..12)
  -- %c Month of year (1..12)
  -- %d Two-digit day of month (01..31)
  -- %e Day of month (1..31)
  -- %W Weekday name (Sunday..Saturday)
  -- %r 12-hour time with AM or PM suffix
  -- %T 24-hour time
  -- %H Two-digit hour
  -- %i Two-digit minute
  -- %s Two-digit second
  -- %% Literal %


-- To convert between strings and datetime objects, use MAKETIME, and TIMESTAMP().
-- To format the strings before/after converting to/from dates, use CONCAT() and LPAD() functions.
MAKETIME()  -- takes component hour, minute, and second values as arguments
CONCAT()  -- string concatenation (generic string function. not specific to date)
LPAD()  -- string padding (generic string function. not specific to date)
TIMESTAMP()  -- converts string to datetime object


-- To convert between time values and seconds since midnight, use TIME_TO_SEC() and SEC_TO_TIME().
-- To convert between date values and days since year 0, use TO_DAYS() and FROM_DAYS().
-- To convert between date-and-time values and seconds since the epoch, use UNIX_TIMESTAMP() and FROM_UNIXTIME().
TIME_TO_SEC()  -- converts time object to seconds since midnight (00:00:00).
SEC_TO_TIME()  -- converts the numeric value of seconds since midnight (00:00:00) to time object.
TO_DAYS()  -- converts a date to the corresponding number of days since year 0 (0000-01-01).
FROM_DAYS()  -- converts number of days from year 0 (0000-01-01) to the corresponding date.
UNIX_TIMESTAMP()  -- convert to the number of seconds elapsed since UNIX epoch (1970-01-01).
FROM_UNIXTIME()  -- convert from the number of seconds elapsed since the beginning of (1970-01-01).


CONVERT_TZ()  -- To convert from one timezone to another. Need to load timezones first: http://stackoverflow.com/questions/14454304/convert-tz-returns-null


DATEDIFF()  -- To calculate an interval in days between two date values
TIMEDIFF()  -- To calculate an interval in seconds between two datetime values
TIMESTAMPDIFF()  -- To calculate the difference between two datetime or timestamp data types. 
-- Example: TIMESTAMPDIFF(MINUTE, '1900-01-01 00:00:00', '1910-01-01 00:00:00')
-- Accepted time units: MICROSECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.


-- To add or substract time
ADDTIME()  -- adds time to a date or time object.
TIMESTAMP()  -- the time argument can be added to the datetime argument. Example: TIMESTAMP('1984-03-01 12:00:00', '12:00:00')
DATE_ADD()  -- Adds date with a specific unit to the date object. DATE_ADD(d, INTERVAL val unit). Common unit specifiers are SECOND, MINUTE, HOUR, DAY, MONTH, and YEAR.
DATE_SUB()  -- same as DATE_ADD() but for subtraction.


-- To jump to the end of a month, use LAST_DAY()
-- To jump to any other day in a given month or week, see the recipes above.
LAST_DAY()  -- returns the date for last day of month for a given date.


-- To extract parts of a datetime object, use the following functions:
YEAR() Year of date
MONTH() Month number (1..12)
MONTHNAME() Month name (January..December)
DAYOFMONTH() Day of month (1..31)
DAYNAME() Day name (Sunday..Saturday)
DAYOFWEEK() Day of week (1..7 for Sunday..Saturday)
WEEKDAY() Day of week (0..6 for Monday..Sunday)
DAYOFYEAR() Day of year (1..366)
HOUR() Hour of time (0..23)
MINUTE() Minute of time (0..59)
SECOND() Second of time (0..59)
EXTRACT() Varies
---------------------------------------
---------------------------------------

---------------------------------------
-- USAGE EXAMPLES
TO_CHAR(b.audit_end_date, 'YYYY-MM-DD')
to_char(trunc(sysdate),'YYYYMMDD')
TO_CHAR(TO_TIMESTAMP(CAST(max_repgen_reported_date AS VARCHAR(8)), 'YYYYMMDD'),'YYYY-MM-DD')
CAST(EXTRACT(DAY FROM trunc(sysdate)-TO_TIMESTAMP(CAST(max_repgen_reported_date AS VARCHAR(8)), 'YYYYMMDD'))-1 AS VARCHAR(20))
TO_CHAR(TO_TIMESTAMP(CAST(last_nightime_date AS VARCHAR(8)), 'YYYYMMDD'),'YYYY-MM-DD')
TO_CHAR(MAX(AUDIT_END_DATE), '{unix_format}')
TO_DATE(TO_CHAR(TO_TIMESTAMP('{max_date_limit}', 'YYYY-MM-DD HH24:MI:SS')-{minDaysFromPTO}, 'YYYYMMDD'), 'YYYYMMDD')
TO_NUMBER(TO_CHAR(TO_TIMESTAMP('{max_date_limit}', 'YYYY-MM-DD HH24:MI:SS'), 'YYYYMMDD'))
CAST(SYSDATE-0.5 as date)
TO_DATE(TO_CHAR(d.min_active_date), 'YYYYMMDD')
TO_CHAR(TO_DATE(TO_CHAR(fsd_info.case_create_date_fkey),'YYYYMMDD'),'YYYY-MM-DD')





