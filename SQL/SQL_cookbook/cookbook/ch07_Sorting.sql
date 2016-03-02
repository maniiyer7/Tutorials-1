
-- Rows do tend to be returned from a table in the order in which they were originally inserted, 
-- but only until the table is subjected to delete and update operations. 
-- Rows inserted after that are likely to be returned in the middle of the result set somewhere.
-- In effect, the retrieval order changes over time as you modify the table contents.

-- Note that Multiple-column sorts can be descending as well, but DESC must be specified after each
-- column name to perform a fully descending sort.

-- You can use original column names, aliases, or an expression in the ORDER BY statement.

-- Use expression:
SELECT 
  t, srcuser, FLOOR((size+1023)/1024)
FROM 
  cookbook.mail 
WHERE 
  size > 50000
ORDER BY 
  FLOOR((size+1023)/1024);

-- Use alias:
SELECT 
  t, srcuser, FLOOR((size+1023)/1024) AS kilobytes
FROM 
  cookbook.mail 
WHERE 
  size > 50000
ORDER BY 
  kilobytes;


-- display the string values and use the numeric values for sorting.
-- add zero to the jersey_num values to force a string-to-number conversion:

SELECT name, jersey_num FROM cookbook.roster ORDER BY jersey_num+0;


-- 
SELECT 
  t, CONCAT(srcuser,'@',srchost) AS sender, size
FROM 
  cookbook.mail 
WHERE 
  size > 50000
ORDER BY 
  srchost, srcuser;


---------------------------------------
-- CONTROLING CASE SENSITIVITY WHEN COMPARING STRINGS

-- We have two types of strings in SQL: binary and non-binary
-- (a) Binary strings are sequences of bytes. They are compared byte by byte using numeric
--     byte values. Character set and lettercase have no meaning for comparisons.
-- (b) Nonbinary strings are sequences of characters. They have a character set and collation
--     and are compared character by character using the order defined by the collation.

-- These properties also apply to string sorting because sorting is based on comparison.
-- To alter the sorting properties of a string column, alter its comparison properties.

-- The table below has case-insensitive and case-sensitive nonbinary columns, and a binary column:
CREATE TABLE cookbook.str_val
  (
  ci_str CHAR(3) CHARACTER SET latin1 COLLATE latin1_swedish_ci,
  cs_str CHAR(3) CHARACTER SET latin1 COLLATE latin1_general_cs,
  bin_str BINARY(3)
  );

-- To add values, use the Terminal and the script provided by the book.
DROP TABLE cookbook.str_val;

SELECT * FROM cookbook.str_val;


-- In the table, Each column contains the same values, 
-- but the natural sort orders for the column data types produce three different results:
SELECT ci_str FROM cookbook.str_val ORDER BY ci_str;

-- The case-sensitive collation puts A and a before B and b, and sorts uppercase before lowercase:
SELECT cs_str FROM cookbook.str_val ORDER BY cs_str;

-- The binary strings sort numerically. Assuming that uppercase letters have numeric
-- values less than those of lowercase letters, a binary sort results in the following ordering:
SELECT * FROM cookbook.str_val ORDER BY bin_str;


-- To sort case-insensitive strings in case-sensitive fashion, 
-- order the sorted values using a case-sensitive collation:
SELECT ci_str 
FROM cookbook.str_val
ORDER BY ci_str COLLATE latin1_general_cs;

-- To sort case-sensitive strings in case-insensitive fashion, 
-- order the sorted values using a case-insensitive collation:
SELECT cs_str 
FROM cookbook.str_val
ORDER BY cs_str COLLATE latin1_swedish_ci;

-- Alternatively, sort using values that have been converted to the same lettercase,
-- which makes lettercase irrelevant:
SELECT cs_str 
FROM cookbook.str_val
ORDER BY UPPER(cs_str);


-- Binary strings sort using numeric byte values, so there is no concept of lettercase involved.
-- However, because letters in different cases have different byte values,
-- comparisons of binary strings effectively are case sensitive.
-- To sort binary strings using a case-insensitive ordering, 
-- convert them to nonbinary strings and apply an appropriate collation
SELECT bin_str 
FROM cookbook.str_val
ORDER BY CONVERT(bin_str USING latin1) COLLATE latin1_swedish_ci;


---------------------------------------
-- COMPARING DATES
-- We have different data types that represent time: DATE, TIME, DATETIME, TIMESTAMP
SELECT * FROM cookbook.mail WHERE srcuser = 'phil';

-- Sort by time
SELECT * FROM cookbook.mail WHERE srcuser = 'phil' ORDER BY t DESC;


-- SORT BY TIME OF DAY
-- If the values are stored in a TIME column named timecol, 
-- just sort them directly using ORDER BY timecol.
-- To put DATETIME or TIMESTAMP values in time-of-day order, 
-- extract the time parts and sort them.
SELECT * FROM cookbook.mail ORDER BY TIME(t);


-- SORT BY CALENDAR DAY
-- To sort date values in calendar order, ignore the year part of the dates and use only the
-- month and day to order values by where they fall during the calendar year.
SELECT date, description FROM cookbook.occasion ORDER BY date;

-- To put these items in calendar order, sort them by month and day within month:
SELECT date, description FROM cookbook.occasion
ORDER BY MONTH(date), DAYOFMONTH(date);



-- SORT BY DAY OF WEEK
-- The function DAYNAME() produces strings that sort lexically rather than in day-of-week order.
-- The function DAYOFWEEK() returns numeric values from 1 to 7 for Sunday through Saturday.
-- We can use DAYOFWEEK() to order, and DAYNAME() to display names of the days of week.
SELECT DAYNAME(date) AS day, date, description
FROM cookbook.occasion
ORDER BY DAYOFWEEK(date);

-- To sort rows in day-of-week order but treat Monday as the first day of the week and
-- Sunday as the last, use the MOD() function to map Monday to 0, Tuesday to 1, …, Sunday to 6:
SELECT DAYNAME(date), date, description
FROM cookbook.occasion
ORDER BY MOD(DAYOFWEEK(date)+5, 7);


---------------------------------------
-- SORTING BY SUBSTRINGS OF COLUMN VALUES
-- If you know the specific location of the substring within the larger string,
-- pull out the parts you need with LEFT(), MID(), or RIGHT(), and sort them.

SELECT * FROM cookbook.housewares ORDER BY id;


-- Those fixed-length substrings of the id values can be used for sorting, either alone or in combination.
SELECT 
  id,
  LEFT(id,3) AS category,
  MID(id,4,5) AS serial,
  RIGHT(id,2) AS country
FROM 
  cookbook.housewares;

SELECT * FROM cookbook.housewares ORDER BY LEFT(id,3);


-- To sort by product serial number, use MID() to extract the middle five characters 
-- from the id values, beginning with the fourth.
SELECT * FROM cookbook.housewares ORDER BY MID(id,4,5);
-- This appears to be a numeric sort, but it’s actually a string sort because MID() returns strings.


---------------------------------------
-- SORTING BY VARIABLE_LENGTH SUBSTRINGS

-- use SUBSTRING() to skip the first three characters,
-- 
SELECT 
  id, 
  SUBSTRING(id,4),
  CHAR_LENGTH(SUBSTRING(id,4)-2),
  LEFT(SUBSTRING(id,4), CHAR_LENGTH(SUBSTRING(id,4)-2))
FROM 
  cookbook.housewares3;


SELECT 
  id, 
  SUBSTRING(id,4), 
  SUBSTRING(id,4,CHAR_LENGTH(id)-5)
FROM 
  cookbook.housewares3;

-- Even though the above query corectly selects the numeric part of the ID,
-- it does not order them correctly, because the SUBSTRING() function returns a string, not a numeric value.
-- To convert the results of the SUBSTRING() to numeric so that we can sort them properly,
-- we can add a 0 to the results to make SQL conver the results to numeric values.
SELECT * 
FROM 
  cookbook.housewares3
ORDER BY 
  SUBSTRING(id,4,CHAR_LENGTH(id)-5);

SELECT * 
FROM 
  cookbook.housewares3
ORDER BY 
  SUBSTRING(id,4,CHAR_LENGTH(id)-5)+0;

-- To simplify the query even further, note that a string to numeric converstion will by default remove 
-- the trailing character values. 

SELECT * FROM cookbook.housewares3
ORDER BY SUBSTRING(id,4)+0;


SELECT * FROM cookbook.housewares4;

-- To extract segments from these values, use SUBSTRING_INDEX(str,c,n). It searches a
-- string str for the n-th occurrence of a given character c and returns everything to 
-- the left of that character.
-- If n is negative, the search for c proceeds from the right and returns the rightmost string.

-- In the table above, by varying n from 1 to 4, we get the successive segments from left to right:

SELECT
  id, 
  SUBSTRING_INDEX(id,'-',1),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',1),'-',-1),
  SUBSTRING_INDEX(id,'-',2),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',2),'-',-1),
  SUBSTRING_INDEX(id,'-',3),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',3),'-',-1),
  SUBSTRING_INDEX(id,'-',4),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',4),'-',-1)
FROM 
  cookbook.housewares4;

-- Another way to obtain substrings is to extract the rightmost n segments of the value and pull off the first one.
SELECT
  id,
  SUBSTRING_INDEX(id,'-',-4),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',-4),'-',1),
  SUBSTRING_INDEX(id,'-',-3),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',-3),'-',1),
  SUBSTRING_INDEX(id,'-',-2),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',-2),'-',1),
  SUBSTRING_INDEX(id,'-',-1),
  SUBSTRING_INDEX(SUBSTRING_INDEX(id,'-',-1),'-',1)
FROM 
  cookbook.housewares4;


---------------------------------------
-- SORTING HOSTNAMES IN DOMAIN ORDER
-- to sort hostnames in domain order, where the rightmost segments of the
-- hostname values are more significant than the leftmost segments.
SELECT name FROM cookbook.hostname ORDER BY name;


-- The hostname values have a maximum of three segments, 
-- from which the pieces can be extracted left to right like this:

SELECT 
  name,
  SUBSTRING_INDEX(SUBSTRING_INDEX(name,'.',-3),'.',1) AS leftmost,
  SUBSTRING_INDEX(SUBSTRING_INDEX(name,'.',-2),'.',1) AS middle,
  SUBSTRING_INDEX(name,'.',-1) AS rightmost
FROM cookbook.hostname;

-- Notice the output for the mysql.com row; it has mysql for the value of the leftmost
-- column, where it should have an empty string. The segment-extraction expressions
-- work by pulling off the rightmost n segments, and then returning the leftmost segment
-- of the result. The source of the problem for mysql.com is that if there aren’t n segments,
-- the expression simply returns the leftmost segment of however many there are.

-- To fix this problem, add a sufficient number of periods at the beginning of the hostname values
-- to guarantee that they have the requisite number of segments:
SELECT 
  name,
  SUBSTRING_INDEX(SUBSTRING_INDEX(CONCAT('..',name),'.',-3),'.',1) AS leftmost,
  SUBSTRING_INDEX(SUBSTRING_INDEX(CONCAT('.',name),'.',-2),'.',1) AS middle,
  SUBSTRING_INDEX(name,'.',-1) AS rightmost
FROM 
  cookbook.hostname;

-- the expressions do serve to extract the substrings that are needed
-- for sorting hostname values correctly in right-to-left fashion:
SELECT 
  name 
FROM 
  cookbook.hostname
ORDER BY
  SUBSTRING_INDEX(name,'.',-1),
  SUBSTRING_INDEX(SUBSTRING_INDEX(CONCAT('.',name),'.',-2),'.',1),
  SUBSTRING_INDEX(SUBSTRING_INDEX(CONCAT('..',name),'.',-3),'.',1);


---------------------------------------
-- SORT IN NUMERIC ORDER STRINGS THAT REPRESENT IP NUMBERS
-- to sort in numeric order strings that represent IP numbers
-- Break apart the strings, and sort the pieces numerically. Or just use INET_ATON(). 
-- Or consider storing the values as numbers instead

SELECT ip FROM cookbook.hostip ORDER BY ip;

SELECT 
  ip 
FROM 
  cookbook.hostip
ORDER BY
  SUBSTRING_INDEX(ip,'.',1)+0,
  SUBSTRING_INDEX(SUBSTRING_INDEX(ip,'.',-3),'.',1)+0,
  SUBSTRING_INDEX(SUBSTRING_INDEX(ip,'.',-2),'.',1)+0,
  SUBSTRING_INDEX(ip,'.',-1)+0;


-- A simpler solution uses the INET_ATON() function to convert network addresses in string
-- form to their underlying numeric values, then sorts those numbers:
SELECT 
  ip,
  INET_ATON(ip)
FROM 
  cookbook.hostip 
ORDER BY INET_ATON(ip);


---------------------------------------
-- FLOATING VALUES TO THE HEAD OR TAIL OF THE SORT ORDER

-- You want a column to sort the way it normally does, 
-- except for a few values that should appear at the beginning or end of the sort order.
-- For example, you want to sort a list in
-- lexical order except for certain high-priority values that should appear first no matter
-- where they fall in the normal sort order.

-- Add an initial sort column to the ORDER BY clause that places those few values where you want them.

-- To sort a result set normally except that you want particular values first, 
-- create an additional sort column that is 0 for those values and 1 for everything else.


-- For example, to put the mails sent by Phil at the very top of the list:
-- The value of the extra sort column is 0 for rows in which the srcuser value is phil, and 1 
-- for all other rows. By making that the most significant sort column, rows for messages
-- sent by phil float to the top of the output.
SELECT 
  t, srcuser, dstuser, size
FROM 
  cookbook.mail
ORDER BY 
  IF(srcuser='phil',0,1), srcuser, dstuser;


-- To put first those rows where people sent messages to themselves: 
SELECT 
  t, srcuser, dstuser, size
FROM 
  cookbook.mail
ORDER BY 
  IF(srcuser=dstuser,0,1), srcuser, dstuser;


---------------------------------------
-- DEFINING A CUSTOM SORT ORDER
-- to sort values in a nonstandard order,
-- use FIELD() to map column values to a sequence that places the values in the desired order.

-- use the FIELD() function to map the values of a column to a list of numeric values 
-- and use the numbers for sorting. 
-- FIELD() compares its first argument to the following arguments and 
-- returns an integer indicating which one it matches.

-- The following FIELD() call compares value to str1, str2, str3, and str4, and 
-- returns 1, 2, 3, or 4, depending on which of them value is equal to:
-- If value is NULL or none of the values match, FIELD() returns 0.
--   FIELD(value,str1,str2,str3,str4)

-- For example, to order drivers by an arbitrary order of their names:
SELECT * FROM cookbook.driver_log
ORDER BY FIELD(name,'Henry','Suzi','Ben');


---------------------------------------
-- SORTING ENUM VALUES
-- ENUM values don’t sort like other string columns.
-- ENUM is a string data type, but ENUM values actually are stored numerically with values
-- ordered the same way they are listed in the table definition.

CREATE TABLE cookbook.weekday
  (
  day ENUM('Sunday','Monday','Tuesday','Wednesday',
  'Thursday','Friday','Saturday')
  );

INSERT INTO cookbook.weekday (day) 
VALUES('Monday'),('Friday'), ('Tuesday'), ('Sunday'), ('Thursday'), ('Saturday'), ('Wednesday');

-- Internally, MySQL defines the enumeration values Sunday through Saturday in that
-- definition to have numeric values from 1 to 7.
SELECT day, day+0 FROM cookbook.weekday;


-- When you want to sort ENUM values in lexical order? Force them
-- to be treated as strings for sorting using the CAST() function
SELECT day, day+0 FROM cookbook.weekday ORDER BY CAST(day AS CHAR);


CREATE TABLE cookbook.color (name CHAR(10));
INSERT INTO cookbook.color (name) VALUES ('blue'),('green'),
('indigo'),('orange'),('red'),('violet'),('yellow');

SELECT name FROM cookbook.color ORDER BY name;


-- One way to retrieve the table rows in a custom order is to field book().
SELECT name FROM cookbook.color
ORDER BY
  FIELD(name,'red','orange','yellow','green','blue','indigo','violet');

-- To accomplish the same end without FIELD(), use ALTER TABLE to convert the name
-- column to an ENUM that lists the colors in the desired sort order:
ALTER TABLE cookbook.color
MODIFY name
ENUM('red','orange','yellow','green','blue','indigo','violet');