

-- Normally, MySQL coerces input values to the data types of your table columns if the input doesn’t match.
-- This can result in unexpected behavior, when an invalid value may be recorded in a table because MySQL tried to coerce it into the data type of that column.

-- Several mode values are available to control how strict the server is.
-- Some modes apply generally to all input values. Others apply to specific data types such as dates.

DROP TABLE cookbook.t;

CREATE TABLE cookbook.t (i INT, c CHAR(6), d DATE);

INSERT INTO cookbook.t (i,c,d) VALUES('-1x','too-long string!','1999-02-31');

select * from cookbook.t;

-- To make MySQL more strict, set the sql_mode system variable to enable server restrictions on input data acceptance.
-- With the proper restrictions in place, data values that would otherwise result
-- in conversions and warnings result in errors instead.

SET sql_mode = 'STRICT_ALL_TABLES';

INSERT INTO cookbook.t (i,c,d) VALUES('-1x','too-long string!','1999-02-31');

-- to prohibit dates with zero parts or “zero” dates, set the SQL mode like this:
SET sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE';


---------------------------------------
-- TRADITIONAL mode is actually a constellation of modes, as you can
-- see by setting and displaying the sql_mode value:
SET sql_mode = 'TRADITIONAL';

SELECT @@sql_mode;

-------------------------------------------------------------------------------

-- For step-by-step explanation of specific regex formuals:
-- http://www.regexplained.co.uk

-- In regex, paranthesis are for grouping and capturing
-- http://www.regular-expressions.info/brackets.html
-- Only parentheses can be used for grouping. Square brackets define a character class, and curly braces are used by a quantifier with specific limits.
-- http://stackoverflow.com/questions/9801630/what-is-the-difference-between-square-brackets-and-parentheses-in-a-regex

-------------------------------------------------------------------------------
-- Some general patterns for regex pattern recognition

--   /^$/            Empty value
--   /./             Nonempty value
--   /^\s*$/         Whitespace, possibly empty
--   /^\s+$/         Nonempty whitespace
--   /\S/            Nonempty, and not only whitespace
--   /^\d+$/         Digits only, nonempty
--   /^[a-z]+$/i     Alphabetic characters only (case insensitive), nonempty
--   /^\w+$/         Alphanumeric or underscore characters only, nonempty

--   /^\d+$/                         Unsigned integer
--   /^-?\d+$/                       Negative or unsigned integer
--   /^[-+]?\d+$/                    Signed or unsigned integer
--   /^[-+]?(\d+(\.\d*)?|\.\d+)$/    Floating-point number

-- If you care only that a value begins with an integer, you can match an initial numeric part and extract it. 
-- To do this, match only the initial part of the string (omit the $ that requires the pattern to
-- match to the end of the string) and place parentheses around the \d+ part
--   /^(\d+)/

-------------------------------------------------------------------------------
-- To match zipcodes:
--   /^\d{5}$/             ZIP code, five digits only
--   /^\d{5}-\d{4}$/       ZIP+4 code
--   /^\d{5}(-\d{4})?$/    ZIP or ZIP+4 code

-- To match credit card numbers, written with spaces, dashes, or other characters between groups of digits:
--   0123456789012345
--   0123 4567 8901 2345
--   0123-4567-8901-2345
-- For all of the above patterns, use:    /^[- \d]+/

-------------------------------------------------------------------------------
-- DATE TIME VALUES MATCHING
-- To require values to be dates in ISO (CCYY-MM-DD) format, use this pattern
--   /^\d{4}-\d{2}-\d{2}$/
-- To permit either - or / as the delimiter:
--   /^\d{4}[-\/]\d{2}[-\/]\d{2}$/
-- To avoid the backslashes, use a different delimiter around the pattern:
--   m|^\d{4}[-/]\d{2}[-/]\d{2}$|

-- To permit any nondigit delimiter
--   /^\d{4}\D\d{2}\D\d{2}$/

-- To permit leading zeros in values like 03 to be missing, just look for three nonempty digit sequences:
--   /^\d+\D\d+\D\d+$/

-- To constrain the subpart lengths by requiring two to four digits in the year part and one or two digits in the month and day parts, use this pattern:
--   /^\d{2,4}?\D\d{1,2}\D\d{1,2}$/

-- For dates in other formats such as MM-DD-YY or DD-MM-YY
--   /^\d{2}-\d{2}-\d{2}$/

-- For dates to be in ISO format
--   /^(\d{2,4})\D(\d{1,2})\D(\d{1,2})$/
-- In this case, we enclosed the year, month, day patterns in paranthesis to make them "capturing". This means that the regex returns any matches it finds. The matches are accessible via: 
--   ($year, $month, $day) = ($1, $2, $3);  # perl syntax

-- The capturing regex also allows us to rewrite values in a different order. For example, to rewrite date values assumed to be in MM-DD-YY format into YYMM- DD format, do this:
--   s/^(\d+)\D(\d+)\D(\d+)$/$3-$1-$2/

-- To match time values:
--   /^\d{2}:\d{2}:\d{2}$/

-- permit the hours part to have a single digit, or the seconds part to be missing:
--   /^\d{1,2}:\d{2}(:\d{2})?$/

-- To rewrite times from 12-hour format with AM and PM suffixes to 24-hour format, do this:
--   /^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?$/i


---------------------------------------
-- RANGE VALIDITY CHECKING FOR DATE TIME OBJECTS
-- You need to write a function to perform this operation. The function will have the following components:
-- is_valide_date() is the wrappe function. It accepts the year, month, day components of the date object.
--   It checks to ensure year is between 1 and 9999
--   It checks to ensure month is between 1 and 12
--   For day, we need to write a separate function to check number of days per each month, and also consider leap years.
-- is_leap_year(): returns a True/False value wether the year is a leap year or not. 
-- days_in_month(): generally, days in month are (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
--   If leap year, add 1 day to February.

-- See the chapter for implementation details.

-------------------------------------------------------------------------------
-- EMAIL AND URL MATCHING
-- The most minimal pattern is containing @ sign:
--   /.@./

-- To add more complexities: 

-- Username and the domain name should consist entirely of characters other than @ characters or spaces:
--   /^[^@ ]+@[^@ ]+$/

-- The domain name part hould contain at least two parts separated by a dot:
--   /^[^@ ]+@[^@ .]+\.[^@ .]+/

-- To look for URL values that begin with a protocol specifier of http://, ftp://, or mailto:
--   m#^(http://|ftp://|mailto:)#i
-- Two points about the pattern above:
-- (a) The alternatives in the pattern are grouped within parentheses because otherwise the ^ anchors only the first of them to the beginning of the string.
-- (b) The patterns contain slashes; therefore, we used a different separator to avoid confusion with the slashes that are part of the pattern.


------------------------------------------------------------------------------
-- USING META DATA TO VALIDATE DATA
-- Get the column definition, extract the list of members from it, and check data values against the list.
-- Get the list of legal column values into an array using the information in INFORMATION_SCHEMA, then perform an array membership test

SELECT COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'cookbook' AND TABLE_NAME = 'profile'
AND COLUMN_NAME = 'color';

-- This is done in the programming language (eg, Python). Read the book chapter for more details on this.

-------------------------------------------------------------------------------
-- USING A LOOKUP TABLE TO VALIDATE DATA
-- whereas ENUM and SET columns usually have a small number of member values, a
-- lookup table can have an essentially unlimited number of values. You might not want
-- to read them all into memory.

-- To validate a large number of values, it’s more efficient to pull the lookup values into
-- memory, save them in a data structure, and check each input value against the contents
-- of that structure.

-- Another lookup technique mixes individual statements with a hash that stores lookup
-- value existence information. This approach can be useful if you have a very large lookup table.

-- Begin with an empty hash
-- Then, for each value to be tested, check whether it’s present in the hash. If not, execute
-- a query to check whether the value is present in the lookup table, and record the result
-- of the query in the hash.
-- The validity of the input value is determined by the value
-- associated with the key, not by the existence of the key.
-- For this method, the hash acts as a cache, so that you execute a lookup query for any
-- given value only once, no matter how many times it occurs in the input.

-------------------------------------------------------------------------------
-- CONVERTING TWO-DIGIT YEAR VALUES TO FOUR-DIGIT FORM
-- If you store a date containing a two-digit year, MySQL automatically converts it to four-digit form. 
-- MySQL uses a transition point of 1970; it interprets values from 00 to 69 as the years 2000 to 2069, and
-- values from 70 to 99 as the years 1970 to 1999. 
-- These rules are appropriate for year values in the range from 1970 to 2069.
-- If your values lie outside this range, add the proper century yourself before storing them into MySQL.


-------------------------------------------------------------------------------












