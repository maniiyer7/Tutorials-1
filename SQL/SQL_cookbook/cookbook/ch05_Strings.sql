-- STRING: a few general notes about strings:
-- (a) A string can be binary or nonbinary. Binary strings are used for raw data such as
-- images, music files, or encrypted values. Nonbinary strings are used for character
-- data such as text and are associated with a character set and collation (sort order).
-- (b) A character set determines which characters are legal in a string.

-- Data types for binary strings are BINARY, VARBINARY, and BLOB. 
-- Data types for nonbinary strings are CHAR, VARCHAR, and TEXT, each of which permits CHARACTER SET and COLLATE attributes.

-- You can convert a binary string to a nonbinary string and vice versa, 
-- or convert a nonbinary string from one character set or collation to another.

-- Binary strings are compared byte by byte using numeric byte values.
-- A nonbinary string is a sequence of characters. It stores text that has a particular character set and collation.


---------------------------------------
-- LENGTH(): the length of a string in bytes
-- CHAR_LENGTH() the length of a string in characters.
-- If LENGTH() is greater than CHAR_LENGTH() for a given string, multibyte characters are present.

-- Example: a given utf8 string might contain only single-byte characters. As a result, LENGTH() and CHAR_LENGTH() return the same values.
SELECT @s := CONVERT('abc' USING utf8);
SELECT LENGTH(@s), CHAR_LENGTH(@s);

-- For the ucs2 Unicode character set, all characters are encoded using two bytes
SELECT @s := CONVERT('abc' USING ucs2);
SELECT LENGTH(@s), CHAR_LENGTH(@s);


--------------------------------------------------------------------------------
-- COLLATION AND SORT ORDER
-- collation determines the sort order of characters in the character set.
-- A collation can be case sensitive (a and A are different), 
-- case insensitive (a and A are the same), 
-- or binary (two characters are the same or different based on whether their numeric values are equal).
-- This property is usually specified in the collation name, as it ends in collation name ending in _ci, _cs, or _bin.

-- Example of how collation changes the way a column is sorted
CREATE TABLE cookbook.t (c CHAR(3) CHARACTER SET latin1);

INSERT INTO cookbook.t (c) VALUES ('AAA'),('bbb'),('aaa'),('BBB');

SELECT c FROM cookbook.t;

-- A case-insensitive collation sorts a and A together, placing them before b and B.
-- However, for a given letter, it does not necessarily order one lettercase before another.
SELECT c FROM cookbook.t ORDER BY c COLLATE latin1_swedish_ci;

-- A case-sensitive collation puts A and a before B and b, and sorts uppercase before lowercase
SELECT c FROM cookbook.t ORDER BY c COLLATE latin1_general_cs;

-- A binary collation sorts characters using their numeric values.
-- In this case, uppercase letters normally have numeric values less than those of lowercase letters.
SELECT c FROM cookbook.t ORDER BY c COLLATE latin1_bin;


-- Criteria to choose the collation type:
--   Are the strings binary or nonbinary?
--   Does case sensitivity matter?
--   What is the maximum string length?
--   Do you want to store fixed- or variable-length values?
--   Do you need to retain trailing spaces?
--   Is there a fixed set of permitted values?

-- MySQL provides several binary and nonbinary string data types. 
-- These types come in
-- pairs as shown in the following table. 
-- For nonbinary types, the maximum number of characters is less
-- for strings that contain multibyte characters.

-- binary data type | non-binary data type | max length
-- BINARY           | CHAR                 | 255
-- VARBINARY        | VARCHAR              | 65,535
-- TINYBLOB         | TINYTEXT             | 255
-- BLOB             | TEXT                 | 65,535
-- MEDIUMBLOB       | MEDIUMTEXT           | 16,777,215
-- LONGBLOB         | LONGTEXT             | 4,294,967,295

-- For the BINARY and CHAR data types, MySQL stores column values using a fixed width.
-- Shorter values are padded to the required length as necessary.
-- For VARBINARY, VARCHAR, and the BLOB and TEXT types, MySQL stores values using only
-- as much storage as required, up to the maximum column length. No padding is added
-- or stripped when values are stored or retrieved.


---------------------------------------
-- By default, padding characters are removed when the data are retrieved.
-- To preserve trailing pad values that are present in the original strings that are stored,
-- use a data type for which no stripping occurs; e.g., VARCHAR()
DROP TABLE cookbook.t;
CREATE TABLE cookbook.t (c1 CHAR(10), c2 VARCHAR(10));
INSERT INTO cookbook.t (c1,c2) VALUES('abc        ','abc       '), ('abcdefghijklmnop', 'abcdefghijklmnop');
SELECT c1, c2, CHAR_LENGTH(c1) as `CHAR(10)`, CHAR_LENGTH(c2) AS `VARCHAR(10)` FROM cookbook.t;

-- The column length value which is assigned at the time of table creation sets an upper bound on the length of strings to insert into the column.
-- If a string with larger length is supplied, MySQL issues as alert (it does not cut the string at max length).
INSERT INTO cookbook.t (c1,c2) VALUES('abc        ','abc       '), ('abcdefghijklmnop', 'abcdefghijklmnop');

---------------------------------------
-- CONFIGURING THE CLIENT CHARACTER SET
-- When you send information back and forth between your application and the server,
-- you may need to tell MySQL the appropriate character set.
-- To resolve this, configure your connection to use the appropriate character set.
-- Issue a SET NAMES statement after you connect:
SET NAMES 'utf8' COLLATE 'utf8_general_ci';


-------------------------------------------------------------------------------
-- WRITING STRING LITERALS
-- It is good practice to use single-quoted strings.
-- Under ANSI_QUOTES SQL mode, the server interprets double quote as the quoting character for
-- identifiers such as table or column names, and not for strings.

-- You can also use hexadecimal notation to write strings in a table.
-- MySQL treats strings written using hex notation as binary strings.

-- If you want to specify the character set for a string literal,
-- use an introducer clause, consisting of the character-set name preceded by an underscore.
-- Example: _latin1 'abcd'

-- If you have a quote character in your string literal, 
-- escape the quote by either using two quote characters or use a backslash
-- Example:
-- SELECT 'I''m asleep', 'I\'m wide awake';

-- More examples of escaping non-character inputs using backslash:
-- \\ (backslash iteself)
-- \b (backspace)
-- \n (newline, also called linefeed)
-- \r (carriage return)
-- \t (tab)
-- \0 (ASCII NUL).

-- You can also write the string as a hex value
SELECT 0x49276D2061736C656570;


-------------------------------------------------------------------------------
-- CHECKING OR CHANGING A STRING CHARACTER SET OR COLLATION
-- To check a string’s character set or collation, use the CHARSET() or COLLATION() function.
-- To change its character set, use the CONVERT() function. 
-- To change its collation,  use the COLLATE operator.

-- If you have the wrong characterset or collation mismatch, you get an error when comparing strings,
-- or a lettercase conversion doesn’t work properly.

SELECT USER(), CHARSET(USER()), COLLATION(USER());

SET NAMES 'latin1';
SELECT CHARSET('abc'), COLLATION('abc');

SET NAMES 'utf8' COLLATE 'utf8_bin';
SELECT CHARSET('abc'), COLLATION('abc');

-- To convert a string from one character set to another, use the CONVERT() function:
SELECT @s1 := _latin1 'my string', @s2 := CONVERT(@s1 USING utf8);
SELECT CHARSET(@s1), CHARSET(@s2);

-- To change the collation of a string, use the COLLATE operator
SELECT @s1 := _latin1 'my string', @s2 := @s1 COLLATE latin1_spanish_ci;
SELECT COLLATION(@s1), COLLATION(@s2);

-- To change both character set and collation of a string, use both CONVERT function followed by a COLLATION operator.
SELECT @s1 := _latin1 'my string';
SELECT @s2 := CONVERT(@s1 USING utf8) COLLATE utf8_spanish_ci;
SELECT CHARSET(@s1), COLLATION(@s1), CHARSET(@s2), COLLATION(@s2);

-- The CONVERT() function can also convert binary strings to nonbinary strings and vice versa.
SELECT @s1 := _latin1 'my string';
SELECT @s2 := CONVERT(@s1 USING binary);
SELECT @s3 := CONVERT(@s2 USING utf8);
SELECT CHARSET(@s1), CHARSET(@s2), CHARSET(@s3);

-- Another method to create binary strings is to use the BINARY operator.
SELECT CHARSET(BINARY _latin1 'my string');


-------------------------------------------------------------------------------
-- CONVERTING THE LETTERCASE OF A STRING
-- Use the UPPER() or LOWER() function on non-binary strings. 
SELECT thing, UPPER(thing), LOWER(thing) FROM cookbook.limbs;

-- Make sure you are not operating on a binary string such as BLOB. In wuch cases, UPPER() and LOWER() do nothing.
DROP TABLE cookbook.t;
CREATE TABLE cookbook.t (b BLOB) SELECT 'aBcD' AS b;
SELECT b, UPPER(b), LOWER(b) FROM cookbook.t;

-- To map a binary string to a given lettercase, convert it to a nonbinary string, choosing
-- a character set that has uppercase and lowercase characters.
SELECT 
  b,
  UPPER(CONVERT(b USING latin1)) AS upper,
  LOWER(CONVERT(b USING latin1)) AS lower
FROM cookbook.t;

-- To make a string first-letter capitalized:
-- CONCAT(UPPER(LEFT(str,1)),MID(str,2));

--CREATE FUNCTION cookbook.initial_cap (s VARCHAR(255))
--RETURNS VARCHAR(255) DETERMINISTIC
--RETURN CONCAT(UPPER(LEFT(s,1)),MID(s,2));

SELECT thing, cookbook.initial_cap(thing) FROM cookbook.limbs;


-------------------------------------------------------------------------------
-- CONTROLING CASE SENSITIVITY IN STRING COMPARISONS
-- A binary string is a sequence of bytes and is compared using numeric byte values.
-- Because letters in different cases have different byte values, comparisons of binary strings effectively are case sensitive.
-- However, the other of the byte values is not always consistent. Therefore, it is not a good idea to compare strings in their binary format.
-- To compare binary strings such that lettercase does not matter, convert them to nonbinary strings that have a a case-insensitive collation.
-- A nonbinary string is a sequence of characters and is compared in character units.

-- By default, strings have a character set of latin1 and a collation of latin1_swedish_ci 
-- unless you reconfigure the server. 
-- This results in case-insensitive string comparisons.

-- Binary is case-sensitive:
SELECT @s1 := BINARY 'cat', @s2 := BINARY 'CAT';
SELECT @s1 = @s2;

-- default collation (latine1_swedish_ci) is case insensitive
SELECT @s1 := CONVERT(@s1 USING latin1) COLLATE latin1_swedish_ci;
SELECT @s2 := CONVERT(@s2 USING latin1) COLLATE latin1_swedish_ci;
SELECT @s1 = @s2;

-- latin1_general_cs is case-sensitive
SELECT @s1 COLLATE latin1_general_cs = @s2 COLLATE latin1_general_cs AS '@s1 = @s2';

-- If you compare a binary string with a nonbinary string, the comparison treats both operands as binary strings.
-- Thus, to compare two non-binary strings as binary strings, apply the BINARY operator to either one when comparing them.
SELECT _latin1 'cat' = BINARY 'CAT';


-------------------------------------------------------------------------------
-- PATTERN MATCHING WITH SQL PATTERNS
-- MySQL provides two kinds of pattern matching. 
-- One is based on SQL patterns and the other on regular expressions.

-- SQL pattern matching uses the LIKE and NOT LIKE operators.
-- Patterns may contain two special meta-characters:
-- _ matches any single character, and 
-- % matches any sequence of characters,

-- Strings that begin with a particular substring:
SELECT name FROM cookbook.metal WHERE name LIKE 'me%';

-- Strings that end with a particular substring:
SELECT name FROM cookbook.metal WHERE name LIKE '%d';

-- Strings that contain a particular substring at any position:
SELECT name FROM cookbook.metal WHERE name LIKE '%in%';

-- Strings that contain a substring at a specific position.
-- Example: the pattern matches only if at occurs at the third position of the name column):
SELECT name FROM cookbook.metal WHERE name LIKE '__at%';


-- To reverse the sense of a pattern match, use NOT LIKE.
SELECT name FROM cookbook.metal WHERE name NOT LIKE '%i%';

-- SQL patterns do not match NULL values.
SELECT NULL LIKE '%', NULL NOT LIKE '%';

-- If you’re matching against a column that is indexed and you have a choice of using a
-- pattern or an equivalent LEFT() expression, you’ll likely find the pattern match to be
-- faster. MySQL can use the index to narrow the search for a pattern that begins with a
-- literal string. With LEFT(), it cannot use the index.

-- We can apply pattern matching to dates too
-- Function value test     |   Pettern matching test
-- YEAR(d) = 1976          |   d LIKE '1976-%'
-- MONTH(d) = 4            |   d LIKE '%-04-%'
-- DAYOFMONTH(d) = 1       |   d LIKE '%-01'


-------------------------------------------------------------------------------
-- PATTERN MATCHING USING REGULAR EXPRESSIONS
-- REGEXP matching uses the pattern elements shown in the following table:
-- |  ^            |   Beginning of string
-- |  $            |   End of string
-- |  .            |   Any single character
-- |  [...]        |   Any character listed between the square brackets
-- |  [^...]       |   Any character not listed between the square brackets
-- |  p1|p2|p3     |   Alternation; matches any of the patterns p1, p2, or p3
-- |  *            |   Zero or more instances of preceding element
-- |  +            |   One or more instances of preceding element
-- |  {n}          |   n instances of preceding element
-- |  {m,n}        |   m through n instances of preceding element

-- Strings that begin with a particular substring
SELECT name FROM cookbook.metal WHERE name REGEXP '^me';

-- Strings that end with a particular substring
SELECT name FROM cookbook.metal WHERE name REGEXP 'd$';

-- Strings that contain a particular substring at any position
SELECT name FROM cookbook.metal WHERE name REGEXP 'in';
-- Note that the REGEX itself returns a True/False value
SELECT name REGEXP 'in' FROM cookbook.metal;

-- Strings that contain a particular substring at a specific position (in this example: on the third position)
SELECT name FROM cookbook.metal WHERE name REGEXP '^..at';


---------------------------------------
-- REGEX WITH CHARACTER CLASSES
-- regular expressions can contain character classes, which match any character in the class.
-- To write a character class, use square brackets: the pattern [abc] matches a, b, or c.
-- Use a dash between the beginning and end of the range. 
-- [a-z] matches any letter, [0-9] matches digits, and [a-z0-9] matches letters or digits.
-- [^] negates the range. For example, [^0-9] matches anything but digits.


-- MySQL’s regular-expression capabilities also support POSIX character classes. Use POSIX classes within brackets too.
-- |  [:alnum:]     |   Alphabetic and numeric characters
-- |  [:alpha:]     |   Alphabetic characters
-- |  [:blank:]     |   Whitespace (space or tab characters)
-- |  [:cntrl:]     |   Control characters
-- |  [:digit:]     |   Digits
-- |  [:graph:]     |   Graphic (nonblank) characters
-- |  [:lower:]     |   Lowercase alphabetic characters
-- |  [:print:]     |   Graphic or space characters
-- |  [:punct:]     |   Punctuation characters
-- |  [:space:]     |   Space, tab, newline, carriage return
-- |  [:upper:]     |   Uppercase alphabetic characters
-- |  [:xdigit:]    |   Hexadecimal digits (0-9, a-f, A-F)

SELECT name, name REGEXP '[[:xdigit:]]' FROM cookbook.metal;
SELECT name, name REGEXP '[[:alpha:]]' FROM cookbook.metal;
SELECT name, name REGEXP '[[:graph:]]' FROM cookbook.metal;
SELECT name, name REGEXP '[[:punct:]]' FROM cookbook.metal;


-- With REGEX, the alternatives are not limited to single characters. They can be multiple-character strings or even patterns
-- The following alternation matches strings that begin with a vowel or end with er:
SELECT name FROM cookbook.metal WHERE name REGEXP '^[aeiou]|d$';


---------------------------------------
-- If you group the alternatives within parentheses, the ^ and $ apply to both of them.
-- Note the difference between the following patterns:
-- Match strings that begin with one or more digits, or strings that end with one or more letters.
SELECT '0m' REGEXP '^[[:digit:]]+|[[:alpha:]]+$';

-- Match strings that consist entirely of digits or entirely of letters (note the placement of parentheses at the two ends of the regex pattern).
SELECT '0m' REGEXP '^([[:digit:]]+|[[:alpha:]]+)$';


---------------------------------------
-- Regular expressions are successful if the pattern matches anywhere within the value.
-- Regular expressions do not match NULL values.
-- Take care not to inadvertently specify a pattern that matches the empty string. If you do, it matches any non-NULL value.
-- Note that asterisk (*) matches zero or more. So it is not very informative if the goal is to find a specific character.
SELECT 'abcdef' REGEXP 'g*';
-- If your goal is to match only strings containing nonempty sequences of a characters, use a+ instead.
SELECT 'abcdef' REGEXP 'g+';
SELECT 'abcdef' REGEXP 'b+';

-- So, to match strings that begin with any nonempty sequence of digits, use this pattern match:
SELECT 'abcdef' REGEXP '^[0-9]+';
-- But do not use this pattern:
SELECT 'abcdef' REGEXP '^[0-9]*';


-------------------------------------------------------------------------------
-- BREAKING APART OR COMBINING STRINGS
-- MID(string, starting_pos, num_chars)
-- SUBSTRING(string, starting_pos, num_chars)
-- LEFT(string, num_chars)
-- RIGHT(string, num_chars)

-- LEFT(), MID(), and RIGHT() extract substrings from the left, middle, or right part of a string:
SELECT @date := '2015-07-21';
SELECT 
  @date, 
  LEFT(@date,4) AS year,
  MID(@date,6,2) AS month, 
  RIGHT(@date,2) AS day;

-- SUBSTRING_INDEX(str,c,n) returns everything to the right or left of a given character.
-- It searches into a string str for the n-th occurrence of the character c and
-- returns everything to its left. If n is negative, the search for c starts from the right and
-- returns everything to the right of the character.
SELECT @email := 'postmaster@example.com';
SELECT 
  @email,
  SUBSTRING_INDEX(@email,'@',1) AS user,
  SUBSTRING_INDEX(@email,'@',-1) AS host;


--------------------------------------------------------------------------------
-- To find metal names having a first letter that lies in the last half of the alphabet:
SELECT name from cookbook.metal WHERE LEFT(name,1) >= 'n';


--------------------------------------------------------------------------------
-- CONCAT() function attaches strings together
SELECT 
  CONCAT(name,' ends in "d": ',IF(RIGHT(name,1)='d','YES','NO')) AS 'ends in "d"?'
FROM cookbook.metal;

-- the following UPDATE statement adds a string to the end of each name value in the metal table:
UPDATE cookbook.metal SET name = CONCAT(name,'ide');
SELECT name FROM cookbook.metal;

-- To undo the operation
UPDATE cookbook.metal SET name = LEFT(name,CHAR_LENGTH(name)-3);
SELECT name FROM cookbook.metal;


-------------------------------------------------------------------------------
-- SEARCHING FOR SUBSTRINGS
-- Use LOCATE() or a pattern match to see if a string occurs within another string.
-- LOCATE() parameters: 
--   the substring that we are looking for
--   the string in which to look for it
--   which index to start looking at
SELECT name, LOCATE('in',name), LOCATE('in',name,3) FROM cookbook.metal;

-- Note that if the position of the substring is not important, we can use LIKE or REGEXP
SELECT name, name LIKE '%in%', name REGEXP 'in' FROM cookbook.metal;


-------------------------------------------------------------------------------
-- USING FULL-TEXT SEARCH
-- For large texts or searching in multiple columns, pattern matching becomes slow quickly.
-- Full-text searching is designed for looking through large amounts of text and can search multiple columns simultaneously.

-- To use this capability, add a FULLTEXT index to your table, 
-- and then use the MATCH operator to look for strings in the indexed column or columns.

-- FULLTEXT indexing can be used with MyISAM tables (or, as of MySQL 5.6, InnoDB tables) for 
-- nonbinary string data types (CHAR, VARCHAR, or TEXT).

-- Make sure you can create a table using one of the following engines: MyISM, InnDB, MySQL 5.6 or higher
SELECT `ENGINE` FROM `information_schema`.`TABLES`
  WHERE `TABLE_SCHEMA`='cookbook';

DROP TABLE cookbook.kjv;

-- Note that the table has a FULLTEXT index to enable its use in full-text searching.
CREATE TABLE cookbook.kjv
(
bsect ENUM('O','N') NOT NULL, # book section (testament)
bname VARCHAR(20) NOT NULL, # book name
bnum TINYINT UNSIGNED NOT NULL, # book number
cnum TINYINT UNSIGNED NOT NULL, # chapter number
vnum TINYINT UNSIGNED NOT NULL, # verse number
vtext TEXT NOT NULL, # text of verse
FULLTEXT (vtext) # full-text index
) ENGINE = InnoDB;

LOAD DATA LOCAL INFILE '/Users/amirkavousian/Documents/SQL_Codes/Tutorials/mcb-kjv/kjv.txt' INTO TABLE cookbook.kjv;

SELECT * FROM cookbook.kjv;


-- To perform a search using the FULLTEXT index, use MATCH() to name the indexed column
-- and AGAINST() to specify what text to look for.

-- For example, to find out how many times the name Hadoram occurs in the text:
SELECT COUNT(*) from cookbook.kjv 
WHERE MATCH(vtext) AGAINST('Hadoram');

-- To find out what those verses are, select the columns you want to see:
SELECT bname, cnum, vnum, LEFT(vtext,65) AS vtext
FROM cookbook.kjv 
WHERE MATCH(vtext) AGAINST('Hadoram');

-- If you expect to use search criteria frequently that include other non-FULLTEXT columns,
-- add regular indexes to those columns so that queries perform better.
ALTER TABLE cookbook.kjv ADD INDEX (bnum), ADD INDEX (cnum), ADD INDEX (vnum);

-- If you add more words to the string to search against,
-- full-text search returns rows that contain any of the words.
-- In fact, the query performs an OR search for any of the words.
SELECT COUNT(*) from cookbook.kjv
WHERE MATCH(vtext) AGAINST('Abraham');

SELECT COUNT(*) from cookbook.kjv
WHERE MATCH(vtext) AGAINST('Abraham Sarah');


-- To use full-text searches that look through multiple columns simultaneously, name all
-- the columns when you construct the FULLTEXT index
-- ALTER TABLE tbl_name ADD FULLTEXT (col1, col2, col3);

-- To issue a search query that uses the index, name those same columns in the MATCH() list:
-- SELECT ... FROM tbl_name
-- WHERE MATCH(col1, col2, col3) AGAINST('search string');
-- NOTE that you need one such FULLTEXT index for each distinct combination of columns that you want to search.


-- Note that the fulltext search index has a minimum word length parameter.
-- If your query does not return any results for short word, change that parameter. 
-- The parameter name is 'ft_min_word_len' and it can be set in my.cnf file.
-- If you are using InnoDB, the parameter name is innodb_ft_min_token_size
SELECT COUNT(*) FROM cookbook.kjv WHERE MATCH(vtext) AGAINST('God');

-- Another property of the indexing engine is that it ignores words that are “too common”
-- (that is, words that occur in more than half the rows)

---------------------------------------
-- To require or prohibit specific words in a fulltext search query, use a boolean mode search.

-- Normally, full-text searches return rows that contain any of the words in the search
-- string, even if some of them are missing
SELECT COUNT(*) FROM cookbook.kjv
WHERE MATCH(vtext) AGAINST('David Goliath');

-- If you want only rows that contain both words,
-- one strategy is to do separate searches and use a AND condition.
SELECT COUNT(*) 
FROM cookbook.kjv
WHERE MATCH(vtext) AGAINST('David')
AND MATCH(vtext) AGAINST('Goliath');

-- Another way to do this is to use Boolean mode search.
-- precede each word in the search string with a + character and add IN BOOLEAN MODE after the string:
SELECT COUNT(*) 
FROM cookbook.kjv
WHERE MATCH(vtext) AGAINST('+David +Goliath' IN BOOLEAN MODE);

-- Boolean mode searches also permit you to exclude words by preceding each one with a - character.
SELECT COUNT(*) 
FROM cookbook.kjv
WHERE MATCH(vtext) AGAINST('+David -Goliath' IN BOOLEAN MODE);


-------------------------------------------------------------------------------
-- PERFORMING FULLTEXT PHRASE SEARCHES
-- A phrase is a collection of words that occur adjacent to each other and in a specific order.
-- To find lines with a phrase in them, a simple full-text search doesn’t work. The text search would return any line with any of those words in it.
-- Instead, use full-text Boolean mode, which supports phrase searching. 
-- Enclose the phrase in double quotes within the search string:
SELECT COUNT(*) 
FROM cookbook.kjv
WHERE MATCH(vtext) AGAINST('"still small voice"' IN BOOLEAN MODE);









