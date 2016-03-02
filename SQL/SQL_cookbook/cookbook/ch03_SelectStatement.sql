
-- create the table that we will be working with
CREATE TABLE cookbook.mail
(
t DATETIME, -- when message was sent
srcuser VARCHAR(8), -- sender (source user and host)
srchost VARCHAR(20),
dstuser VARCHAR(8), -- recipient (destination user and host)
dsthost VARCHAR(20),
size BIGINT, -- message size in bytes
INDEX (t)
);

-- to populate it, use the mail.sql script in tables directory.

SELECT * FROM cookbook.mail;


--------------------------------------------------------------------------------
-- DISTINCT COMMAND
SELECT DISTINCT srcuser FROM cookbook.mail;

SELECT COUNT(DISTINCT srcuser) FROM cookbook.mail;

SELECT DISTINCT YEAR(t), MONTH(t), DAYOFMONTH(t) FROM cookbook.mail;


--------------------------------------------------------------------------------
-- COMPARING WITH NULL
-- comparisons such as value = NULL or value <> NULL always produce a result of NULL (not true or false)
SELECT * FROM cookbook.expt WHERE score IS NULL;

-- even directly comparing NULL against NULL will result in NULL, unless using MySQL-specific command <=> 
SELECT NULL = NULL, NULL <=> NULL;

-- if NULL values have a specific meaning in the context of your application, you can 
-- map the NULL values into a more specific value or string
SELECT subject, test, IF(score IS NULL,'Unknown', score) AS 'score' FROM cookbook.expt;
-- alternatively, use the built-in function IFNULL()
SELECT subject, test, IFNULL(score,'Unknown') AS 'score' FROM cookbook.expt;


--------------------------------------------------------------------------------
-- USING A VIEW
CREATE VIEW cookbook.mail_view AS
SELECT
  DATE_FORMAT(t,'%M %e, %Y') AS date_sent,
  CONCAT(srcuser,'@',srchost) AS sender,
  CONCAT(dstuser,'@',dsthost) AS recipient,
  size 
FROM 
  cookbook.mail;

-- you can use a view like any other table
SELECT 
  date_sent, sender, size 
FROM 
  cookbook.mail_view
WHERE 
  size > 100000 
ORDER BY 
  size;


-- USING MULTIPLE TABLES: join or subquery
SELECT * FROM cookbook.profile_contact ORDER BY profile_id, service;

SELECT 
  id, name, service, contact_name
FROM 
  cookbook.profile 
  INNER JOIN cookbook.profile_contact 
    ON id = profile_id;



-------------------------------------------------------------------------------
-- FIND USERS WITH MORE THAN 3 SOCIAL MEDIA LINKS
SELECT 
  name,
  counts.sm_count
FROM
  (SELECT
    contact_name,
    profile_id,
    service,
    @counter := IF(@prev = LEFT(contact_name, 5), @counter+1, 1) AS sm_count,
    @prev := LEFT(contact_name, 5)
  FROM
    cookbook.profile_contact y, 
    (SELECT @counter := 1, @prev := NULL) tmp) counts
  INNER JOIN cookbook.profile
  ON profile.id = counts.profile_id
WHERE
sm_count >= 3;


---------------------------------------
-- How many social media pages each user has?
-- Note how we defined the variables in a sub-query (no need to defined them in a separate query as the examples in this book suggest)
-- Note: SQL puts the table together sequentially, line by line. As a result, we can have variables link @counter and @prev in this example
-- that change and evolve line-by-line.

SELECT
  contact_name,
  service,
  @counter := IF(@prev = LEFT(contact_name, 5), @counter+1, 1) AS sm_count,
  @prev := LEFT(contact_name, 5)
FROM
  cookbook.profile_contact y, 
  (SELECT @counter := 1, @prev := NULL) tmp;

-- Another way is to use summary functions such as COUNT()
SELECT
  profile_id,
  contact_name,
  COUNT(*)
FROM
  cookbook.profile_contact
GROUP BY
  profile_id;


SELECT * FROM cookbook.profile_contact;
SELECT * FROM cookbook.profile;
