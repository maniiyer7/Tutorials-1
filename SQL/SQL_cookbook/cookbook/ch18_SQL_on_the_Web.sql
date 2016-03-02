

--

SELECT phrase_val FROM cookbook.phrase ORDER BY phrase_val;

SELECT NOW(), VERSION(), DATABASE();

SELECT * FROM cookbook.ingredient ORDER BY id;

SELECT year, artist, title FROM cookbook.cd ORDER BY artist, year;


-------------------------------------------------------------------------------
-- STORING AND RETRIEVING IMAGE DATA IN MySQL

SELECT * FROM cookbook.image;


SELECT LOAD_FILE('/Users/amirkavousian/Documents/AMIR/Website/Appearence/Logo.png');
SHOW grants;


INSERT INTO cookbook.image (name, type, data) VALUES('Logo', 'PNG', LOAD_FILE('/Users/amirkavousian/Documents/AMIR/Website/Appearence/Logo.png'));

SELECT name, type, data FROM cookbook.image ORDER BY RAND() LIMIT 1;

-------------------------------------------------------------------------------
-- ch 19.2
SELECT color FROM cookbook.cow_color ORDER BY color;

-- The list of legal figurine sizes in the size column of the cow_order table
SELECT COLUMN_TYPE, COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA='cookbook' AND TABLE_NAME='cow_order'
AND COLUMN_NAME='size';

SELECT COLUMN_TYPE, COLUMN_DEFAULT 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA='cookbook' AND TABLE_NAME='cow_order'
AND COLUMN_NAME='accessories';

-------------------------------------------------------------------------------
SELECT * FROM cookbook.image;

SELECT * FROM cookbook.states;

SELECT name, abbrev, statehood, pop FROM cookbook.states ORDER BY name LIMIT 5,10;

SELECT column_name FROM information_schema.columns 
WHERE table_schema = 'cookbook' AND table_name = 'driver_log' AND ordinal_position = 1;
        
        
SELECT column_name FROM information_schema.columns WHERE table_schema = 'cookbook' AND table_name = 'driver_log' AND ordinal_position = 1;

SELECT * FROM cookbook.driver_log;
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- PAGE VIEW COUNT
CREATE TABLE cookbook.hitcount
  (
  path VARCHAR(255)
  CHARACTER SET latin1 COLLATE latin1_general_cs NOT NULL,
  hits BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (path)
  );
---------------------------------------

---------------------------------------
-- PAGE COUNT
-- We could have used hits = hits + 1. However, between the time we increment hits by 1, and the time
-- we query it in the next command, another user may have accessed the page, resulting in hits value being obsolete by the time we make the second query.

-- By using LAST_INSERT_ID(), we tell MySQL to treat the value as an AUTO_INCREMENT value, which enables it to be retrieved using LAST_INSEET_ID() statement.

-- In this method, we do not need to query the table twice. Instead, we increment the hits once, and tell MySQL to record the latest value of hits
-- in its LAST_INSERT_ID() value. Then, we can retrieve the LAST_INSERT_ID() value in a separate statement.

INSERT INTO cookbook.hitcount (path,hits) VALUES('some path',LAST_INSERT_ID(1))
ON DUPLICATE KEY UPDATE hits = LAST_INSERT_ID(hits+1);

INSERT INTO cookbook.hitcount (path, hits) VALUES('driver_log.py', LAST_INSERT_ID(1)) ON DUPLICATE KEY UPDATE hits = LAST_INSERT_ID(hits+1);


SELECT LAST_INSERT_ID();

SELECT * FROM cookbook.hitcount;

SELECT hits from cookbook.hitcount WHERE path='driver_log.py';
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- WEBSITE ACCESS LOGGING
CREATE TABLE cookbook.hitlog
  (
  path VARCHAR(255)
  CHARACTER SET latin1 COLLATE latin1_general_cs NOT NULL,
  t TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  host VARCHAR(255),
  INDEX (path)
  );

INSERT INTO cookbook.hitlog (path, host) VALUES(path_val,host_val);
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- ACCESS LOGGING IN MySQL
CREATE TABLE httpdlog
(
  dt DATETIME NOT NULL, # request date
  host VARCHAR(255) NOT NULL, # client host
  method VARCHAR(4) NOT NULL, # request method (GET, PUT, etc.)
  url VARCHAR(255) # URL path
  CHARACTER SET latin1 COLLATE latin1_general_cs NOT NULL,
  status INT NOT NULL, # request status
  size INT, # number of bytes transferred
  referer VARCHAR(255), # referring page
  agent VARCHAR(255) # user agent
);


---------------------------------------
SELECT * FROM cookbook.httpdlog;


-------------------------------------------------------------------------------
-- SESSION LOGGING IN MySQL
CREATE TABLE cookbook.perl_session
(
id CHAR(32) NOT NULL, # session identifier
a_session LONGBLOB, # session data
PRIMARY KEY (id)
);

ALTER TABLE cookbook.perl_session
ADD update_time TIMESTAMP NOT NULL,
ADD INDEX (update_time);


-- To expire sessions, run a statement periodically that sweeps the table and removes old rows. 
-- The following statement uses an expiration time of four hours:
DELETE FROM cookbook.perl_session WHERE update_time < NOW() - INTERVAL 4 HOUR;

-- To expire rows automatically, create a scheduled event:
CREATE EVENT cookbook.expire_perl_session
ON SCHEDULE EVERY 4 HOUR
DO DELETE FROM perl_session WHERE update_time < NOW() - INTERVAL 4 HOUR;


---------------------------------------
-- USING Ruby
CREATE TABLE cookbook.ruby_session
(
session_id VARCHAR(255) NOT NULL,
session_value LONGBLOB NOT NULL,
update_time DATETIME NOT NULL,
PRIMARY KEY (session_id)
);











