
-- native MySQL facilities for importing data:
-- the LOAD DATA statement and the mysqlimport command-line program)

-- exporting data: SELECT … INTO OUTFILE statement

-------------------------------------------------------------------------------
-- IMPORTING DATA TO MySQL USING LOAD DATA COMMAND IN MySQL
-- MySQL provides a LOAD DATA statement that acts as a bulk data loader.

LOAD DATA LOCAL INFILE 'mytbl.txt' INTO TABLE mytbl;

-- At some MySQL installations, the LOCAL loading capability may have been disabled for
-- security reasons. If that is true at your site, omit LOCAL from the statement and specify
-- the full pathname to the file, which must be readable by the server.
LOAD DATA INFILE '/Users/amirkavousian/Documents/SQL_Codes/Tutorials/Data/mytbl.txt' INTO TABLE mytbl;


-------------------------------------------------------------------------------
-- IMPORTING DATA TO MySQL USING mysqlimport COMMAND FROM COMMAND LINE

-- The MySQL utility program mysqlimport acts as a wrapper around LOAD DATA so that
-- you can load input files directly from the command line.
-- For the most, what can be done with LOAD DATA can be done with mysqlimport as well.

-- $ mysqlimport --local cookbook mytbl.txt

-- For mysqlimport, as with other MySQL programs, you may need to specify connection
-- parameter options such as --user or --host



-------------------------------------------------------------------------------
-- By default, the MySQL server assumes that the datafile is located on the server host. 
-- You can load local files that are located on the client host using LOAD DATA LOCAL rather than
-- LOAD DATA, unless LOCAL capability is disabled by default.
-- When using mysqlimport on the command line, use --local-infile option. 
-- If that does not work, your server has been configured to prohibit LOAD DATA LOCAL.

-- If the LOAD DATA statement includes no LOCAL keyword, the MySQL server looks for the
-- file on the server host using the following rules:

-- Most of the import/export solutions are resolved in the programming languages other than SQL (Python, Ruby, Perl, etc.)
-- Refer to your own Python code base.
-- READ THE REST OF THE MATERIALS FROM THE BOOK CHAPTER.


-- If having access denied issues:
grant all privileges 
  on cookbook.* 
  to 'cbuser'@'localhost' 
  identified by 'cbpass';

flush privileges; 

GRANT FILE ON *.* TO 'cbuser'@'localhost';

SELECT * FROM cookbook.mail 
INTO OUTFILE '/Users/amirkavousian/Documents/SQL_Codes/Tutorials/Results/mail.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
;

-------------------------------------------------------------------------------















