
show create table mysql.user;
select * from cookbook.driver_log;

---------------------------------------
-- Creating accounts (CREATE USER, SET PASSWORD)
-- One alternative is to create the user and determine its password in one statement. This does not leave space for setting the auth method.
-- As a result, the default auth method will be used.
CREATE USER 'user_name'@'host_name' IDENTIFIED BY 'password';

-- Another alternative is to set the auth method at the same time as the user, and then set the pass separately.
-- This method is also explained below.
CREATE USER 'user_name'@'host_name' IDENTIFIED WITH 'auth_plugin';



-- (step 1) Execute one the following statements to create the user 
-- The mysql_native_password is the most universal method.
-- You may need to drop the user first.
DROP USER 'amirk'@'localhost';

CREATE USER 'amirk'@'localhost' IDENTIFIED WITH 'mysql_native_password';
SET old_passwords = 0;

CREATE USER 'amirk'@'localhost' IDENTIFIED WITH 'mysql_old_password';
SET old_passwords = 1;

CREATE USER 'amirk'@'localhost' IDENTIFIED WITH 'sha256_password';
SET old_passwords = 2;


-- (step 2) Set the account password
SET PASSWORD FOR 'amirk'@'localhost' = PASSWORD('amirkPass');

-- To test whether the user was created or not
SELECT * FROM mysql.user WHERE user = 'amirk';

-- Putting the above process in a procedure:

CREATE PROCEDURE create_user(user TEXT, host TEXT,
password TEXT, plugin TEXT)
BEGIN
DECLARE account TEXT;
SET account = CONCAT(QUOTE(user),'@',QUOTE(host));
CALL exec_stmt(CONCAT('CREATE USER ',account,
' IDENTIFIED WITH ',QUOTE(plugin)));
IF password IS NOT NULL THEN
BEGIN
DECLARE saved_old_passwords INT;
SET saved_old_passwords = @@old_passwords;
CASE plugin
WHEN 'mysql_native_password' THEN SET old_passwords = 0;
WHEN 'mysql_old_password' THEN SET old_passwords = 1;
WHEN 'sha256_password' THEN SET old_passwords = 2;
ELSE SIGNAL SQLSTATE 'HY000'
SET MYSQL_ERRNO = 1525,
MESSAGE_TEXT = 'unhandled auth plugin';
END CASE;
CALL exec_stmt(CONCAT('SET PASSWORD FOR ',account,
' = PASSWORD(',QUOTE(PASSWORD),')'));
SET old_passwords = saved_old_passwords;
END;
END IF;
IF server_version() >= 50607 AND user <> '' THEN
CALL exec_stmt(CONCAT('ALTER USER ',account,' PASSWORD EXPIRE'));
END IF;
END;

CALL create_user('user_name','host_name','password','auth_plugin');


---------------------------------------
-- Assigning and checking privileges (GRANT, REVOKE, SHOW GRANTS)




---------------------------------------
-- Removing and renaming accounts (DROP USER, RENAME USER)


-- INSTALLING PLUGINS
-- See recipe 22.3

SHOW VARIABLES LIKE 'validate_password%';

SELECT VALIDATE_PASSWORD_STRENGTH('abc') ;

SELECT CURRENT_USER();

SELECT @@secure_auth;

