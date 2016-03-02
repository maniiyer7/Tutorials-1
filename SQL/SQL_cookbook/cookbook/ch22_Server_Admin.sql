
-- After the server starts, you can make runtime adjustments by changing system variables using the SET statement.
-- SET GLOBAL var_name = value;

SET @@SESSION.sql_mode = 'STRICT_ALL_TABLES';
-- STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION

SELECT @@GLOBAL.secure_auth, @@SESSION.sql_mode;

-- Find the MySQL plugin directory
SELECT @@plugin_dir;

SHOW GLOBAL STATUS LIKE 'Threads_connected';

SELECT VARIABLE_VALUE FROM INFORMATION_SCHEMA.GLOBAL_STATUS
WHERE VARIABLE_NAME = 'Threads_connected';

SHOW TABLES FROM INFORMATION_SCHEMA LIKE 'innodb%';

SHOW GLOBAL STATUS LIKE 'Uptime';


-------------------------------------------------------------------------------
------------------------------ CH 23: SECURITY --------------------------------
SHOW CREATE TABLE mysql.user;

