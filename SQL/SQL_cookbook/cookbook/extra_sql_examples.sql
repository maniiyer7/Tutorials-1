
-- These queries are not from MySQL Cookbook. I have used the tables created by cookbook for demonstration.

-- Use GROUP_CONCAT() to condense all lines in one column using a GROUP BY
SELECT GROUP_CONCAT(miles) FROM cookbook.driver_log GROUP BY name;

SELECT * FROM cookbook.driver_log;
EXPLAIN SELECT MIN(miles) FROM cookbook.driver_log WHERE rec_id = 1;

SELECT @one := 11;

SELECT @row_num := 0;


-- In SQL, you can assign a value to a variable and use that value at the same time.
SELECT @row_num := 0;
SELECT *, @row_num := @row_num + 1
FROM cookbook.driver_log
ORDER BY name;

-- You can assign the value in different stages of the query. Note that the timing when you update the value matters.

SELECT @row_num := 0;
SELECT *, @row_num AS row_num
FROM cookbook.driver_log
ORDER BY name, @row_num := @row_num + 1;


SELECT @row_num := 0;
SELECT *, @row_num := @row_num + 1 AS row_num
FROM cookbook.driver_log
WHERE name = 'Ben'
;


SELECT @row_num := 0;
SELECT *, @row_num AS row_num
FROM cookbook.driver_log
WHERE @row_num := @row_num + 1;


-- See how the order of execution impacts the value of user-defined values.
SELECT @row_num := 0;
SELECT *, @row_num AS row_num
FROM cookbook.driver_log
WHERE (@row_num := @row_num + 1) <= 1
ORDER BY name;
