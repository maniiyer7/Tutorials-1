

SHOW VARIABLES LIKE 'AUTOCOMMIT';

SHOW GLOBAL VARIABLES;


-------------------------------------------------------------------------------
-- CHAPTER 2: BENCHMALRKING

-- BENCHMARK() is MySQL internal benchmarking tool that can be used to test execution speeds for certain types of operations.
-- The return value is always 0; you time the execution by looking at how long the client
-- application reported the query took.

SET @input := 'hello world';
SELECT BENCHMARK(1000000, MD5(@input));
SELECT BENCHMARK(1000000, SHA1(@input));



