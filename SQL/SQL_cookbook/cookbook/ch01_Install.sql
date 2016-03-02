select * from cookbook.limbs;
select MAX(arms+legs) FROM cookbook.limbs;

-- using user-defined variable
SELECT @max_limbs := MAX(arms+legs) FROM cookbook.limbs;
SELECT * FROM cookbook.limbs WHERE arms+legs = @max_limbs;
SELECT @max_limbs;

-- LAST_INSERT_ID() returns the most recent AUTO_INCREMENT value
SELECT @last_id := LAST_INSERT_ID();

-- To set a variable explicitly to a particular value, use a SET statement.
SELECT @summ := 4 + 7;
