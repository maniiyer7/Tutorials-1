
-- A complete join that produces all possible row combinations is called a Cartesian product.

-- A join can easily cause MySQL to process large numbers of row combinations, 
-- so it’s a good idea to index the comparison columns

SELECT * FROM cookbook.painting;

SELECT * 
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting
  ON artist.a_id = painting.a_id
ORDER BY 
  artist.a_id;


---------------------------------------
-- In the special case of equality comparisons between columns with the same name in
-- both tables, you can use an INNER JOIN with a USING clause instead. This requires no
-- table qualifiers and names each joined column only once:

SELECT * 
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting
    USING (a_id)
ORDER BY a_id;

-- For SELECT * queries, the USING form produces a result that differs from the ON form: it
-- returns only one instance of each join column, so a_id appears once, not twice.

---------------------------------------
-- As a rule of thumb, it’s conventional to use ON or USING
-- to specify how to join the tables, and the WHERE clause to  
-- restrict which of the joined rows to select.
SELECT 
  * 
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting
    ON artist.a_id = painting.a_id
WHERE 
  painting.state = 'KY';


---------------------------------------
-- Joins can use more than two tables.
SELECT 
  artist.name, painting.title, states.name, painting.price
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting 
  INNER JOIN cookbook.states
ON 
  artist.a_id = painting.a_id 
  AND painting.state = states.abbrev
WHERE 
  painting.state = 'KY';


---------------------------------------
-- how many paintings you have per artist:
SELECT 
  artist.name, 
  COUNT(*) AS 'number of paintings'
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting 
    ON artist.a_id = painting.a_id
GROUP BY 
  artist.name;


---------------------------------------
-- how much you paid for each artist’s paintings, in total and on average:
SELECT 
  artist.name,
  COUNT(*) AS 'number of paintings',
  SUM(painting.price) AS 'total price',
  AVG(painting.price) AS 'average price'
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;

---------------------------------------
-- To summarize all artists, including those for whom you have no paintings,
-- you must use a different kind of join—specifically, an outer join.

-- Inner joins are useful for identifying matches, and 
-- outer joins are useful for identifying mismatches


---------------------------------------
-- OUTER JOIN
-- to find rows in one table that have no match in another:
-- use an outer join (a LEFT JOIN or a RIGHT JOIN) or a NOT IN subquery.

-- Like inner joins, an outer join finds matches between tables. 
-- But unlike an inner join, an outer join also determines
-- which rows in one table have no match in another.
-- Two types of outer join are LEFT JOIN and RIGHT JOIN.

-- The strategy for using OUTER JOIN to find missing values from one table in another:
-- if the left table row has no match in the right table, a LEFT JOIN still produces a row — one in which
-- all the columns from the right table are set to NULL. This means you can find values that
-- are missing from the right table by looking for NULL.

SELECT 
  * 
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting
    ON artist.a_id = painting.a_id
WHERE 
  painting.a_id IS NULL;


-- To show only the artist table values that are missing from the painting table,
-- write the output column list to name only columns from the artist table.
SELECT 
  artist.* 
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting
ON 
  artist.a_id = painting.a_id
WHERE 
  painting.a_id IS NULL;


---------------------------------------
-- To report each left-table value along with an indicator as to
-- whether it’s present in the right table, perform the following:

-- (a) perform a LEFT JOIN 
-- (b) Use the COUNT() function on one of the columns of the the right table 
--     to count the number of times each left-table value occurs in the right table. 
-- (c) Use an indicator calculated value to indicate where a match does not exist.
SELECT 
  artist.name,
  COUNT(painting.a_id) as cnt,
  IF(COUNT(painting.a_id)>0,'yes','no') AS 'in collection?'
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;


---------------------------------------
-- A RIGHT JOIN is an outer join that is like LEFT JOIN but reverses the roles of the left and right tables
-- Syntactically, tbl1 LEFT JOIN tbl2 is equivalent to tbl2 RIGHT JOIN tbl1.


---------------------------------------
-- Another way to identify values present in one table but missing from another is to use a NOT IN subquery.
-- This method does not need a join.
SELECT 
  * 
FROM 
  cookbook.artist
WHERE 
  a_id NOT IN (SELECT a_id FROM cookbook.painting);


---------------------------------------
-- to determine whether there are records in either dataset that are “unattached” (not matched by any
-- record in the other dataset), and perhaps remove them if so.
-- To identify unmatched values in each table, use a LEFT JOIN or a NOT IN subquery. 
-- To remove them, use DELETE with a NOT IN subquery.

-- Mismatch identification is a matter of using outer joins.

-- TWO IMPORTANT POINTS REGARING THE WHERE STATEMENT:
-- (a) since the columns in two tables have the same name (region_id), 
-- we need to be explicit in naming the column in WHERE statement. 
-- (b) To find missing values, use 'IS NULL', instead of '= NULL'

-- ABOUT THE ORDER OF TABLES IN A LEFT JOIN TABLE:
-- If you are looking for missing values in a table, put that table on the opposite side of the join statement.
-- to find missing values in table1, use table2 LEFT JOIN table1, and use table2 in the SELECT statement.
-- to find missing values in table2, use table1 LEFT JOIN table2, and use table1 in the SELECT statement.
-- to represent the missing values usign a column from table1, use table1 LEFT JOIN table2.
-- to represent the missing values usign a column from table2, use table2 LEFT JOIN table1.


-- Example: To find sales regions for which there are no sales volume rows.
-- Thinking process: 
-- We want to find sales regions --> use sales_region table on the left side of the LEFT JOIN.
-- We want to find those rows without sales volume --> put sales_volume on the right side.
-- We are looking for sales region --> put sales_region in the SELECT statement.
SELECT 
  cookbook.sales_region.*
FROM
  cookbook.sales_region
  LEFT JOIN cookbook.sales_volume
ON 
  cookbook.sales_volume.region_id = cookbook.sales_region.region_id
WHERE
  cookbook.sales_volume.region_id IS NULL;


-- Example: conversely, to find sales volume rows that are not associated with any known region
-- Thinking process for this query:
-- We need to find sales volumes --> put that on the left side of the LEFT JOIN.
-- We want those that are not associated with any region --> put region on the right side.
-- We want region IDs --> use region_id in the SELECT statement.
SELECT
  cookbook.sales_volume.region_id
FROM
  cookbook.sales_volume
  LEFT JOIN cookbook.sales_region
ON
  cookbook.sales_volume.region_id = cookbook.sales_region.region_id
WHERE
  cookbook.sales_region.region_id IS NULL
;


-- Use DISTINCT to remove duplicates
SELECT
  DISTINCT cookbook.sales_volume.region_id AS 'Unmatched volume row IDs'
FROM
  cookbook.sales_volume
  LEFT JOIN cookbook.sales_region
ON
  cookbook.sales_volume.region_id = cookbook.sales_region.region_id
WHERE
  cookbook.sales_region.region_id IS NULL
;


-- Alternatviely, use a NOT IN statement with a subquery:
SELECT 
  region_id
FROM 
  cookbook.sales_volume
WHERE
  region_id NOT IN (SELECT region_id FROM cookbook.sales_region);

  
---------------------------------------
-- To get rid of unmatched rows, use a NOT IN subquery in a DELETE statement.
-- For example, to remove sales_region rows that match no sales_volume rows,
DELETE 
FROM 
  cookbook.sales_region
WHERE 
  region_id NOT IN (SELECT region_id FROM cookbook.sales_volume);

-- To remove sales_region rows that match no sales_volume rows,
DELETE 
FROM 
  cookbook.sales_volume
WHERE 
  region_id NOT IN (SELECT region_id FROM cookbook.sales_region);


-------------------------------------------------------------------------------
-- COMPARING A TABLE TO ITSELF
-- Example: to compare rows in a table to other rows in the same table.
-- Self-joins are needed when we want to know which pairs of rows in a table 
-- satisfy some condition, and the condition is based on the values of the table itself.
-- In a normal join, the conditions is either external (e.g., a constant value), or
-- from another table. In self-join, the condition comes from the values within the table itself.
-- It is the same as a join, with one detail: we have to use table aliases to refer to the correct table when comparing.

SELECT a_id, title FROM cookbook.painting ORDER BY a_id;

-- Example: find all rows in the table that have the same artist as the 'The Potato Eaters'
SELECT 
  p2.title
FROM 
  cookbook.painting AS p1 
  INNER JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id
WHERE 
  p1.title = 'The Potato Eaters';


-- Note that in the results of the self-join, the output includes the reference value itself. 
-- That makes sense: after all, the reference matches itself. 
-- To find only other paintings by the same artist, explicitly exclude the reference value from the output:
SELECT 
  p2.title
FROM 
  cookbook.painting AS p1 
  INNER JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id
WHERE 
  p1.title = 'The Potato Eaters' 
  AND p2.title <> p1.title;


-- Note that if you just use the title in the WHERE statement, you will not get all the rows with the 
-- desired title. You need to self-join the table to get that. The WHERE statement constraint will
-- only return the first row that matches the WHERE statement constraint.
SELECT 
  *
FROM 
  cookbook.painting AS p1 
WHERE
  p1.title = 'The Potato Eaters';
  
-- Or, we can use a sub-query to achieve that too.
SELECT
  p1.title 
FROM 
  cookbook.painting p1
WHERE p1.a_id = (SELECT p2.a_id FROM cookbook.painting p2 WHERE p2.title = 'The Potato Eaters');


---------------------------------------
-- Which states joined the Union in the same year as New York?
-- Perform a temporal pairwise comparison based on the year part of the dates in the table’s statehood column.
-- Side note: Note how we used a function in 'ON' statement of the JOIN.
SELECT 
  s2.name, 
  s2.statehood
FROM 
  cookbook.states AS s1 
  INNER JOIN cookbook.states AS s2
ON 
  YEAR(s1.statehood) = YEAR(s2.statehood) 
  AND s1.name <> s2.name
WHERE 
  s1.name = 'New York'
ORDER BY 
  s2.name;


---------------------------------------
-- Find every pair of states that joined the Union in the same year.
SELECT 
  YEAR(s1.statehood) AS year,
  s1.name AS name1, 
  s1.statehood AS statehood1,
  s2.name AS name2, 
  s2.statehood AS statehood2
FROM 
  cookbook.states AS s1 
  INNER JOIN cookbook.states AS s2
ON 
  YEAR(s1.statehood) = YEAR(s2.statehood) 
  AND s1.name <> s2.name  -- prevent redundant rows (joining a row to itself).
ORDER BY 
  year, 
  name1, 
  name2;

-- In the results of the query above, we still have duplicate pairs:
-- pairs that containt the same two states, but in different orders.
-- To remove those duplicates, we can use '<' rather than '<>'
-- The one-way inequality selects only those rows in which the first state name is lexically
-- less than the second, and eliminates rows in which the names appear in opposite order
-- (as well as rows in which the state names are identical).
SELECT 
  YEAR(s1.statehood) AS year,
  s1.name AS name1, 
  s1.statehood AS statehood1,
  s2.name AS name2, 
  s2.statehood AS statehood2
FROM 
  cookbook.states AS s1 
  INNER JOIN cookbook.states AS s2
ON 
  YEAR(s1.statehood) = YEAR(s2.statehood) 
  AND s1.name < s2.name
ORDER BY 
  year, 
  name1, 
  name2;


-------------------------------------------------------------------------------
-- For self-join problems of the type “Which values are not matched by other rows in the table?”, 
-- use a LEFT JOIN rather than an INNER JOIN

-- Example: “Which states did not join the Union in the same year as any other state?"
-- For each row in the states table, the statement selects rows for which the state has a
-- statehood value in the same year, not including that state itself. 
-- For rows having no such match, the LEFT JOIN forces the output to contain a row anyway, 
-- with all the s2 columns set to NULL. 
-- Those rows identify the states with no other state that joined the Union in the same year.

-- Thought process for this query:
-- We are looking for rows that do not match any other row --> OUTER JOIN
-- We are looking within the table iteself --> SELF-JOIN
-- Therefore, we do a SELF OUTER JOIN.

SELECT
  DISTINCT s1.name
FROM
  cookbook.states AS s1
  LEFT JOIN cookbook.states AS s2  -- left join against itself to find differences
ON
  YEAR(s1.statehood) = YEAR(s2.statehood)  -- joining on the year. When the year of a row is not found on the table itself, the second year value on the row will be reported as NULL.
  AND s1.name <> s2.name  -- to prevent duplicate self-joins (joining a row with itself)
WHERE 
  s2.statehood IS NULL  -- finding differences 
ORDER BY
  s1.name
;


-------------------------------------------------------------------------------
-- PRODUCING MASTER-DETAIL LISTS AND SUMMARIES
-- Example: Two tables have a master-detail relationship, 
-- and you want to produce a list that shows each master row with its detail rows or 
-- a list that produces a summary of the detail rows for each master row.
-- This is a one-to-many relationship. The solution to this problem involves a join.

-- Example: Find the number of paitings that each artist has drawn.
SELECT 
  artist.name, 
  COUNT(painting.a_id) AS paintings
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;
  
  
-- Note: use COUNT(col_name); do not use COUNT(*). 
-- COUNT(col_name) does not include NULL rows; but COUNT(*) also includes NULL values.
SELECT 
  artist.name, 
  COUNT(*) AS paintings
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;


---------------------------------------
-- Calculate the total and average prices of the paintings for each artist in the artist table.
SELECT 
  artist.name,
  COUNT(painting.a_id) AS 'number of paintings',
  SUM(painting.price) AS 'total price',
  AVG(painting.price) AS 'average price'
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;


-- Note that COUNT(expr) will only include non-null values by default.
-- However, all other aggregate functions (including SUM and AVG) will
-- return NULL when they include NULL values. 
-- To return 0 instead of NULL in such situations, use the idion IFNULL(SUM(expr),0).
SELECT 
  artist.name,
  COUNT(painting.a_id) AS 'number of paintings',
  IFNULL(SUM(painting.price),0) AS 'total price',
  IFNULL(AVG(painting.price),0) AS 'average price'
FROM 
  cookbook.artist 
  LEFT JOIN cookbook.painting 
ON 
  artist.a_id = painting.a_id
GROUP BY 
  artist.name;


-------------------------------------------------------------------------------
-- MANY TO MANY RELATIONSHIPS
-- occurs when a row in one table may have many matches in the other, and vice versa.

-- A many-to-many relationship requires a third table for associating the two
-- primary tables and a three-way join to produce the correspondences between them.

SELECT 
  * 
FROM 
  cookbook.movies_actors 
ORDER BY 
  year, movie, actor;


-- Table normalization explained in an example:
-- To better represent this many-to-many relationship, use multiple tables:
-- Store each movie year and name once in a table named movies.
-- Store each actor name once in a table named actors.
-- Create a third table, movies_actors_link, that stores movie-actor associations. 
-- To minimize the information stored in this table, 
-- assign unique IDs to each movie and actor within their
-- respective tables, and store only those IDs in the movies_actors_link table

SELECT * FROM cookbook.movies ORDER BY id;
SELECT * FROM cookbook.actors ORDER BY id;
SELECT * FROM cookbook.movies_actors_link ORDER BY movie_id, actor_id;

-- Example: List all the pairings that show each movie and who acted in it.
SELECT 
  m.year, m.movie, a.actor
FROM 
  cookbook.movies AS m 
  INNER JOIN cookbook.movies_actors_link AS l
  INNER JOIN cookbook.actors AS a
ON 
  m.id = l.movie_id 
  AND a.id = l.actor_id
ORDER BY 
  m.year, m.movie, a.actor;


---------------------------------------
---------------------------------------
---------------------------------------
-- GROUP-WIDE CENTRAL VALUE vs INDIVIDUAL ROWS 
-- i.e., COMPARING ROWS AT DIFFERENT LEVELS OF DETAIL
-- Example: which row within each group of rows in a table contains the maximum
-- or minimum value for a given column.

-- With SQL, it is not possible to directly compare rows from different levels of detail against each other. 
-- Generally, to compare rows at different levels of detail, there a number of strategies:
-- (a) Create a user-defined variable (if we have a single summary value that we want to compare against multiple rows of a table)
-- (b) Create a temporary table (works even if we have multiple summary values that we want to compare agains multiple other values)
-- (c) Use a sub-query

-- The following examples show these strategies in action.
-- Example: What is the most expensive painting in the collection, and who painted it?

-------------------
-- (Strategy a) Store the summary value in a user-defined variable, 
-- then use the variable to identify the row containing that value.
SELECT @max_price := (SELECT MAX(price) FROM cookbook.painting);

SELECT 
  artist.name, 
  painting.title, 
  painting.price
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting
ON 
  painting.a_id = artist.a_id
WHERE 
  painting.price = @max_price;


-------------------
-- (Strategy b) If you prefer a single-query solution, use a subquery in the FROM clause rather than a temporary table.
SELECT 
  artist.name, 
  painting.title, 
  painting.price
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting
ON 
  painting.a_id = artist.a_id
WHERE 
  painting.price = (SELECT MAX(price) FROM cookbook.painting);


-------------------
-- (Strategy c): Create a temporary table to hold the per-group maximum or minimum values, then
-- join the temporary table with the original one to pull out the matching row for each group. 

DROP TABLE cookbook.tmp;
-- Or, use a temp table
CREATE TABLE cookbook.tmp
SELECT a_id, MAX(price) AS max_price 
FROM cookbook.painting GROUP BY a_id;

SELECT 
  artist.name, painting.title, painting.price
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting 
  INNER JOIN cookbook.tmp
ON 
  painting.a_id = artist.a_id
AND painting.a_id = tmp.a_id
AND painting.price = tmp.max_price;


--------------------------------------
-- If we want to compare the rows against multiple summary stats, 
-- then we need to either use a temp table or a sub-query (user-defined var will not work).
-- To obtain the same result with a single statement, use a subquery in the FROM clause.
-- The sub-query should calculate the summary values of interest.
-- Then, join the sub-query with the original table and find matching rows from the original table based on criteria of interest.
SELECT 
  artist.name, painting.title, painting.price
FROM 
  cookbook.artist 
  INNER JOIN cookbook.painting 
  INNER JOIN
    (SELECT a_id, MAX(price) AS max_price 
    FROM cookbook.painting 
    GROUP BY a_id) AS tmp
ON 
  painting.a_id = artist.a_id
  AND painting.a_id = tmp.a_id
  AND painting.price = tmp.max_price;


--------------------------------------
-- Yet another way to answer maximum-per-group questions is to use a LEFT JOIN that joins a table to itself.

-- This statement finds the highest-priced painting per artist ID.
-- In this case, 'IS NULL' clause selects all the rows from p1 for which there is no row in p2 with a higher price).
SELECT 
  p1.a_id, p1.title, p1.price
FROM 
  cookbook.painting AS p1 
  LEFT JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id 
  AND p1.price < p2.price
WHERE 
  p2.a_id IS NULL;


-- Lets break down this query step-by-step:
-- (step 1) First of all, since we want to find a central value 'per artist', we join the table agains itself, using artist ID as the matching criteria.
-- In this way, each row of the result will contain a pair of the rows of the original table, with the same artist.
SELECT 
  *
FROM 
  cookbook.painting AS p1 
  LEFT JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id;

-- (step 2) Then, we want to make a pair-wise comparison of the prices of paintings of each artist ID, so
-- we add a second constraint to the LEFT JOIN, which compares the prices of each pair of the paintings.
SELECT 
  *
FROM 
  cookbook.painting AS p1 
  LEFT JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id
  AND p1.price < p2.price;

-- (step 3) 
-- Since we used a LEFT JOIN, when the constraint is not satisfied, the columns for the second table will be NULL.
-- These values are the values we are looking for: they are the rows where the first painting price is greater than the second.
-- Therefore, we add a WHERE constraint to pick those rows with NULL values.
SELECT 
  p1.a_id, p1.title, p1.price
FROM 
  cookbook.painting AS p1 
  LEFT JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id 
  AND p1.price < p2.price
WHERE 
  p2.a_id IS NULL;


-------------------
-- To display artist names rather than ID values, join the result of the LEFT JOIN to the artist table
SELECT 
  artist.name, p1.title, p1.price
FROM 
  cookbook.painting AS p1 
  LEFT JOIN cookbook.painting AS p2
ON 
  p1.a_id = p2.a_id 
  AND p1.price < p2.price
INNER JOIN cookbook.artist 
  ON p1.a_id = artist.a_id
WHERE 
  p2.a_id IS NULL;


---------------------------------------
---------------------------------------
---------------------------------------
-- FIND HOLES IN A LIST
-- to determine how many drivers were on the road each day
SELECT * FROM cookbook.driver_log ORDER BY rec_id;

SELECT
  trav_date,
  COUNT(name)
FROM
  cookbook.driver_log
GROUP BY
  trav_date
ORDER BY
  trav_date;


-- To produce a summary that includes all categories (all dates within the date range represented in the table), 
-- including those for which no driver was active, create a reference table that lists each date:
CREATE TABLE cookbook.dates (d DATE);

INSERT INTO cookbook.dates (d)
VALUES('2014-07-26'),('2014-07-27'),('2014-07-28'),
  ('2014-07-29'),('2014-07-30'),('2014-07-31'),
  ('2014-08-01'),('2014-08-02');

SELECT * FROM cookbook.dates;

-- Then join the reference table to the driver_log table using a LEFT JOIN.
SELECT 
  dates.d, 
  COUNT(driver_log.trav_date) AS drivers
FROM 
  cookbook.dates 
  LEFT JOIN cookbook.driver_log 
ON 
  dates.d = driver_log.trav_date
GROUP BY 
  d 
ORDER BY 
  d;


---------------------------------------
-- use the reference table to detect holes in the dataset
-- Remember that when you do a COUNT(), it skips over NULLs. So all those 0 values on the table above are NULL values.
SELECT 
  dates.d
FROM 
  cookbook.dates 
  LEFT JOIN cookbook.driver_log 
ON 
  dates.d = driver_log.trav_date
WHERE 
  driver_log.trav_date IS NULL;

---------------------------------------

---------------------------------------
-- SORT QUERY RESULTS BY A VALUE THAT IS NOT DIRECTLY ACCESSIBLE THROUGH COLUMNS.
-- Sometimes the values you want to sort by aren’t present in the rows to be sorted.
-- For example, this happens when you want to use group characteristics to order the rows.
-- I.e., when you want to sort the results on the basis of a summary value not present in the rows.

-- You cant do this with a summary query because then you wouldn’t get back the individual driver rows.
-- But you can’t do it without a summary query, either, because the summary values are required for sorting.

-- In such cases, derive the ordering information and store it in an auxiliary table. Then join the original
-- table to the auxiliary table, using the auxiliary table to control the sort order.

-- As is the case with temp tables, to avoid defining a temp table we can also use a subquery.


-- Example: we want to order the driver_log table based on total miles driven by each driver.
-- We need to create a temp table that stores the total miles driven by each dirver.
DROP TABLE cookbook.tmp;

CREATE TABLE cookbook.tmp
SELECT 
  name, 
  SUM(miles) AS driver_miles 
FROM 
  cookbook.driver_log 
GROUP BY 
  name;

SELECT * FROM cookbook.tmp ORDER BY driver_miles DESC;

SELECT 
  tmp.driver_miles, driver_log.*
FROM 
  cookbook.driver_log 
  INNER JOIN cookbook.tmp 
ON 
  cookbook.driver_log.name = cookbook.tmp.name
ORDER BY 
  tmp.driver_miles DESC, driver_log.trav_date;


-- To avoid using the temporary table, select the same rows using a subquery in the FROM clause:
SELECT 
  tmp.driver_miles, driver_log.*
FROM 
  cookbook.driver_log 
  INNER JOIN
    (SELECT name, SUM(miles) AS driver_miles
    FROM cookbook.driver_log GROUP BY name) AS tmp
ON 
  driver_log.name = tmp.name
ORDER BY 
  tmp.driver_miles DESC, 
  driver_log.trav_date;









