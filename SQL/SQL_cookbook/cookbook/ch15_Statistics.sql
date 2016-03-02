
-- techniques for calculating summary statistics;
-- generating frequency distributions, counting missing values, and calculating leastsquares regressions or correlation coefficients
-- using both internal functions and a collection of joins and low-level calcs
-- Randomization techniques
-- successive-observation differences, cumulative sums, and running averages.
-- rank assignments and generating team standings

-- A good first step in any data analysis job is to create some summarys statistics.
-- These summary statistics usually fall within the following categories:
-- Quantity measures such as the number of observations, thei sum, their range
-- Measures of central

-- OTHER RESOURCES
-- MySQL Math functions reference: https://dev.mysql.com/doc/refman/5.0/en/mathematical-functions.html

-- NOTE: in the cookbook, to assign a value to a variable, some times this syntax has been used: 'SET @VAR = ...' but in my case, only the format Ive used here works: 'SELECT @var := ...'

-------------------------------------------------------------------------------
SELECT subject, age, sex, score FROM cookbook.testscore ORDER BY subject;

-- Aside from the median and mode, all others have functions associated with them in MySQL. Oracle has median function too.
SELECT 
  COUNT(score) AS n,
  SUM(score) AS sum,
  MIN(score) AS minimum,
  MAX(score) AS maximum,
  AVG(score) AS mean,
  STDDEV_SAMP(score) AS 'sample std. dev.',  -- divides by (n-1)
  VAR_SAMP(score) AS 'sample variance',  -- divides by (n-1)
  STDDEV_POP(score) AS 'pop std. dev.',  -- divides by (n)
  VAR_POP(score) AS 'pop variance',  -- divides by (n)
  STDDEV(score) AS 'same as pop std. dev.',  -- synonym for STDDEV_POP
  VARIANCE(score) AS 'same as pop variance'  -- synonym for VAR_POP
FROM 
  cookbook.testscore;

-------------------------------------------------------------------------------
-- to select values that lie more than three standard deviations from the mean
SELECT @mean := AVG(score), @std := STDDEV_SAMP(score) FROM cookbook.testscore;
SELECT score FROM cookbook.testscore WHERE ABS(score-@mean) > @std * 1;


-- To determine the mode
SELECT 
  score, 
  COUNT(score) AS frequency
FROM 
  cookbook.testscore 
GROUP BY 
  score 
ORDER BY 
  frequency DESC;



---------------------------------------
-- The median of a set of ordered values
SELECT @obs_ct := COUNT(score) FROM cookbook.testscore;
SELECT @obs_ct FROM cookbook.testscore;

SELECT 
  @mid_value := (CASE WHEN MOD(@obs_ct,2)=0 THEN (FLOOR(@obs_ct/2)-1+(FLOOR(@obs_ct/2)+1)) ELSE (FLOOR(@obs_ct/2)+1) END) 
FROM 
  score_sorted; 

SELECT IF(MOD(@obs_ct,2)=0, @obs_ct/2, @obs_ct/2) FROM cookbook.testscore;
SELECT FLOOR(@obs_ct/2) FROM cookbook.testscore;


---------------------------------------
-- Query to calculate median
SELECT 
  avg(t1.score) as median_val 
FROM 

  (SELECT 
    @rownum:=@rownum+1 as `row_number`, 
    score
  FROM 
    cookbook.testscore,  
    (SELECT @rownum:=0) r
    WHERE 1
      -- put some where clause here
    ORDER BY score
  ) as t1, 
  
  (SELECT 
    count(*) as total_rows
  FROM 
    cookbook.testscore d
  WHERE 1
    -- put same where clause here
  ) as t2

WHERE 1
  AND t1.row_number in ( floor((total_rows+1)/2), floor((total_rows+2)/2) );


---------------------------------------
-- Average test scores by age
SELECT 
  age, COUNT(score) AS n,
  SUM(score) AS sum,
  MIN(score) AS minimum,
  MAX(score) AS maximum,
  AVG(score) AS mean,
  STDDEV_SAMP(score) AS 'std. dev.',
  VAR_SAMP(score) AS 'variance'
FROM 
  cookbook.testscore
GROUP BY 
  age;


---------------------------------------
-- To show each count as a percentage of the total, use one query to get the total
-- number of observations and another to calculate the percentages for each group.

SELECT @n := (SELECT COUNT(score) FROM cookbook.testscore);

SELECT 
  score, (COUNT(score)*100)/@n AS percent
FROM 
  cookbook.testscore 
GROUP BY 
  score;
  
-- Or, use a sub-query instead of a variable:
SELECT 
  score, 
  (COUNT(score)*100) / (SELECT COUNT(score) FROM cookbook.testscore) AS percent
FROM 
  cookbook.testscore 
GROUP BY 
  score;

-- To display an ASCII bar chart of the test score counts, convert the counts to strings of * characters:
SELECT 
  score, REPEAT('*',COUNT(score)) AS 'count histogram'
FROM 
  cookbook.testscore 
GROUP BY 
  score;


-- To force each category to be displayed, use a reference table and a LEFT JOIN
CREATE TABLE cookbook.ref (score INT);

INSERT INTO cookbook.ref (score)
VALUES(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10);

SELECT 
  ref.score, COUNT(testscore.score) AS counts,
  REPEAT('*',COUNT(testscore.score)) AS 'count histogram'
FROM 
  cookbook.ref 
  LEFT JOIN cookbook.testscore 
    ON ref.score = testscore.score
GROUP BY 
  ref.score;
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- To find the number of missing values,
-- count the number of NULL values in the set.
-- COUNT(*) counts the total number of rows, and COUNT(score) counts the number of
-- nonmissing scores. The difference between the two values is the number of missing scores.

SELECT subject, score FROM cookbook.testscore ORDER BY subject;

SELECT 
  COUNT(*) AS 'n (total)',
  COUNT(score) AS 'n (nonmissing)',
  COUNT(*) - COUNT(score) AS 'n (missing)',
  ((COUNT(*) - COUNT(score)) * 100) / COUNT(*) AS '% missing'
FROM 
  cookbook.testscore;


-- The alternative way to do this is by using SUM(ISNULL(score)) 
-- The ISNULL() function returns 1 if its argument is NULL, zero otherwise.
SELECT 
  COUNT(*) AS 'n (total)',
  COUNT(score) AS 'n (nonmissing)',
  SUM(ISNULL(score)) AS 'n (missing)',
  (SUM(ISNULL(score)) * 100) / COUNT(*) AS '% missing'
FROM 
  cookbook.testscore;


-- To produce a summary for each combination of conditions, use a GROUP BY clause.
SELECT 
  age, sex,
  COUNT(*) AS 'n (total)',
  COUNT(score) AS 'n (nonmissing)',
  SUM(ISNULL(score)) AS 'n (missing)',
  (SUM(ISNULL(score)) * 100) / COUNT(*) AS '% missing'
FROM 
  cookbook.testscore
GROUP BY
  age, sex;
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- CALCULATING LIEAR REGRESSION OR CORRELATION COEFFICIENTS
-- We need to calculate the summary stats that will be used in finding the linear regression coefficient:
-- These include the number of observations; the means, sums, and sums of
-- squares for each variable; and the sum of the products of each variable.
SELECT
  @n := COUNT(score) AS N,
  @meanX := AVG(age) AS 'X mean',
  @sumX := SUM(age) AS 'X sum',
  @sumXX := SUM(age*age) AS 'X sum of squares',
  @meanY := AVG(score) AS 'Y mean',
  @sumY := SUM(score) AS 'Y sum',
  @sumYY := SUM(score*score) AS 'Y sum of squares',
  @sumXY := SUM(age*score) AS 'X*Y sum'
FROM 
  cookbook.testscore;

-- Calculate and report the coefficient formula
SELECT @b := (@n*@sumXY - @sumX*@sumY) / (@n*@sumXX - @sumX*@sumX);
SELECT @a := (@meanY - @b*@meanX);
SELECT @b AS slope, @a AS intercept;
SELECT CONCAT('Y = ',@b,'X + ',@a) AS 'least-squares regression';

-- Calculate and report the correlation coefficient
SELECT
(@n*@sumXY - @sumX*@sumY)
  / SQRT((@n*@sumXX - @sumX*@sumX) * (@n*@sumYY - @sumY*@sumY))
AS correlation;
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- GENERATING RANDOM NUMBERS
-- MySQL has a RAND() function that produces random numbers between 0 and 1.
-- You can provide a seed value to the random number generator.
SELECT RAND(), RAND(), RAND(), RAND(1), RAND(1), RAND(1);

-- A random number between 0 and 10:
SELECT FLOOR(RAND()*10)+1;

---------------------------------------
-- RANDOMIZING A SET OF ROWS
-- Use ORDER BY RAND().
SELECT subject, age, sex, score FROM cookbook.testscore ORDER BY RAND();

-- Some real-world applications for randomizing a set of rows:
-- Determining the starting order for participants in an event
-- Shuffling a deck of cards
-- etc.

---------------------------------------
-- SELECTING RANDOM ITEMS FROM A SET OF ROWS
-- Solution: Randomize the values, then pick the first one

SELECT n FROM cookbook.die ORDER BY RAND() LIMIT 1;
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
-- CALCULATING SUCCESSIVE ROW DIFFERENCES
-- Solution: Use a self-join that matches pairs of adjacent rows and calculates the differences between members of each pair.
-- When you want to compare across different rows of a table, you most likely will need a self-join.

SELECT seq, city, miles FROM cookbook.trip_log ORDER BY seq;


-- Suppose we want to know the distance travelled between two cities by subtracting cumulative milage reads from travel logs.
SELECT 
  t1.seq AS seq1, t2.seq AS seq2,
  t1.city AS city1, t2.city AS city2,
  t1.miles AS miles1, t2.miles AS miles2,
  t2.miles - t1.miles AS dist
FROM 
  cookbook.trip_log AS t1 
  INNER JOIN cookbook.trip_log AS t2
    ON t1.seq+1 = t2.seq
ORDER BY 
  t1.seq;

-- NOTE that to perform relative-difference calculations using a table of absolute or cumulative values, 
-- it must include a sequence column that has no gaps. 
-- If the table contains a sequence column but there are gaps, renumber it.


---------------------------------------
-- The table below provides a general idea of how the player’s hitting performance changed over the course of the season.
SELECT 
  id, date, ab, h, 
  TRUNCATE(IFNULL(h/ab,0),3) AS ba
FROM 
  cookbook.player_stats 
ORDER BY 
  id;


-- Now, to get incremental improvement (or regress) of the player over time at each month:
SELECT
  t1.id AS id1, t2.id AS id2,
  t2.date,
  t1.ab AS ab1, t2.ab AS ab2,
  t1.h AS h1, t2.h AS h2,
  t2.ab-t1.ab AS abdiff,
  t2.h-t1.h AS hdiff,
  TRUNCATE(IFNULL((t2.h-t1.h)/(t2.ab-t1.ab),0),3) AS ba
FROM 
  cookbook.player_stats AS t1 
  INNER JOIN cookbook.player_stats AS t2
    ON t1.id+1 = t2.id
ORDER BY 
  t1.id;


-------------------------------------------------------------------------------
-- FINDING CUMULATIVE SUMS AND RUNNING AVERAGES
-- Use a self-join to produce the sets of successive observations at each measurement point,
-- then apply aggregate functions to each set of values to compute its sum or average.

-- This is the opposite problem from above: instead of finding incremental change from cumulative values,
-- we want to find the cumulative pattern from interval data.

SELECT date, precip FROM cookbook.rainfall ORDER BY date;

-- To calculate cumulative rainfall for a given day, add that day’s precipitation value to the
-- values for all the previous days.
-- To implement it, use one instance of the rainfall table as a reference, and determine for the date
-- in each row the sum of the precip values in all rows occurring up through that date in
-- another instance of the table.

SELECT 
  t1.date, t1.precip AS 'daily precip',
  SUM(t2.precip) AS 'cum. precip'
FROM 
  cookbook.rainfall AS t1 
  INNER JOIN cookbook.rainfall AS t2
    ON t1.date >= t2.date
GROUP BY 
  t1.date;
  
  
-- display the number of days elapsed at each date, as well
-- as the running averages for amount of precipitation each day
SELECT 
  t1.date, 
  t1.precip AS 'daily precip',
  SUM(t2.precip) AS 'cum. precip',
  COUNT(t2.precip) AS 'days elapsed',  -- assuming we have data on every day between the two dates
  AVG(t2.precip) AS 'avg. precip'  -- assuming we have data on every day between the two dates
FROM 
  cookbook.rainfall AS t1 
  INNER JOIN cookbook.rainfall AS t2
    ON t1.date >= t2.date
GROUP BY 
  t1.date;  


-- If we have missing days between the two dates, use MySQL function:
-- Take the minimum and maximum date involved in each sum, and 
-- calculate a days-elapsed value from them.
-- DATEDIFF(MAX(t2.date),MIN(t2.date)) + 1
SELECT 
  t1.date, t1.precip AS 'daily precip',
  SUM(t2.precip) AS 'cum. precip',
  DATEDIFF(MAX(t2.date),MIN(t2.date)) + 1 AS 'days elapsed',
  SUM(t2.precip) / (DATEDIFF(MAX(t2.date),MIN(t2.date)) + 1) AS 'avg. precip'
FROM 
  cookbook.rainfall AS t1 
  INNER JOIN cookbook.rainfall AS t2
    ON t1.date >= t2.date
GROUP BY 
  t1.date;


-------------------------------------------------------------------------------
-- The values in the following table are interval values, not cumulative.
SELECT stage, km, t FROM cookbook.marathon ORDER BY stage;

-- To calculate cumulative distance at each stage
SELECT 
  t1.stage, 
  t1.km, 
  SUM(t2.km) AS 'cum. km'
FROM 
  cookbook.marathon AS t1 
  INNER JOIN cookbook.marathon AS t2
    ON t1.stage >= t2.stage
GROUP BY 
  t1.stage;


-- To calculate accumulating time values: convert times to seconds,
-- total the resulting values, and convert the sum back to a time value.
SELECT t1.stage, t1.km, t1.t,
  SUM(t2.km) AS 'cum. km',
  SEC_TO_TIME(SUM(TIME_TO_SEC(t2.t))) AS 'cum. t',
  SUM(t2.km)/(SUM(TIME_TO_SEC(t2.t))/(60*60)) AS 'avg. km/hour'
FROM cookbook.marathon AS t1 
  INNER JOIN cookbook.marathon AS t2
    ON t1.stage >= t2.stage
GROUP BY 
  t1.stage;


-------------------------------------------------------------------------------
-- ASSIGNING RANKS
-- One type of ranking simply assigns each value its row number within the ordered set of values.
-- The following syntax creates a variable that advanced by 1 at any new row:
-- @rownum := @rownum + 1

SELECT @rownum := 0;

SELECT @rownum := @rownum + 1 AS rank, score
FROM cookbook.testscore ORDER BY score DESC;


-- That kind of ranking doesn’t take into account the possibility of ties (instances of values that are the same). 
-- A second ranking method does so by advancing the rank only when values change:
SELECT @rank := 0, @prev_val := NULL;

SELECT 
  @rank := IF(@prev_val=score,@rank,@rank+1) AS rank,
  @prev_val := score AS score
FROM 
  cookbook.testscore 
ORDER BY 
  score DESC;


-- A third ranking method is something of a combination of the other two methods. It
-- ranks values by row number, except when ties occur. In that case, the tied values each
-- get a rank equal to the row number of the first of the values.
SELECT @rownum := 0, @rank := 0, @prev_val := NULL;

SELECT 
  @rownum := @rownum + 1 AS row,
  @rank := IF(@prev_val<>score,@rownum,@rank) AS rank,
  @prev_val := score AS score
FROM 
  cookbook.testscore 
ORDER BY 
  score DESC;


--
-- This type of ranking is particularly used in sporting tables: 
SELECT name, wins FROM cookbook.al_winner ORDER BY wins DESC, name;


SELECT @rownum = 0, @rank = 0, @prev_val = NULL;

SELECT 
  @rownum := @rownum + 1 AS row,
  @rank := IF(@prev_val<>wins,@rownum,@rank) AS rank,
  name,
  @prev_val := wins AS wins
FROM 
  cookbook.al_winner 
ORDER BY 
  wins DESC;
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
-- COMPUTE TEAM STANDINGS
-- Standings for sports teams that compete against each other
-- is based on two values: wins and losses

-- Teams are ranked according to which has the best win-loss
-- record, and teams not in first place are assigned a “games-behind” value indicating how
-- many games out of first place they are.

SELECT team, wins, losses 
FROM cookbook.standings1
ORDER BY wins-losses DESC;

-- Determining the games-behind value is a little trickier. It’s based on the relationship of
-- the win-loss records for two teams, calculated as the average of two values:
--   How many more wins the first-place team has than the second-place team
--   How many fewer losses the first-place team has than the second-place team

-- The first-place team is the one with the largest win-loss differential. 
-- Find that value and save it in a variable, use this statement.
SELECT @wl_diff := (SELECT MAX(wins-losses) FROM cookbook.standings1);

SELECT 
  team, 
  wins AS W, losses AS L,
  wins/(wins+losses) AS PCT,
  (@wl_diff - (wins-losses)) / 2 AS GB
FROM 
  cookbook.standings1
ORDER BY 
  wins-losses DESC, PCT DESC;


-- Now we can add more dimensions to group the summaries by different variables.
-- ordered by season half, division, and win-loss differential:
SELECT half, division, team, wins, losses 
FROM cookbook.standings2
ORDER BY half, division, wins-losses DESC;

-- Generating the standings for these rows requires computing the GB values separately
-- for each of the four combinations of season half and division.
-- First, create a temp table that stores the winner per division and season-half:
CREATE TEMPORARY TABLE cookbook.firstplace
SELECT 
  half, division, 
  MAX(wins-losses) AS wl_diff
FROM 
  cookbook.standings2
GROUP BY 
  half, division;

SELECT * FROM cookbook.firstplace;

-- Then join the firstplace table to the original standings, associating each team record
-- with the proper win-loss differential to compute its GB value:
SELECT 
  wl.half, wl.division, 
  wl.team, 
  wl.wins AS W, wl.losses AS L,
  TRUNCATE(wl.wins/(wl.wins+wl.losses),3) AS PCT,
  IF(fp.wl_diff = wl.wins-wl.losses,
    '-',TRUNCATE((fp.wl_diff - (wl.wins-wl.losses)) / 2,1)) AS GB
FROM 
  cookbook.standings2 AS wl 
  INNER JOIN cookbook.firstplace AS fp
    ON wl.half = fp.half 
    AND wl.division = fp.division
ORDER BY 
  wl.half, wl.division, 
  wl.wins-wl.losses DESC, PCT DESC;









