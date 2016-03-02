###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: July 28, 2015
# summary: spark playground
# Source: https://spark.apache.org/docs/latest/api/python/pyspark.html
#TODO: re-visit this page, go through all examples, and add commentary on this page.
###############################################################################

###############################################################################
########################### LIBRARIES AND OPTIONS #############################
###############################################################################

###############################################################################
### GENERAL MODULES
import numpy as np
import pandas as pd

# to get ggplot-like style for plots
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot

import functools
import itertools
import os, sys


###############################################################################
### START PySpark
# Note: if PyCharm is able to read env vars, use os.environ.get(). Otherwise, hard-code the spark_home variable.
import os, sys
spark_home = os.environ.get('SPARK_HOME', None)  # '/usr/local/Cellar/apache-spark/1.4.1/libexec/'
spark_release_file = spark_home + "/RELEASE"
if os.path.exists(spark_release_file) and "Spark 1.4" in open(spark_release_file).read():
        pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
        if not "pyspark-shell" in pyspark_submit_args: pyspark_submit_args += " pyspark-shell"
        os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, "python/lib/py4j-0.8.2.1-src.zip"))
execfile(os.path.join(spark_home, "python/pyspark/shell.py"))
import pyspark
###############################################################################

###############################################################################
############################## MAIN PROCESS ###################################
###############################################################################

###############################################################################
### FROM: http://spark.apache.org/examples.html
# test spark connection
words = sc.parallelize(["scala","java","hadoop","spark","akka", "test", "test3"])
print words.count()

### SIMPLE TEXT PARSING
# Read a log file and see how many errors we got
text_file = sc.textFile("/Users/amirkavousian/Documents/Py_Codes/Tutorials_Files/Spark/Data/app_log.rtf")
errors = text_file.filter(lambda line: "ERROR" in line)
# Count all the errors
counts = errors.count()
print counts
# Count errors mentioning MySQL
errors.filter(lambda line: "MySQL" in line).count()
# Fetch the MySQL errors as an array of strings
errors.filter(lambda line: "MySQL" in line).collect()
counts.saveAsTextFile("/Users/amirkavousian/Documents/Py_Codes/Tutorials_Files/Spark/Data/app_log_counts_MySQL.rtf")

# If we want to work with the results multiple times, it makes sense to cache them in memory:
errors.cache()

# Use more transformations to build up a dataset:
text_file = sc.textFile("/Users/amirkavousian/Documents/Py_Codes/Data/app_log.rtf")
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
print counts
counts.saveAsTextFile("/Users/amirkavousian/Documents/Py_Codes/Data/app_log_counts.rtf")


### ESTIMATING Pi
# using the brute-force method
NUM_SAMPLES = 100000
def sample(p):
    x, y = np.random.rand(1), np.random.rand(1)
    return 1 if x*x + y*y < 1 else 0

count = sc.parallelize(xrange(0, NUM_SAMPLES)).map(sample) \
             .reduce(lambda a, b: a + b)
print "Pi is roughly %f" % (4.0 * count / NUM_SAMPLES)


### LOGISTIC REGRESSION
ITERATIONS = 1000
points = sc.textFile(...).map(parsePoint).cache()
w = np.random.ranf(size = D) # current separating plane
for i in range(ITERATIONS):
    gradient = points.map(
        lambda p: (1 / (1 + exp(-p.y*(w.dot(p.x)))) - 1) * p.y * p.x
    ).reduce(lambda a, b: a + b)
    w -= gradient
print "Final separating plane: %s" % w


### TO STOP Spark at any point:
sc.stop()


###############################################################################
### From: https://spark.apache.org/docs/latest/api/python/pyspark.html
from pyspark import SparkFiles
path = os.path.join(tempdir, "test.txt")
with open(path, "w") as testFile:
    _ = testFile.write("100")
    sc.addFile(path)
def func(iterator):
    with open(SparkFiles.get("test.txt")) as testFile:
        fileVal = int(testFile.readline())
        return [x * fileVal for x in iterator]
sc.parallelize([1, 2, 3, 4]).mapPartitions(func).collect()

sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()

sc.parallelize(xrange(0, 6, 2), 5).glom().collect()

tmpFile = NamedTemporaryFile(delete=True)
tmpFile.close()
sc.parallelize(range(10)).saveAsPickleFile(tmpFile.name, 5)
sorted(sc.pickleFile(tmpFile.name, 3).collect())

sc.range(5).collect()
sc.range(2, 4).collect()
sc.range(1, 7, 2).collect()

myRDD = sc.parallelize(range(6), 3)
sc.runJob(myRDD, lambda part: [x * x for x in part])

myRDD = sc.parallelize(range(6), 3)
sc.runJob(myRDD, lambda part: [x * x for x in part], [0, 2], True)


# The application can use SparkContext.cancelJobGroup to cancel all running jobs in this group.
import multiprocessing
from time import sleep
result = "Not Set"
lock = multiprocessing.Lock()
def map_func(x):
    sleep(100)
    raise Exception("Task should have been cancelled")
def start_job(x):
    global result
    try:
        sc.setJobGroup("job_to_cancel", "some description")
        result = sc.parallelize(range(x)).map(map_func).collect()
    except Exception as e:
        result = "Cancelled"
    lock.release()

def stop_job():
    sleep(5)
    sc.cancelJobGroup("job_to_cancel")

supress = lock.acquire()
supress = multiprocessing.Thread(target=start_job, args=(10,)).start()
supress = multiprocessing.Thread(target=stop_job).start()
supress = lock.acquire()
print(result)


path = os.path.join(tempdir, "sample-text.txt")
with open(path, "w") as testFile:
    _ = testFile.write("Hello world!")
textFile = sc.textFile(path)
textFile.collect()


path = os.path.join(tempdir, "union-text.txt")
with open(path, "w") as testFile:
    _ = testFile.write("Hello")
textFile = sc.textFile(path)
textFile.collect()
parallelized = sc.parallelize(["World!"])
sorted(sc.union([textFile, parallelized]).collect())


###
dirPath = os.path.join(tempdir, "files")
os.mkdir(dirPath)
with open(os.path.join(dirPath, "1.txt"), "w") as file1:
    _ = file1.write("1")
with open(os.path.join(dirPath, "2.txt"), "w") as file2:
    _ = file2.write("2")
textFiles = sc.wholeTextFiles(dirPath)
sorted(textFiles.collect())


### RDD
# Aggregate()
seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
sc.parallelize([1, 2, 3, 4]).aggregate((0, 0), seqOp, combOp)
sc.parallelize([]).aggregate((0, 0), seqOp, combOp)

# Cartesian()
rdd = sc.parallelize([1, 2])
sorted(rdd.cartesian(rdd).collect())

# Coalesce()
sc.parallelize([1, 2, 3, 4, 5], 3).glom().collect()
sc.parallelize([1, 2, 3, 4, 5], 3).coalesce(1).glom().collect()

# cogroup()
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2)])
[(x, tuple(map(list, y))) for x, y in sorted(list(x.cogroup(y).collect()))]

# collectAsMap()
m = sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
m[1]
m[3]

# combineByKey()
x = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
def f(x): return x
def add(a, b): return a + str(b)
sorted(x.combineByKey(str, add, add).collect())

# count()
sc.parallelize([2, 3, 4]).count()

# countApprox(timeout, confidence=0.95)
rdd = sc.parallelize(range(1000), 10)
rdd.countApprox(1000, 1.0)

# countApproxDistinct(relativeSD=0.05)
n = sc.parallelize(range(1000)).map(str).countApproxDistinct()
900 < n < 1100
n = sc.parallelize([i % 20 for i in range(1000)]).countApproxDistinct()
16 < n < 24

# countByKey()
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
sorted(rdd.countByKey().items())

# countByValue()
sorted(sc.parallelize([1, 2, 1, 2, 2], 2).countByValue().items())

# distinct(numPartitions=None)
sorted(sc.parallelize([1, 1, 2, 3]).distinct().collect())

# filter(f)
rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.filter(lambda x: x % 2 == 0).collect()

# first()
sc.parallelize([2, 3, 4]).first()
sc.parallelize([]).first()

### flatMap(f, preservesPartitioning=False)
# Return a new RDD by first applying a function to all elements of this RDD, and then flattening the results.
rdd = sc.parallelize([2, 3, 4])
sorted(rdd.flatMap(lambda x: range(1, x)).collect())
sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect())

# flatMapValues(f)
x = sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
def f(x): return x
x.flatMapValues(f).collect()

# fold(zeroValue, op)
from operator import add
sc.parallelize([1, 2, 3, 4, 5]).fold(0, add)

# foldByKey(zeroValue, func, numPartitions=None)
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
from operator import add
sorted(rdd.foldByKey(0, add).collect())

# foreach(f)
def f(x): print(x)
sc.parallelize([1, 2, 3, 4, 5]).foreach(f)

# foreachPartition(f)
def f(iterator):
    for x in iterator:
        print(x)
sc.parallelize([1, 2, 3, 4, 5]).foreachPartition(f)

# fullOuterJoin(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2), ("c", 8)])
sorted(x.fullOuterJoin(y).collect())

# getNumPartitions()
getNumPartitions()
rdd = sc.parallelize([1, 2, 3, 4], 2)
rdd.getNumPartitions()

# getStorageLevel()
rdd1 = sc.parallelize([1,2])
rdd1.getStorageLevel()
print(rdd1.getStorageLevel())

# glom()
rdd = sc.parallelize([1, 2, 3, 4], 2)
sorted(rdd.glom().collect())

# groupBy(f, numPartitions=None)
rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
result = rdd.groupBy(lambda x: x % 2).collect()
sorted([(x, sorted(y)) for (x, y) in result])

# groupByKey(numPartitions=None)
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
sorted(rdd.groupByKey().mapValues(len).collect())
sorted(rdd.groupByKey().mapValues(list).collect())

# groupWith(other, *others)
w = sc.parallelize([("a", 5), ("b", 6)])
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2)])
z = sc.parallelize([("b", 42)])
[(x, tuple(map(list, y))) for x, y in sorted(list(w.groupWith(x, y, z).collect()))]

# histogram(buckets)
rdd = sc.parallelize(range(51))
rdd.histogram(2)
rdd.histogram([0, 5, 25, 50])
rdd.histogram([0, 15, 30, 45, 60])  # evenly spaced buckets
rdd = sc.parallelize(["ab", "ac", "b", "bd", "ef"])
rdd.histogram(("a", "b", "c"))

# id()

# intersection(other)
rdd1 = sc.parallelize([1, 10, 2, 3, 4, 5])
rdd2 = sc.parallelize([1, 6, 2, 3, 7, 8])
rdd1.intersection(rdd2).collect()

# isCheckpointed()

# isEmpty()
sc.parallelize([]).isEmpty()
sc.parallelize([1]).isEmpty()

# join(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2), ("a", 3)])
sorted(x.join(y).collect())

# keyBy(f)
x = sc.parallelize(range(0,3)).keyBy(lambda x: x*x)
y = sc.parallelize(zip(range(0,5), range(0,5)))
[(x, list(map(list, y))) for x, y in sorted(x.cogroup(y).collect())]

# keys()
m = sc.parallelize([(1, 2), (3, 4)]).keys()
m.collect()

# leftOuterJoin(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2)])
sorted(x.leftOuterJoin(y).collect())

# lookup(key)
l = range(1000)
rdd = sc.parallelize(zip(l, l), 10)
rdd.lookup(42)  # slow
sorted = rdd.sortByKey()
sorted.lookup(42)  # fast
sorted.lookup(1024)

# map(f, preservesPartitioning=False)
rdd = sc.parallelize(["b", "a", "c"])
sorted(rdd.map(lambda x: (x, 1)).collect())

# mapPartitions(f, preservesPartitioning=False)
rdd = sc.parallelize([1, 2, 3, 4], 2)
def f(iterator): yield sum(iterator)
rdd.mapPartitions(f).collect()

# mapPartitionsWithIndex(f, preservesPartitioning=False)
rdd = sc.parallelize([1, 2, 3, 4], 4)
def f(splitIndex, iterator): yield splitIndex
rdd.mapPartitionsWithIndex(f).sum()

# mapValues(f)
x = sc.parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])])
def f(x): return len(x)
x.mapValues(f).collect()

# max(key=None)
rdd = sc.parallelize([1.0, 5.0, 43.0, 10.0])
rdd.max()
rdd.max(key=str)

# mean()
sc.parallelize([1, 2, 3]).mean()

# meanApprox()
rdd = sc.parallelize(range(1000), 10)
r = sum(range(1000)) / 1000.0
abs(rdd.meanApprox(1000) - r) / r < 0.05

# min(key=None)
rdd = sc.parallelize([2.0, 5.0, 43.0, 10.0])
rdd.min()
rdd.min(key=str)

# partitionBy(numPartitions, partitionFunc=<function portable_hash at 0x7f8d1be68a28>)
pairs = sc.parallelize([1, 2, 3, 4, 2, 4, 1]).map(lambda x: (x, x))
sets = pairs.partitionBy(2).glom().collect()
len(set(sets[0]).intersection(set(sets[1])))

# persist(storageLevel=StorageLevel(False, True, False, False, 1))
rdd = sc.parallelize(["b", "a", "c"])
rdd.persist().is_cached

# pipe(command, env={})
sc.parallelize(['1', '2', '', '3']).pipe('cat').collect()

# randomSplit(weights, seed=None)
rdd = sc.parallelize(range(500), 1)
rdd1, rdd2 = rdd.randomSplit([2, 3], 17)
len(rdd1.collect() + rdd2.collect())
150 < rdd1.count() < 250
250 < rdd2.count() < 350

# reduce(f)
from operator import add
sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
sc.parallelize([]).reduce(add)

# reduceByKey(func, numPartitions=None)
from operator import add
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
sorted(rdd.reduceByKey(add).collect())

# reduceByKeyLocally(func)
from operator import add
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
sorted(rdd.reduceByKeyLocally(add).items())

# repartition(numPartitions)
rdd = sc.parallelize([1,2,3,4,5,6,7], 4)
sorted(rdd.glom().collect())
len(rdd.repartition(2).glom().collect())
len(rdd.repartition(10).glom().collect())

# repartitionAndSortWithinPartitions(numPartitions=None, partitionFunc=<function portable_hash at 0x7f8d1be68a28>, ascending=True, keyfunc=<function <lambda> at 0x7f8d1be6e230>)
rdd = sc.parallelize([(0, 5), (3, 8), (2, 6), (0, 8), (3, 8), (1, 3)])
rdd2 = rdd.repartitionAndSortWithinPartitions(2, lambda x: x % 2, 2)
rdd2.glom().collect()

# rightOuterJoin(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2)])
sorted(y.rightOuterJoin(x).collect())

# sample(withReplacement, fraction, seed=None)
rdd = sc.parallelize(range(100), 4)
6 <= rdd.sample(False, 0.1, 81).count() <= 14

# sampleByKey(withReplacement, fractions, seed=None)
fractions = {"a": 0.2, "b": 0.1}
rdd = sc.parallelize(fractions.keys()).cartesian(sc.parallelize(range(0, 1000)))
sample = dict(rdd.sampleByKey(False, fractions, 2).groupByKey().collect())
100 < len(sample["a"]) < 300 and 50 < len(sample["b"]) < 150
max(sample["a"]) <= 999 and min(sample["a"]) >= 0
max(sample["b"]) <= 999 and min(sample["b"]) >= 0

# sampleStdev()
sc.parallelize([1, 2, 3]).sampleStdev()

# saveAsTextFile(path, compressionCodecClass=None)
tempFile = NamedTemporaryFile(delete=True)
tempFile.close()
sc.parallelize(range(10)).saveAsTextFile(tempFile.name)
from fileinput import input
from glob import glob
''.join(sorted(input(glob(tempFile.name + "/part-0000*"))))

# Empty lines are tolerated when saving to text files.
tempFile2 = NamedTemporaryFile(delete=True)
tempFile2.close()
sc.parallelize(['', 'foo', '', 'bar', '']).saveAsTextFile(tempFile2.name)
''.join(sorted(input(glob(tempFile2.name + "/part-0000*"))))

# Using compressionCodecClass
tempFile3 = NamedTemporaryFile(delete=True)
tempFile3.close()
codec = "org.apache.hadoop.io.compress.GzipCodec"
sc.parallelize(['foo', 'bar']).saveAsTextFile(tempFile3.name, codec)
from fileinput import input, hook_compressed
result = sorted(input(glob(tempFile3.name + "/part*.gz"), openhook=hook_compressed))
b''.join(result).decode('utf-8')

# setName(name)
rdd1 = sc.parallelize([1, 2])
rdd1.setName('RDD1').name()

# sortBy(keyfunc, ascending=True, numPartitions=None)
tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()
sc.parallelize(tmp).sortBy(lambda x: x[1]).collect()

# sortByKey(ascending=True, numPartitions=None, keyfunc=<function <lambda> at 0x7f8d1be6e320>)
tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
sc.parallelize(tmp).sortByKey().first()
sc.parallelize(tmp).sortByKey(True, 1).collect()
sc.parallelize(tmp).sortByKey(True, 2).collect()
tmp2 = [('Mary', 1), ('had', 2), ('a', 3), ('little', 4), ('lamb', 5)]
tmp2.extend([('whose', 6), ('fleece', 7), ('was', 8), ('white', 9)])
sc.parallelize(tmp2).sortByKey(True, 3, keyfunc=lambda k: k.lower()).collect()

# stdev()
sc.parallelize([1, 2, 3]).stdev()

# subtract(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 3)])
y = sc.parallelize([("a", 3), ("c", None)])
sorted(x.subtract(y).collect())

# subtractByKey(other, numPartitions=None)
x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 2)])
y = sc.parallelize([("a", 3), ("c", None)])
sorted(x.subtractByKey(y).collect())

# sum()
sc.parallelize([1.0, 2.0, 3.0]).sum()

# sumApprox(timeout, confidence=0.95)
rdd = sc.parallelize(range(1000), 10)
r = sum(range(1000))
abs(rdd.sumApprox(1000) - r) / r < 0.05

# take(num)
sc.parallelize([2, 3, 4, 5, 6]).cache().take(2)
sc.parallelize([2, 3, 4, 5, 6]).take(10)
sc.parallelize(range(100), 100).filter(lambda x: x > 90).take(3)

# takeOrdered(num, key=None)
sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7]).takeOrdered(6)
sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7], 2).takeOrdered(6, key=lambda x: -x)


# takeSample(withReplacement, seed=None)
rdd = sc.parallelize(range(0, 10))
len(rdd.takeSample(True, 20, 1))
len(rdd.takeSample(False, 5, 2))
len(rdd.takeSample(False, 15, 3))

# top(num, key=None)
sc.parallelize([10, 4, 2, 12, 3]).top(1)
sc.parallelize([2, 3, 4, 5, 6], 2).top(2)
sc.parallelize([10, 4, 2, 12, 3]).top(3, key=str)

# treeAggregate(zeroValue, seqOp, combOp, depth=2)
add = lambda x, y: x + y
rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
rdd.treeAggregate(0, add, add)
rdd.treeAggregate(0, add, add, 1)
rdd.treeAggregate(0, add, add, 2)
rdd.treeAggregate(0, add, add, 5)
rdd.treeAggregate(0, add, add, 10)

# treeReduce(f, depth=2)
add = lambda x, y: x + y
rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
rdd.treeReduce(add)
rdd.treeReduce(add, 1)
rdd.treeReduce(add, 2)
rdd.treeReduce(add, 5)
rdd.treeReduce(add, 10)

# union(other)
rdd = sc.parallelize([1, 1, 2, 3])
rdd.union(rdd).collect()

# values()
m = sc.parallelize([(1, 2), (3, 4)]).values()
m.collect()

# variance()
sc.parallelize([1, 2, 3]).variance()

# zip(other)
x = sc.parallelize(range(0,5))
y = sc.parallelize(range(1000, 1005))
x.zip(y).collect()

# zipWithIndex()
sc.parallelize(["a", "b", "c", "d"], 3).zipWithIndex().collect()

# zipWithUniqueId()
sc.parallelize(["a", "b", "c", "d", "e"], 3).zipWithUniqueId().collect()


### BROADCAST
from pyspark.context import SparkContext
sc = SparkContext('local', 'test')
b = sc.broadcast([1, 2, 3, 4, 5])
b.value
sc.parallelize([0, 0]).flatMap(lambda x: b.value).collect()
b.unpersist()

large_broadcast = sc.broadcast(range(10000))


