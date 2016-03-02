###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: August 6, 2015
# summary: pyspark examples
# Source: https://github.com/apache/spark/tree/master/examples/src/main/python
###############################################################################

###############################################################################
########################### IMPORTS AND OPTIONS ###############################
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
###################################### MAIN ###################################
###############################################################################




###############################################################################
### ALS EXAMPLE IN SPARK
"""ALS is Alternating Least Squares, and is used in recommendation systems.
ALS models the rating matrix (R) as the multiplication of low-rank user (U) and product (V) factors,
and learns these factors by minimizing the reconstruction error of the observed ratings.
ALS is an iterative algorithm. In each iteration, the algorithm alternatively
fixes one factor matrix and solves for the other, and this process continues until it converges.
"""

# https://github.com/apache/spark/blob/master/examples/src/main/python/als.py
from __future__ import print_function

import sys
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkContext

LAMBDA = 0.01   # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / M * U)


def update(i, vec, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use the ALS method found in pyspark.mllib.recommendation for more
      conventional use.""", file=sys.stderr)

    sc = SparkContext(appName="PythonALS")
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
          (M, U, F, ITERATIONS, partitions))

    R = matrix(rand(M, F)) * matrix(rand(U, F).T)
    ms = matrix(rand(M, F))
    us = matrix(rand(U, F))

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)

    sc.stop()


#######################################
### BENCHMARKING ALS ON SPARK USING EC2
# https://github.com/databricks/als-benchmark-scripts


#######################################
### ALS EXAMPLE IN SPARK
# https://databricks.com/blog/2014/07/23/scalable-collaborative-filtering-with-spark-mllib.html
from pyspark.mllib.recommendation import ALS

# load training and test data into (user, product, rating) tuples
def parseRating(line):
  fields = line.split()
  return (int(fields[0]), int(fields[1]), float(fields[2]))
training = sc.textFile("...").map(parseRating).cache()
test = sc.textFile("...").map(parseRating)

# train a recommendation model
model = ALS.train(training, rank = 10, iterations = 5)

# make predictions on (user, product) pairs from the test data
predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))


#######################################
### SNAP DATA SET
# https://snap.stanford.edu/data/web-Amazon.html

###############################################################################
############################ COMMON INTERFACES ################################
###############################################################################

###############################################################################
### AVRO EXAMPLE IN SPARK
# https://github.com/apache/spark/blob/master/examples/src/main/python/avro_inputformat.py

from __future__ import print_function
import sys
from pyspark import SparkContext
from functools import reduce


###############################################################################
### CASSANDRA INPUT FORMAT
# https://github.com/apache/spark/blob/master/examples/src/main/python/cassandra_inputformat.py
from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("""
        Usage: cassandra_inputformat <host> <keyspace> <cf>
        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \
        /path/to/examples/cassandra_inputformat.py <host> <keyspace> <cf>
        Assumes you have some data in Cassandra already, running on <host>, in <keyspace> and <cf>
        """, file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    keyspace = sys.argv[2]
    cf = sys.argv[3]
    sc = SparkContext(appName="CassandraInputFormat")

    conf = {"cassandra.input.thrift.address": host,
            "cassandra.input.thrift.port": "9160",
            "cassandra.input.keyspace": keyspace,
            "cassandra.input.columnfamily": cf,
            "cassandra.input.partitioner.class": "Murmur3Partitioner",
            "cassandra.input.page.row.size": "3"}
    cass_rdd = sc.newAPIHadoopRDD(
        "org.apache.cassandra.hadoop.cql3.CqlPagingInputFormat",
        "java.util.Map",
        "java.util.Map",
        keyConverter="org.apache.spark.examples.pythonconverters.CassandraCQLKeyConverter",
        valueConverter="org.apache.spark.examples.pythonconverters.CassandraCQLValueConverter",
        conf=conf)
    output = cass_rdd.collect()
    for (k, v) in output:
        print((k, v))

    sc.stop()


###############################################################################
### CASSANDRA OUTPUT FORMAT
# https://github.com/apache/spark/blob/master/examples/src/main/python/cassandra_outputformat.py
from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("""
        Usage: cassandra_outputformat <host> <keyspace> <cf> <user_id> <fname> <lname>
        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \
        /path/to/examples/cassandra_outputformat.py <args>
        Assumes you have created the following table <cf> in Cassandra already,
        running on <host>, in <keyspace>.
        cqlsh:<keyspace>> CREATE TABLE <cf> (
           ...   user_id int PRIMARY KEY,
           ...   fname text,
           ...   lname text
           ... );
        """, file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    keyspace = sys.argv[2]
    cf = sys.argv[3]
    sc = SparkContext(appName="CassandraOutputFormat")

    conf = {"cassandra.output.thrift.address": host,
            "cassandra.output.thrift.port": "9160",
            "cassandra.output.keyspace": keyspace,
            "cassandra.output.partitioner.class": "Murmur3Partitioner",
            "cassandra.output.cql": "UPDATE " + keyspace + "." + cf + " SET fname = ?, lname = ?",
            "mapreduce.output.basename": cf,
            "mapreduce.outputformat.class": "org.apache.cassandra.hadoop.cql3.CqlOutputFormat",
            "mapreduce.job.output.key.class": "java.util.Map",
            "mapreduce.job.output.value.class": "java.util.List"}
    key = {"user_id": int(sys.argv[4])}
    sc.parallelize([(key, sys.argv[5:])]).saveAsNewAPIHadoopDataset(
        conf=conf,
        keyConverter="org.apache.spark.examples.pythonconverters.ToCassandraCQLKeyConverter",
        valueConverter="org.apache.spark.examples.pythonconverters.ToCassandraCQLValueConverter")

    sc.stop()


###############################################################################
### HBASE INPUT FORMAT
# https://github.com/apache/spark/blob/master/examples/src/main/python/hbase_inputformat.py
from __future__ import print_function
import sys
import json
from pyspark import SparkContext


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("""
        Usage: hbase_inputformat <host> <table>
        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \
        /path/to/examples/hbase_inputformat.py <host> <table> [<znode>]
        Assumes you have some data in HBase already, running on <host>, in <table>
          optionally, you can specify parent znode for your hbase cluster - <znode>
        """, file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    table = sys.argv[2]
    sc = SparkContext(appName="HBaseInputFormat")

    # Other options for configuring scan behavior are available. More information available at
    # https://github.com/apache/hbase/blob/master/hbase-server/src/main/java/org/apache/hadoop/hbase/mapreduce/TableInputFormat.java
    conf = {"hbase.zookeeper.quorum": host, "hbase.mapreduce.inputtable": table}
    if len(sys.argv) > 3:
        conf = {"hbase.zookeeper.quorum": host, "zookeeper.znode.parent": sys.argv[3],
                "hbase.mapreduce.inputtable": table}
    keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"

    hbase_rdd = sc.newAPIHadoopRDD(
        "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
        "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "org.apache.hadoop.hbase.client.Result",
        keyConverter=keyConv,
        valueConverter=valueConv,
        conf=conf)
    hbase_rdd = hbase_rdd.flatMapValues(lambda v: v.split("\n")).mapValues(json.loads)

    output = hbase_rdd.collect()
    for (k, v) in output:
        print((k, v))

    sc.stop()


###############################################################################
### HBASE OUTPUT FORMAT
# https://github.com/apache/spark/blob/master/examples/src/main/python/hbase_outputformat.py
from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("""
        Usage: hbase_outputformat <host> <table> <row> <family> <qualifier> <value>
        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \
        /path/to/examples/hbase_outputformat.py <args>
        Assumes you have created <table> with column family <family> in HBase
        running on <host> already
        """, file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    table = sys.argv[2]
    sc = SparkContext(appName="HBaseOutputFormat")

    conf = {"hbase.zookeeper.quorum": host,
            "hbase.mapred.outputtable": table,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"

    sc.parallelize([sys.argv[3:]]).map(lambda x: (x[0], x)).saveAsNewAPIHadoopDataset(
        conf=conf,
        keyConverter=keyConv,
        valueConverter=valueConv)

    sc.stop()
###############################################################################


###############################################################################
### PARQUET INPUT FORMAT
# https://github.com/apache/spark/blob/master/examples/src/main/python/parquet_inputformat.py
from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""
        Usage: parquet_inputformat.py <data_file>
        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \\
                /path/to/examples/parquet_inputformat.py <data_file>
        Assumes you have Parquet data stored in <data_file>.
        """, file=sys.stderr)
        exit(-1)

    path = sys.argv[1]
    sc = SparkContext(appName="ParquetInputFormat")

    parquet_rdd = sc.newAPIHadoopFile(
        path,
        'org.apache.parquet.avro.AvroParquetInputFormat',
        'java.lang.Void',
        'org.apache.avro.generic.IndexedRecord',
        valueConverter='org.apache.spark.examples.pythonconverters.IndexedRecordToJavaConverter')
    output = parquet_rdd.map(lambda x: x[1]).collect()
    for k in output:
        print(k)

    sc.stop()
###############################################################################


###############################################################################
### Pi: estimate Pi using brute-force method
import sys
from random import random
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    sc = SparkContext(appName="PythonPi")
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = 100000 * partitions

    def f(_):
        x = np.random.random() * 2 - 1
        y = np.random.random() * 2 - 1
        return 1 if x ** 2 + y ** 2 < 1 else 0

    count = sc.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    print("Pi is roughly %f" % (4.0 * count / n))

    sc.stop()
###############################################################################


###############################################################################
### SORT
# https://github.com/apache/spark/blob/master/examples/src/main/python/sort.py
from __future__ import print_function
import sys
from pyspark import SparkContext


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonSort")

    lines = sc.textFile(sys.argv[1], 1)
    sortedCount = lines.flatMap(lambda x: x.split(' ')) \
        .map(lambda x: (int(x), 1)) \
        .sortByKey(lambda x: x)
    # This is just a demo on how to bring all the sorted data back to a single node.
    # In reality, we wouldn't want to collect all the data to the driver node.
    output = sortedCount.collect()
    for (num, unitcount) in output:
        print(num)

    sc.stop()
###############################################################################


###############################################################################
### SQL
# https://github.com/apache/spark/blob/master/examples/src/main/python/sql.py
from __future__ import print_function

import os
import sys

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

if __name__ == "__main__":
    sc = SparkContext(appName="PythonSQL")
    sqlContext = SQLContext(sc)

    # RDD is created from a list of rows
    some_rdd = sc.parallelize([Row(name="John", age=19),
                              Row(name="Smith", age=23),
                              Row(name="Sarah", age=18)])
    # Infer schema from the first row, create a DataFrame and print the schema
    some_df = sqlContext.createDataFrame(some_rdd)
    some_df.printSchema()

    # Another RDD is created from a list of tuples
    another_rdd = sc.parallelize([("John", 19), ("Smith", 23), ("Sarah", 18)])
    # Schema with two fields - person_name and person_age
    schema = StructType([StructField("person_name", StringType(), False),
                        StructField("person_age", IntegerType(), False)])
    # Create a DataFrame by applying the schema to the RDD and print the schema
    another_df = sqlContext.createDataFrame(another_rdd, schema)
    another_df.printSchema()
    # root
    #  |-- age: integer (nullable = true)
    #  |-- name: string (nullable = true)

    # A JSON dataset is pointed to by path.
    # The path can be either a single text file or a directory storing text files.
    # if len(sys.argv) < 2:
    #     path = "file://" + \
    #         os.path.join(os.environ['SPARK_HOME'], "examples/src/main/resources/people.json")
    # else:
    #     path = sys.argv[1]
    path = '/Users/amirkavousian/Documents/Py_Codes/spark/examples/src/main/resources/people.json'
    # Create a DataFrame from the file(s) pointed to by path
    people = sqlContext.jsonFile(path)
    # root
    #  |-- person_name: string (nullable = false)
    #  |-- person_age: integer (nullable = false)

    # The inferred schema can be visualized using the printSchema() method.
    people.printSchema()
    # root
    #  |-- age: IntegerType
    #  |-- name: StringType

    # Register this DataFrame as a table.
    people.registerAsTable("people")

    # SQL statements can be run by using the sql methods provided by sqlContext
    teenagers = sqlContext.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")

    for each in teenagers.collect():
        print(each[0])

    sc.stop()
###############################################################################


###############################################################################
### STATUS API DEMO
# https://github.com/apache/spark/blob/master/examples/src/main/python/status_api_demo.py
from __future__ import print_function

import time
import multiprocessing
import Queue

from pyspark import SparkConf, SparkContext


def delayed(seconds):
    def f(x):
        time.sleep(seconds)
        return x
    return f

def call_in_background(f, *args):
    result = Queue.Queue(1)
    t = multiprocessing.Thread(target=lambda: result.put(f(*args)))
    t.daemon = True
    t.start()
    return result

def main():
    conf = SparkConf().set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)

    def run():
        rdd = sc.parallelize(range(10), 10).map(delayed(2))
        reduced = rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
        return reduced.map(delayed(2)).collect()

    result = call_in_background(run)
    status = sc.statusTracker()
    while result.empty():
        ids = status.getJobIdsForGroup()
        for id in ids:
            job = status.getJobInfo(id)
            print("Job", id, "status: ", job.status)
            for sid in job.stageIds:
                info = status.getStageInfo(sid)
                if info:
                    print("Stage %d: %d tasks total (%d active, %d complete)" %
                          (sid, info.numTasks, info.numActiveTasks, info.numCompletedTasks))
        time.sleep(1)

    print("Job results are:", result.get())
    sc.stop()

if __name__ == "__main__":
    main()
###############################################################################

###############################################################################
########################## COMMON STATS FUNCTIONS #############################
###############################################################################

###############################################################################
### K-MEANS
# https://github.com/apache/spark/blob/master/examples/src/main/python/kmeans.py
from __future__ import print_function
import sys
import numpy as np
from pyspark import SparkContext

def parseVector(line):
    return np.array([float(x) for x in line.split(',')])

def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of KMeans Clustering and is given
       as an example! Please refer to examples/src/main/python/mllib/kmeans.py for an example on
       how to use MLlib's KMeans implementation.""", file=sys.stderr)

    sc = SparkContext(appName="PythonKMeans")
    lines = sc.textFile('/Users/amirkavousian/Documents/PROJECTS/IrradianceData/Data/'+'LatLngs.csv')
    # lines = sc.textFile(sys.argv[1])
    data = lines.map(parseVector).cache()
    # K = int(sys.argv[2])
    K = 15
    # convergeDist = float(sys.argv[3])
    convergeDist = 1

    kPoints = data.takeSample(False, K, 1)
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    print("Final centers: " + str(kPoints))

    sc.stop()


###############################################################################
### LOGISTIC REGRESSION
# https://github.com/apache/spark/blob/master/examples/src/main/python/logistic_regression.py
from __future__ import print_function
import sys
import numpy as np
from pyspark import SparkContext

D = 2  # Number of dimensions

def readPointBatch(iterator):
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return [matrix]

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: logistic_regression <file> <iterations>", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of Logistic Regression and is
      given as an example! Please refer to examples/src/main/python/mllib/logistic_regression.py
      to see how MLlib's implementation is used.""", file=sys.stderr)

    sc = SparkContext(appName="PythonLR")
    points = sc.textFile('/Users/amirkavousian/Documents/PROJECTS/IrradianceData/Data/'+'StateLatLngs.csv').mapPartitions(readPointBatch).cache()
    iterations = 100

    # Initialize w to a random value
    w = 2 * np.random.ranf(size=D) - 1
    print("Initial w: " + str(w))

    # Compute logistic regression gradient for a matrix of data points
    def gradient(matrix, w):
        Y = matrix[:, 0]    # point labels (first column of input file)
        X = matrix[:, 1:]   # point coordinates
        # For each point (x, y), compute gradient function, then sum these up
        return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

    def add(x, y):
        x += y
        return x

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        w -= points.map(lambda m: gradient(m, w)).reduce(add)

    print("Final w: " + str(w))

    sc.stop()

###############################################################################
### PAGE RANK
# https://github.com/apache/spark/blob/master/examples/src/main/python/pagerank.py
"""
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx
"""

from __future__ import print_function
import re
import sys
from operator import add
from pyspark import SparkContext


def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: pagerank <file> <iterations>", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of PageRank and is
          given as an example! Please refer to PageRank implementation provided by graphx""",
          file=sys.stderr)

    # Initialize the spark context.
    sc = SparkContext(appName="PythonPageRank")

    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    lines = sc.textFile(sys.argv[1], 1)

    # Loads all URLs from input file and initialize their neighbors.
    links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(sys.argv[2])):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

            # Collects all URL ranks and dump them to console.
    for (link, rank) in ranks.collect():
        print("%s has rank: %s." % (link, rank))

    sc.stop()


import itertools
horses = [1, 2, 3, 4]
races = itertools.permutations(horses)
print(races)
print(list(itertools.permutations(horses)))


def f123():
    yield 1
    yield 2
    yield 3

for item in f123():
    print(item)
###############################################################################

###############################################################################
### TRANSITIVE CLOSURE
# https://github.com/apache/spark/blob/master/examples/src/main/python/transitive_closure.py
from __future__ import print_function
import sys
from random import Random
from pyspark import SparkContext

numEdges = 200
numVertices = 100
rand = np.random.rand(42)

def generateGraph():
    edges = set()
    while len(edges) < numEdges:
        src = rand.randrange(0, numEdges)
        dst = rand.randrange(0, numEdges)
        if src != dst:
            edges.add((src, dst))
    return edges

if __name__ == "__main__":
    """
    Usage: transitive_closure [partitions]
    """
    sc = SparkContext(appName="PythonTransitiveClosure")
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tc = sc.parallelize(generateGraph(), partitions).cache()

    # Linear transitive closure: each round grows paths by one edge,
    # by joining the graph's edges with the already-discovered paths.
    # e.g. join the path (y, z) from the TC with the edge (x, y) from
    # the graph to obtain the path (x, z).

    # Because join() joins on keys, the edges are stored in reversed order.
    edges = tc.map(lambda x_y: (x_y[1], x_y[0]))

    oldCount = 0
    nextCount = tc.count()
    while True:
        oldCount = nextCount
        # Perform the join, obtaining an RDD of (y, (z, x)) pairs,
        # then project the result to obtain the new (x, z) paths.
        new_edges = tc.join(edges).map(lambda __a_b: (__a_b[1][1], __a_b[1][0]))
        tc = tc.union(new_edges).distinct().cache()
        nextCount = tc.count()
        if nextCount == oldCount:
            break

    print("TC has %i edges" % tc.count())

    sc.stop()
###############################################################################

###############################################################################
### WORD COUNT
# https://github.com/apache/spark/blob/master/examples/src/main/python/wordcount.py
from __future__ import print_function
import sys
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonWordCount")
    lines = sc.textFile('/Users/amirkavousian/Documents/PROJECTS/IrradianceData/Data/'+'LatLngs.csv', 1)
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))

    sc.stop()
###############################################################################

###############################################################################
###

