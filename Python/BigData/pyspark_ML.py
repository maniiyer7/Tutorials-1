###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: August 6, 2015
# summary: pyspark ML library examples
# Source: https://github.com/apache/spark/tree/master/examples/src/main/python/ml
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
### CROSS VALIDATOR
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/cross_validator.py
# https://databricks.com/blog/2015/01/07/ml-pipelines-a-new-high-level-api-for-mllib.html

from __future__ import print_function

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row, SQLContext

"""
A simple example demonstrating model selection using CrossValidator.
This example also demonstrates how Pipelines are Estimators.
Run with:
  bin/spark-submit examples/src/main/python/ml/cross_validator.py
"""

if __name__ == "__main__":
    # if there is an existing sc running, stop it before this command. sc.stop()
    sc = SparkContext(appName="CrossValidatorExample")
    sqlContext = SQLContext(sc)

    # Prepare training documents, which are labeled.
    LabeledDocument = Row("id", "text", "label")
    training = sc.parallelize([(0, "a b c d e spark", 1.0),
                               (1, "b d", 0.0),
                               (2, "spark f g h", 1.0),
                               (3, "hadoop mapreduce", 0.0),
                               (4, "b spark who", 1.0),
                               (5, "g d a y", 0.0),
                               (6, "spark fly", 1.0),
                               (7, "was mapreduce", 0.0),
                               (8, "e spark program", 1.0),
                               (9, "a e c l", 0.0),
                               (10, "spark compile", 1.0),
                               (11, "hadoop software", 0.0)
                               ]) \
        .map(lambda x: LabeledDocument(*x)).toDF()

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    # This will allow us to jointly choose parameters for all Pipeline stages.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    # this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=2)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(training)

    # Prepare test documents, which are unlabeled.
    Document = Row("id", "text")
    test = sc.parallelize([(4L, "spark i j k"),
                           (5L, "l m n"),
                           (6L, "mapreduce spark"),
                           (7L, "apache hadoop")]) \
        .map(lambda x: Document(*x)).toDF()

    # Make predictions on test documents. cvModel uses the best model found (lrModel).
    prediction = cvModel.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        print(row)

    sc.stop()
###############################################################################

###############################################################################
### GRADIENT-BOOSTED TREES
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/gradient_boosted_trees.py
from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.mllib.evaluation import BinaryClassificationMetrics, RegressionMetrics
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row, SQLContext

# Note: Note: GBTClassifier only supports binary classification currently

def testClassification(train, test):
    # Train a GradientBoostedTrees model.

    rf = GBTClassifier(maxIter=30, maxDepth=4, labelCol="indexedLabel")

    model = rf.fit(train)
    predictionAndLabels = model.transform(test).select("prediction", "indexedLabel") \
        .map(lambda x: (x.prediction, x.indexedLabel))

    metrics = BinaryClassificationMetrics(predictionAndLabels)
    print("AUC %.3f" % metrics.areaUnderROC)


def testRegression(train, test):
    # Train a GradientBoostedTrees model.

    rf = GBTRegressor(maxIter=30, maxDepth=4, labelCol="indexedLabel")

    model = rf.fit(train)
    predictionAndLabels = model.transform(test).select("prediction", "indexedLabel") \
        .map(lambda x: (x.prediction, x.indexedLabel))

    metrics = RegressionMetrics(predictionAndLabels)
    print("rmse %.3f" % metrics.rootMeanSquaredError)
    print("r2 %.3f" % metrics.r2)
    print("mae %.3f" % metrics.meanAbsoluteError)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Usage: gradient_boosted_trees", file=sys.stderr)
        exit(1)
    sc = SparkContext(appName="PythonGBTExample")
    sqlContext = SQLContext(sc)

    # Load and parse the data file into a dataframe.
    df = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

    # Map labels into an indexed column of labels in [0, numLabels)
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    [train, test] = td.randomSplit([0.7, 0.3])
    testClassification(train, test)
    testRegression(train, test)
    sc.stop()
###############################################################################

###############################################################################
### K-MEANS
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/kmeans_example.py
from __future__ import print_function

import sys
import re

import numpy as np
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import VectorUDT, _convert_to_vector
from pyspark.sql import SQLContext
from pyspark.sql.types import Row, StructField, StructType

"""
A simple example demonstrating a k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py <input> <k>
This example requires NumPy (http://www.numpy.org/).
"""


def parseVector(line):
    array = np.array([float(x) for x in line.split(' ')])
    return _convert_to_vector(array)


if __name__ == "__main__":

    FEATURES_COL = "features"

    # if len(sys.argv) != 3:
    #     print("Usage: kmeans_example.py <file> <k>", file=sys.stderr)
    #     exit(-1)
    path = '/Users/amirkavousian/Documents/PROJECTS/IrradianceData/Data/'+'LatLngs.csv'
    k = 15

    sc = SparkContext(appName="PythonKMeansExample")
    sqlContext = SQLContext(sc)

    lines = sc.textFile(path)
    data = lines.map(parseVector)
    row_rdd = data.map(lambda x: Row(x))
    schema = StructType([StructField(FEATURES_COL, VectorUDT(), False)])
    df = sqlContext.createDataFrame(row_rdd, schema)

    kmeans = KMeans().setK(2).setSeed(1).setFeaturesCol(FEATURES_COL)
    model = kmeans.fit(df)
    centers = model.clusterCenters()

    print("Cluster Centers: ")
    for center in centers:
        print(center)

    sc.stop()
###############################################################################

###############################################################################
### LOGISTIC REGRESSION
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/logistic_regression.py
###############################################################################
