#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyhdfs import HdfsClient
import os

# 这里用的是IP，因为现在并不是在容器内运行脚本
# client = HdfsClient(hosts='localhost:50070',user_name='root')
# print(client.listdir('/ml-latest-small'))

# 在本地执行不了，连接不上。但是在容器执行是没问题的
spark=SparkSession.builder.getOrCreate()
# lines = spark.read.text("hdfs://hadoop-node1:9000/ml-latest-small/ratings.csv").rdd
lines = spark.read.text('ml-latest-small/ratings.csv').rdd

header = lines.first()
print(lines.take(5))
lines = lines.filter(lambda row:row!=header)
print(lines.take(5))
parts = lines.map(lambda row: row.value.split(","))
print(parts.take(5))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=float(p[3])))
print(parts.take(5))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
print(userRecs.show())
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
print(movieRecs.show())

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
print(userSubsetRecs.show())
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
print(movieSubSetRecs.show())