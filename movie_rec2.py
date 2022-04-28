#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from pyspark.sql.functions import col, explode
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os,sys

class MovieRecSystem:
    def __init__(self):
        self.spark = SparkSession.builder.appName('Recommendations').getOrCreate()
        self.movie_path = 'ml-latest-small/movies.csv'
        self.rating_path = 'ml-latest-small/ratings.csv'
        self.hdfs_path = 'hdfs://hadoop-node1:9000/'
        # self.hdfs_path = './'

    def prepare_data(self):
        print('=======load data=======')
        self.movies = self.spark.read.csv(self.hdfs_path+self.movie_path,header=True)
        self.ratings = self.spark.read.csv(self.hdfs_path+self.rating_path,header=True)
        self.ratings.show()
        self.ratings.printSchema()
        self.ratings = self.ratings.\
            withColumn('userId', col('userId').cast('integer')).\
            withColumn('movieId', col('movieId').cast('integer')).\
            withColumn('rating', col('rating').cast('float')).\
            drop('timestamp')
        # Count the number of distinct userIds and distinct movieIds
        self.num_users = self.ratings.select("userId").distinct().count()
        self.num_movies = self.ratings.select("movieId").distinct().count()
        self.ratings.show()

    def cal_sparsity(self):
        print('=======Calculate sparsity=======')
        # Count the total number of ratings in the dataset
        numerator = self.ratings.select("rating").count()

        # Count the number of distinct userIds and distinct movieIds
        num_users = self.num_users
        num_movies = self.num_movies

        # Set the denominator equal to the number of users multiplied by the number of movies
        denominator = num_users * num_movies

        # Divide the numerator by the denominator
        sparsity = (1.0 - (numerator *1.0)/denominator)*100
        print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

    def interpret_rating(self):
        print('=======Interpret ratings=======')
        # Group data by userId, count ratings
        userId_ratings = self.ratings.groupBy("userId").count().orderBy('count', ascending=False)
        userId_ratings.show()

        # Group data by movieId, count ratings
        movieId_ratings = self.ratings.groupBy("movieId").count().orderBy('count', ascending=False)
        movieId_ratings.show()

    def train(self,reTrain=False):
        # 不知道为什么加载的模型无法使用recommendForAllUsers方法
        # if not reTrain and os.path.exists('./alsmodel'):
        #     print('=======Load An ALS Model=======')
        #     self.best_model = ALS.load('./alsmodel')
        #     print('load successfully!',type(self.best_model))
        #     return

        self.prepare_data()
        print('=======Build Out An ALS Model=======')
        # Create test and train set
        (train, test) = self.ratings.randomSplit([0.8, 0.2], seed = 1234)

        # Create ALS model
        # als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")

        als = ALS(rank=50,regParam=0.15,maxIter=10,userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

        # Confirm that a model called "als" was created
        print(type(als))

        # print('=======ALS Model Tuning using CrossValidation=======')
        # # Add hyperparameters and their respective values to param_grid
        # param_grid = ParamGridBuilder() \
        #     .addGrid(als.rank, [10, 50, 100, 150]) \
        #     .addGrid(als.regParam, [.01, .05, .1, .15]) \
        #     .build()
        #     #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

        # Define evaluator as RMSE and print length of evaluator
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        # print("Num models to be tested: ", len(param_grid))

        # Build cross validation using CrossValidator
        # cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
        # Confirm cv was built
        # print(cv)

        #Fit cross validator to the 'train' dataset
        # model = cv.fit(train)
        #Extract best model from the cv model above
        # self.best_model = model.bestModel
        self.best_model = als.fit(train)
        # Print best_model
        print(type(self.best_model))

        # View the predictions
        test_predictions = self.best_model.transform(test)
        RMSE = evaluator.evaluate(test_predictions)
        print('RMSE on test_dataset:',RMSE)
        print('=======Prediction on test set=======')
        test_predictions.show()

        # als.save('./alsmodel')

    def show_best_model(self):
        # Complete the code below to extract the ALS model parameters
        print("**Best Model**")
        # # Print "Rank"
        print("  Rank:", self.best_model._java_obj.parent().getRank())
        # Print "MaxIter"
        print("  MaxIter:", self.best_model._java_obj.parent().getMaxIter())
        # Print "RegParam"
        print("  RegParam:", self.best_model._java_obj.parent().getRegParam())

    def recommendForAllUsers(self,n_recommendations=10,n_rows=10):
        print('=======Make Recommendations=======')
        # Generate n Recommendations for all users
        nrecommendations = self.best_model.recommendForAllUsers(n_recommendations)
        nrecommendations.limit(n_rows).show()
        nrecommendations = nrecommendations\
            .withColumn("rec_exp", explode("recommendations"))\
            .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))

        nrecommendations.limit(n_rows).show()
        return nrecommendations

    # 给userId，推荐num部电影
    def recommendForUser(self,userId=100,num=10):
        print(f'=======Make {num} Recommendations For User:{userId}=======')
        nrecommendations = self.best_model.recommendForAllUsers(num)
        nrecommendations = nrecommendations.where(nrecommendations.userId==userId)
        nrecommendations = nrecommendations \
            .withColumn("rec_exp", explode("recommendations")) \
            .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
        nrecommendations.join(self.movies, on='movieId').sort('rating', ascending=False).limit(num).show(num)

        # nrecommendations.join(self.movies, on='movieId').filter('userId = '+userId).show()
        # sorted_rec = self.ratings.join(self.movies, on='movieId').filter('userId = '+userId).sort('rating', ascending=False)
        # sorted_rec.limit(num).show()
        # return self.to_Pandas(sorted_rec)

    # 将movieId推荐给num个用户
    def recommendMovieToUsers(self,movieId=100,num=10):
        print(f'=======Recommend movie:{movieId} to {num} user =======')
        nrecommendations = self.best_model.recommendForAllItems(num)
        nrecommendations = nrecommendations.where(nrecommendations.movieId == movieId)
        nrecommendations = nrecommendations \
            .withColumn("rec_exp", explode("recommendations")) \
            .select('movieId', col("rec_exp.userId"), col("rec_exp.rating"))
        # nrecommendations.show()
        nrecommendations.join(self.movies, on='movieId').sort('rating', ascending=False).limit(num).show(num)

    def get_user_movie(self):
        userId = self.ratings.select("userId").distinct()
        # userId.show()
        print('max userId:',userId.selectExpr('max(userId) as max_userId').first().max_userId)
        print('min userId:',userId.selectExpr('min(userId) as min_userId').first().min_userId)

        movieId = self.ratings.select("movieId").distinct()
        # movieId.show()
        print('max movieId:',movieId.selectExpr('max(movieId) as max_movieId').first().max_movieId)
        print('min movieId:',movieId.selectExpr('min(movieId) as min_movieId').first().min_movieId)

    def _map_to_pandas(self,rdds):
        """ Needs to be here due to pickling issues """
        return [pd.DataFrame(list(rdds))]

    def to_Pandas(self,df, n_partitions=None):
        """
        Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
        repartitioned if `n_partitions` is passed.
        :param df:              pyspark.sql.DataFrame
        :param n_partitions:    int or None
        :return:                pandas.DataFrame
        """
        if n_partitions is not None: df = df.repartition(n_partitions)
        df_pand = df.rdd.mapPartitions(self._map_to_pandas).collect()
        df_pand = pd.concat(df_pand)
        df_pand.columns = df.columns

        return df_pand

if __name__=='__main__':

    if len(sys.argv)>=3:
        first = sys.argv[1]
        num = sys.argv[2]

    rec = MovieRecSystem()
    rec.train()
    # rec.show_best_model()
    # rec.recommendForAllUsers()
    rec.get_user_movie()

    if str.lower(first[0])=='u':
        rec.recommendForUser(int(first[1:]),int(num))
    elif str.lower(first[0])=='m':
        rec.recommendMovieToUsers(int(first[1:]),int(num))
    else:
        print('Param wrong!')
        exit(1)

    # if len(args)==0:
    #     rec.train()
    #     rec.show_best_model()
    #     rec.recommendForAllUsers()
    # else:
    #     if '-train' in args:
    #         idx = args.index('-train')
    #         if idx+1>=len(args):
    #             print('Params Wrong!')
    #             exit(1)
    #         reTrain = args[idx+1]
    #         if reTrain!='True' or reTrain!='False':
    #             print('Params Wrong!')
    #             exit(1)
    #         rec.train(True if reTrain=='True' else False)
    #     if '-uid' in args:
    #         idx = args.index('-uid')
    #         if idx+1>=len(args):
    #             print('Params Wrong!')
    #             exit(1)
    #         userId = args[idx+1]
    #         rec.recommendForUser(userId)




