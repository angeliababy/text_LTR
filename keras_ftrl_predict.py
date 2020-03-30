# coding=utf-8
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("user") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from keras_ftrl import LrWithFtrl


user_id = 1112727762809913344
channel_id = 17
reco_set = [104454, 11078, 14335, 4402, 2, 14839, 44024, 18427, 43997, 17375]

def feature(row):
    user_features = row.user_features[int(channel_id)]
    return user_features

def vector_list(row):
    return list(row.article_features)

user_feature = spark.read.parquet(r'datas\user_features')
article_feature = spark.read.parquet(r'datas\article_features')


def lrftrl_sort_service(reco_set, user_feature, article_feature):
    """
    排序返回推荐文章
    :param reco_set:召回合并过滤后的结果
    :param temp: 参数
    :param hbu: Hbase工具
    :return:
    """
    print(344565)
    # 排序
    # 1、读取用户特征中心特征
    user_feature = user_feature.filter(user_feature.user_id == user_id).select(user_feature.user_features).rdd.map(
        feature).collect()
    from itertools import chain
    user_feature = list(chain.from_iterable(user_feature))
    print(user_feature)


    if user_feature and reco_set:
        # 2、读取文章特征中心特征
        result = []
        for article_id in reco_set:
            try:
                article_feature = article_feature.filter(article_feature.article_id == str(article_id)).rdd.map(vector_list).collect()
                from itertools import chain
                article_feature = list(chain.from_iterable(article_feature))
            except Exception as e:
                article_feature = []

            if not article_feature:
                article_feature = [0.0] * 111
            f = []
            f.extend(user_feature)
            f.extend(article_feature)

            result.append(f)



    if result:
        # 4、预测并进行排序筛选
        arr = np.array(result)
        print(arr)

        # 加载逻辑回归模型
        lwf = LrWithFtrl()
        print(tf.convert_to_tensor(np.reshape(arr, [len(reco_set), 121])))
        predictions = lwf.predict(tf.constant(arr))
        print(predictions)

        df = pd.DataFrame(np.concatenate((np.array(reco_set).reshape(len(reco_set), 1), predictions),
                                          axis=1),
                          columns=['article_id', 'prob'])

        df_sort = df.sort_values(by=['prob'], ascending=True)

        # 排序后，只将排名在前100个文章ID返回给用户推荐
        if len(df_sort) > 100:
            reco_set = list(df_sort.iloc[:100, 0])
        else:
            reco_set = list(df_sort.iloc[:, 0])
        print(reco_set)

    return reco_set


if __name__ == '__main__':
    reco_set = lrftrl_sort_service(reco_set, user_feature, article_feature)
    print(reco_set)