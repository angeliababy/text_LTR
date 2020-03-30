import tensorflow as tf
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("user") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')


train = spark.read.parquet('datas/train')
train = train.rdd.map(lambda x:(x.features,x.clicked)).collect()
train = pd.DataFrame(train)
print(train)


def write_to_tfrecords(feature_batch, click_batch):
    """将用户与文章的点击日志构造的样本写入TFRecords文件
    """
    # 1、构造tfrecords的存储实例
    writer = tf.python_io.TFRecordWriter("datas/train_ctr_20190605.tfrecords")

    # 2、循环将所有样本一个个封装成example，写入文件
    for i in range(len(click_batch)):
        # 取出第i个样本的特征值和目标值，格式转换
        click = click_batch[i]
        feature = feature_batch[i].tostring()
        # 构造example
        example = tf.train.Example(features=tf.train.Features(feature={
            "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[click]))
        }))
        # 序列化example,写入文件
        writer.write(example.SerializeToString())

    writer.close()


with tf.Session() as sess:
    # 创建线程协调器
    coord = tf.train.Coordinator()
    # 开启子线程去读取数据
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 存入数据
    write_to_tfrecords(train.iloc[:, 0], train.iloc[:, 1])
    # 关闭子线程，回收
    coord.request_stop()
    coord.join(threads)
