import tensorflow as tf

# - 1、构建TFRecords的输入数据
# - 2、使用模型进行特征列指定
# - 3、模型训练以及预估

FEATURE_COLUMN = ['channel_id', 'vector', 'user_weights', 'article_weights']


class LrWithFtrl(object):
    """LR以FTRL方式训练
    """
    def __init__(self):
        pass

    @staticmethod
    def get_tfrecords_data():

        def parse_example_function(exmaple):
            """解析每个样本的example
            :param exmaple:
            :return:
            """
            # 定义解析格式，parse_single_example
            features = {
                'label': tf.FixedLenFeature([], tf.int64),
                'feature': tf.FixedLenFeature([], tf.string)
            }

            label_feature = tf.parse_single_example(exmaple, features)
            # 修改其中的特征类型和形状
            # 解码 [121]
            # feature = tf.reshape(tf.decode_raw(label_feature['feature'], tf.float32), [1, 121])
            f = tf.decode_raw(label_feature['feature'], tf.float64)
            feature = tf.reshape(tf.cast(f, tf.float32), [1, 121])

            # 计算其中向量、用户权重、文章权重的平均值
            channel_id = tf.cast(tf.slice(feature, [0, 0], [1, 1]), tf.int32)
            vector = tf.reduce_sum(tf.slice(feature, [0, 1], [1, 100]), axis=1)
            user_weights = tf.reduce_sum(tf.slice(feature, [0, 101], [1, 10]), axis=1)
            article_weights = tf.reduce_sum(tf.slice(feature, [0, 111], [1, 10]), axis=1)

            # 4个特征值进行名称构造字典
            data = [channel_id, vector, user_weights, article_weights]
            feature_dict = dict(zip(FEATURE_COLUMN, data))

            label = tf.cast(label_feature['label'], tf.int32)

            return feature_dict, label

        # Tfrecord dataset读取数据
        dataset = tf.data.TFRecordDataset(['datas/train_ctr_20190605.tfrecords'])
        # map 解析
        dataset = dataset.map(parse_example_function)
        dataset = dataset.batch(64)
        dataset = dataset.repeat(10)
        return dataset

    def train_eval(self):
        """
        进行训练
        :return:
        """
        # 指定列特征
        channel_id = tf.feature_column.categorical_column_with_identity('channel_id', num_buckets=25)

        vector = tf.feature_column.numeric_column('vector')
        user_weights = tf.feature_column.numeric_column('user_weights')
        article_weights = tf.feature_column.numeric_column('article_weights')

        columns = [channel_id, vector, user_weights, article_weights]

        # LinearClassifier
        model = tf.estimator.LinearClassifier(feature_columns=columns,
                                              optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                               l1_regularization_strength=10,
                                                                               l2_regularization_strength=10),
                                              model_dir='../models/ckpt/lr_ftrl')
        model.train(LrWithFtrl.get_tfrecords_data, steps=100)
        result = model.evaluate(LrWithFtrl.get_tfrecords_data)
        print(result)

        # 模型导入
        feature_spec = tf.feature_column.make_parse_example_spec(columns)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel("./serving_model/wdl1/", serving_input_receiver_fn)

    def predict(self, inputs):
        # 指定列特征
        channel_id = tf.feature_column.categorical_column_with_identity('channel_id', num_buckets=25)

        vector = tf.feature_column.numeric_column('vector')
        user_weights = tf.feature_column.numeric_column('user_weights')
        article_weights = tf.feature_column.numeric_column('article_weights')

        columns = [channel_id, vector, user_weights, article_weights]

        # LinearClassifier
        model = tf.estimator.LinearClassifier(feature_columns=columns,
                                              optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                               l1_regularization_strength=10,
                                                                               l2_regularization_strength=10),
                                              model_dir='../models/ckpt/lr_ftrl')

        predictions = model.predict(inputs, checkpoint_path='../models/ckpt/lr_ftrl')

        return predictions


if __name__ == '__main__':
   lw =  LrWithFtrl()
   print(lw.get_tfrecords_data())
   model = lw.train_eval()

# {'accuracy': 0.9046435, 'accuracy_baseline': 0.9046434, 'auc': 0.57956487, 'auc_precision_recall': 0.12670927, 'average_loss': 0.31273547, 'label/mean': 0.095356554, 'loss': 19.850473, 'precision': 0.0, 'prediction/mean': 0.111656144, 'recall': 0.0, 'global_step': 100}