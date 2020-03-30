import tensorflow as tf
from tensorflow.python import keras


class LrWithFtrl(object):
    """LR以FTRL方式优化
    """
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Dense(1, activation='sigmoid', input_shape=(121,))
        ])

    @staticmethod
    def read_ctr_records():
        # 定义转换函数,输入时序列化的
        def parse_tfrecords_function(example_proto):
            features = {
                "label": tf.FixedLenFeature([], tf.int64),
                "feature": tf.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features)

            feature = tf.decode_raw(parsed_features['feature'], tf.float64)
            feature = tf.reshape(tf.cast(feature, tf.float32), [1, 121])
            label = tf.reshape(tf.cast(parsed_features['label'], tf.float32), [1, 1])
            return feature, label

        dataset = tf.data.TFRecordDataset(["datas/train_ctr_20190605.tfrecords"])
        dataset = dataset.map(parse_tfrecords_function)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat(10000)
        return dataset

    def train(self, dataset):

        self.model.compile(optimizer=tf.train.FtrlOptimizer(0.03, l1_regularization_strength=0.01,
                                                            l2_regularization_strength=0.01),
                           loss='binary_crossentropy',
                           metrics=['binary_accuracy'])
        self.model.fit(dataset, steps_per_epoch=10000, epochs=10)
        self.model.summary()
        self.model.save_weights('../models/ckpt/ctr_lr_ftrl.h5')

    def predict(self, inputs):
        """预测
        :return:
        """
        # 首先加载模型
        self.model.load_weights('../models/ckpt/ctr_lr_ftrl.h5')
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            predictions = self.model.predict(sess.run(inputs))
        return predictions


if __name__ == '__main__':
    lwf = LrWithFtrl()
    dataset = lwf.read_ctr_records()
    lwf.train(dataset)
    # 10000/10000 [==============================] - 6s 592us/step - loss: 0.3075 - binary_accuracy: 0.9053
    inputs, labels = dataset.make_one_shot_iterator().get_next()
    print(inputs, labels)
    predictions = lwf.predict(inputs)
    print(predictions)