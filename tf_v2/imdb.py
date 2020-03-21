import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
class Imdb:
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.model = None
    def download_data(self):
        # 훈련 세트를 6대 4로 나눕니다.
        # 결국 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
        train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
        (self.train_data, self.validation_data), self.test_data = tfds.load(
            name="imdb_reviews",
            split=(train_validation_split, tfds.Split.TEST),
            as_supervised=True)
    def env_info(self):
        print("버전: ", tf.__version__)
        print("즉시 실행 모드: ", tf.executing_eagerly())
        print("허브 버전: ", hub.__version__)
        print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")
    def create_sample(self):
        train_examples_batch, train_labels_batch = next(iter(self.train_data.batch(10)))
        return train_examples_batch
    def create_model(self):
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[],
                                   dtype=tf.string, trainable=True)
        hub_layer(self.create_sample()[:3])
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu')) # 은닉층 16개
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # 이미지분석: softmax 사용, 텍스트분류: sigmoid 사용 -> 분석내용에 따라 출력층에서 쓰는 함수가 달라짐
        # model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
    def train_model(self):
        self.model.fit(self.train_data.shuffle(10000).batch(512),
                            epochs=20,
                            validation_data=self.validation_data.batch(512),
                            verbose=1)
    def eval_model(self):
        results = self.model.evaluate(self.test_data.batch(512), verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f" % (name, value))