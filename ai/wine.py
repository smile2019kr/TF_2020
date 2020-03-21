from ai.util import Storage
import pandas as pd
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
import tensorflow as tf

class Wine:
    Wine : object
    def __init__(self):
        self.storage = Storage()

    @property
    def wine(self) -> object: return self._wine
    @wine.setter
    def wine(self, wine): self._wine = wine

    def initialize(self):
        red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
        white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

    def normalization(self):
        pass

    def new_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12, )),
            tf.keras.layers.Dense(units=24, activation='relu' ),
            tf.keras.layers.Dense(units=12, activation='relu' ),
            tf.keras.layers.Dense(units=2, activation='softmax'),
        ]) # 겹겹이 순차적으로 층을 쌓아올려야 하므로 list구조
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print(f'model summary: {model.summary()}')
        return model

    def learn_model(self, model): # 모델학습에서는 기록을 누적하므로 history 가 필요
        this = self.storage
        history = model.fit(this.train_X, this.train_Y, epochs=25, batch_size=32, validation_split=0.25)
        #Adam 을 최적화함수로 사용하였으므로 몇개로 쪼갤지 batch_size가 나와야함
        return history

    def eval_model(self, model):
        this = self.storage
        model.evaluate(this.test_X, this.test_Y)

if __name__ == '__main__':
    wine = Wine()
    wine.initialize()
    wine.normalization()
    model = wine.new_model()
    history = wine.learn_model(model)
    wine.eval_model(model)
