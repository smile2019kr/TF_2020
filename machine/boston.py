from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from tensorflow import keras
from machine.random_number_maker import RandomNumberMaker
import numpy as np
import math
import matplotlib.pyplot as plt

class Boston:
    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = boston_housing.load_data()

    def show_dataset(self):
        print(f'확률변수 X 길이 : {len(self.train_X)}')
        print(f'확률변수 Y 길이 : {len(self.train_Y)}')
        print(f'확률변수 X[0] : {self.train_X[0]}')
        print(f'확률변수 Y[0] : {self.train_Y[0]}')

        """결과값
        확률변수 X 길이 : 404
        확률변수 Y 길이 : 404
        확률변수 X[0] : [  1.23247   0.        8.14      0.        0.538     6.142    91.7
           3.9769    4.      307.       21.      396.9      18.72   ]
        확률변수 Y[0] : 15.2
        """

    def preprocessing(self):  # 데이터 전처리. 정규화
        train_X = self.train_X
        train_Y = self.train_Y
        test_X = self.test_X
        test_Y = self.test_Y

        x_mean = train_X.mean()
        x_std = train_X.std()
        train_X -= x_mean
        train_X /= x_std
        test_X -= x_mean
        test_X /= x_std

        y_mean = train_Y.mean()
        y_std = train_Y.std()
        train_Y -= y_mean
        train_Y /= y_std
        test_Y -= y_mean
        test_Y /= y_std

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(units=52, activation='relu', input_shape=(13, )),
            keras.layers.Dense(units=39, activation='relu'),
            keras.layers.Dense(units=26, activation='relu'),
            keras.layers.Dense(units=1) # 맨마지막 출력층은 하나로 제시. 맞건 틀리건간에 결과값은 하나를 내야한다는 것.
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
        #print(model.summary())

        """결과값
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense (Dense)                (None, 52)                728       
        _________________________________________________________________
        dense_1 (Dense)              (None, 39)                2067      
        _________________________________________________________________
        dense_2 (Dense)              (None, 26)                1040      
        _________________________________________________________________
        dense_3 (Dense)              (None, 1)                 27        
        =================================================================
        Total params: 3,862
        Trainable params: 3,862
        Non-trainable params: 0
        _________________________________________________________________
        None
        """

        x = np.arange(-5, 5, 0.01)
        sigmoid_x = [RandomNumberMaker.sigmoid(z) for z in x]  #sigmoid가 static으로 걸려있으므로 이름으로 바로 호출해서 []안에 사용가능
        tanh_x = [math.tanh(z) for z in x]
        relu = [0 if z < 0 else z for z in x]
        plt.axhline(0, color = 'gray') # ax 축이 horizon 수평
        plt.axvline(0, color = 'gray') # ax 축이 vertical 수직
        plt.plot(x, sigmoid_x, 'b-', label='sigmoid') # 시그모이드 함수를 플롯
        plt.plot(x, tanh_x, 'r--', label='tanh')
        plt.plot(x, relu, 'g.', label='relu')
        plt.legend()
        plt.show()



