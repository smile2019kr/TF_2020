import tensorflow as tf
import math
import numpy as np


class Neuron:
    @staticmethod
    def new_random_uniform_number(dim):
        #균일분포 : max, min 사이값 동일분포에서 수 추출
        #[1] 결곽밧 shape 행, 열 차원의 수 2*3
        rand = tf.random.uniform([dim], 0, 1)
        print(rand)
        return rand

    @staticmethod
    def new_random_normal_number(dim):
        rand = tf.random.uniform([dim], 0, 1)
        print(rand)
        return rand

    @staticmethod
    def sigmoid(x):
        return  1 / (1 + math.exp(-x)) # 항상 1보다 작은값이 나옴

    def new_neuron(self):
        x = 0
        y = 1
        w = tf.random.normal([1], 0, 1)
        b = tf.random.normal([1], 0, 1)
        for i in range(1000):
            neuron = self.sigmoid(x * w + 1 * b)
            error = y - neuron
            w = w + x * 0.1 * error
            b = b + 1 * 0.1 * error

            if i % 100 == 99:
                print(i, error, neuron)
                # 1에 근사한 값들이 loss는 줄어들고 정확도는 올라가는 방향으로 나오다가 특정값으로 나오게하는것
        return neuron

    def sigmoid_tanh_relu(self): # 2020년 tf에서는 relu 함수를 사용
        x = np.arange(-5, 5, 0.01)
        sigmoid_x = [self.sigmoid(z) for z in x]
        tanh_x = [math.tanh(z) for z in x]
        relu = [0 if z < 0 else z for z in x]
        return {'x': x, 'sigmoid_x': sigmoid_x, 'tanh_x': tanh_x, 'relu': relu}





