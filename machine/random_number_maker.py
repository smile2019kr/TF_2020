import tensorflow as tf
import math

class RandomNumberMaker:
    def fetch_randomnumber(self): #fetch : -을 가져오게 하는것
        rand = tf.random.uniform([1], 0, 1)  # 균일분포. np로 만들어낸 랜덤배열보다 tf로 만들어내는게규칙성이 있게 조정가능
        print(rand)
        rand = tf.random.uniform([4], 0, 1)  # 여러개, 균일분포.
        print(rand)
        rand = tf.random.normal([4], 0, 1)  # 여러개, 정규분포. np로 만들어낸 랜덤배열보다 규칙성이 있게 조정가능
        print(rand)
        return rand

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def create_neuron(self):
        x = 1
        y = 0
        w = tf.random.normal([1], 0, 1) #정규분포를 따르는 랜덤 숫자를 하나만 추출
        output = self.sigmoid(x*w)
        print(output)
        # 경사하강법을 이용한 뉴런 학습
        for i in range(1000):
            output = self.sigmoid(x * w)
            error = y - output
            w - w + y * 0.1 * error

            if i % 100 == 99:
                print(i, error, output)

        return output



