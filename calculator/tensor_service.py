import tensorflow as tf


class TensorService:
    def __init__(self, payload):
        self._num1 = payload.num1  # 모델에 있는 속성값
        self._num2 = payload.num2  # payload.num1과 payload.num2는 사칙연산에서 각각 공유하는 값

    @tf.function  # @tf.function를 추가함으로써 인공지능함수가 되었음
    def add(self):
        return tf.add(self._num1 , self._num2)

    @tf.function
    def subtract(self):
        return tf.subtract(self._num1 , self._num2)

    @tf.function
    def multiply(self):
        return tf.multiply(self._num1 , self._num2)

    @tf.function
    def divide(self):
        return tf.divide(self._num1 , self._num2)



