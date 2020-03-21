class Service:
    def __init__(self, payload): # 연산용의 속성값을 담아두는 공간 payload(모델). payload라는 공간을 만들어놓고 사칙연산을 수행하게 함
        self._num1 = payload.num1 # 모델에 있는 속성값
        self._num2 = payload.num2  #payload.num1과 payload.num2는 사칙연산에서 각각 공유하는 값

    def add(self):
        return self._num1 + self._num2

    def subtract(self):
        return self._num1 - self._num2

    def multiply(self):
        return self._num1 * self._num2

    def divide(self):
        return self._num1 / self._num2

