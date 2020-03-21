from ai.util import Storage
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Population:
    population_inc : [] # 비어있는 리스트 객체이므로 []. wine, iris, boston 데이터는 비어있는 단일객체여서 object로 기재
    population_old : []
    def __init__(self):
        self.storage = Storage()

    @property
    def population_inc(self) -> object: return self._population_inc
    @population_inc.setter
    def population_inc(self, population_inc): self._population_inc = population_inc

    @property
    def population_old(self) -> object: return self._population_old
    @population_old.setter
    def population_old(self, population_old): self._population_old = population_old


    def initialize(self):
        # self라는 공간에 이하의 수치가 들어가므로 이 영역이 static개념이 됨.
        self.population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27,
        0.02, -0.76, 2.66]
        self.population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94,
        12.83, 15.51, 17.14, 14.42]

    def population_without_outlier(self):
        self.population_inc = self.population_inc[:5] + self.population_inc[6:] #5미만, 6이상의 위치에 있는 값을 반환
        self.population_old = self.population_old[:5] + self.population_old[6:] # 위에서 getter/setter로 지정해주었으니 _population_old가 아님

    def population_with_regression(self):
        # 최소제곱법으로 회귀선 구하기
        X = self.population_inc # 세종시가 빠진 데이터를 X에 담음. 집합단위로 움직이는 확률변수 -> 대문자 X, Y
        Y = self.population_old
        x_bar = sum(X) / len(X) # 평균
        y_bar = sum(Y) / len(Y) # 평균
        a = sum([(y - y_bar) * ( x - x_bar) for y, x in list(zip(Y, X))]) # 리스트, zip 에 대한 이해 필요
        a /= sum([(x - x_bar) ** 2 for x in X])
        b = y_bar - a * x_bar
        print('a:', a, 'b:', b)
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = a * line_x + b
        #print('line_x:', line_x, 'line_y', line_y)
        return {'line_x': line_x, 'line_y': line_y}

     #def normalization(self):
     #    pass

    def population_with_regression_using_tf(self):
        # tf 머신러닝을 사용해서 회귀선 구하기. nn은 한개. 단층. 층이 여러개 쌓이면 mlp, 딥러닝.
        X = self.population_inc
        Y = self.population_old
        a = tf.Variable(np.random.randn())
        b = tf.Variable(np.random.randn())
        # 잔차의 제곱의 평균을 반환하는 함수

        def compute_loss():
            y_pred = a * X + b
            loss = tf.reduce_mean((Y - y_pred) ** 2) #변수를 줄여서 (feature를 줄여서) 단일값으로 만드는 것
            return loss

        optimizer = tf.keras.optimizers.Adam(lr=0.07)
        for i in range(1000): #1000번의 루프를 돌려서 1000개의 nn을 만들어냄
            optimizer.minimize(compute_loss, var_list=[a, b])
            if i % 100 == 99:
                print(i, 'a: ', a.numpy(), 'b: ', b.numpy(), 'loss: ', compute_loss().numpy())
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = a * line_x + b
        return {'line_x': line_x, 'line_y': line_y}

    def new_model(self):
        X = instance.population_inc
        Y = instance.population_old
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
            tf.keras.layers.Dense(units=1)
            ])
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
        # mse: mean squared error
        model.fit(X, Y, epochs=10)
        model.predict(X)
        return model

    def predict(self, model):
        # deep learning 회귀선 == mlp
        X = self.population_inc
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = model.predict(line_x)
        return {'line_x': line_x, 'line_y': line_y}

        """
        61행에서 model.fit으로 학습이 되었으므로 learn_model은 필요없어짐. 
           예측이므로 맞는지 틀리는지는 시간이 지나야 알수 있어서 evaluation도 필요없어짐
        def learn_model(self, model): # 모델을 학습시키기위해 만들어둔 모델을 집어넣어야 함
            this = self.storage
            history = model.fit(this.train_X, this.train_Y, epochs=25, batch_size=32, validation_split=0.25)
            return history
        
        def eval_model(self, model):
            this = self.storage
            print(model.evaluate(this.test_X, this.test_Y))
        
    
        def predict(self, model):
            X = self.population_inc
            line_x = np.arange(min(X), max(X), 0.01)
            line_y = model.predict(line_x)
            return {'line_x': line_x, 'line_y': line_y}
        """


class View:
    @staticmethod
    def show_population(instance, dic):
        X = instance.population_inc
        Y = instance.population_old
        line_x = dic['line_x']  #45행의 return {'line_x' : line_x, 'line_y' : line_y} 는 dictionary 구조
        line_y = dic['line_y']
        plt.plot(line_x, line_y, 'r-')  #붉은 실선으로 회귀식 그림
        plt.plot(X, Y, 'bo')
        plt.xlabel('Population Growth Rate (%)')
        plt.ylabel('Elderly Population Rate (%)')
        plt.show()

if __name__ == '__main__':
    instance = Population()
    view = View()
    instance.initialize()
    instance.population_without_outlier()
    #dic = instance.population_with_regression() #dic는 neuron network
    #dic = instance.population_with_regression_using_tf()
    dic = instance.predict(instance.new_model())
    view.show_population(instance, dic)


