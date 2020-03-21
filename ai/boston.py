from ai.util import Storage
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
import tensorflow as tf

class Boston:
    boston : object
    def __init__(self):
        self.storage = Storage()

    @property
    def boston(self) -> object: return self._boston
    @boston.setter
    def boston(self, boston): self._boston = boston

    def initialize(self):
        this = self.storage # this안으로 storage를 연결해서 누적된 데이터를 그대로 이어받아서 사용. 새롭게 초기화하지 않음. flask내에서 사용할수 있게 편집
        (this.train_X, this.train_Y), (this.test_X, this.test_Y) = boston_housing.load_data()
        #이미 숫자로 프로세싱 끝난 상태여서 iris처럼 head()로는 확인어려움

        """
        보스톤 컬럼명 확인
        crim : crime rate, zn : residential area rate, indus : biz district rate, chas: close river 1 or 0
        nox : nitrous oxide concentration, rm : a number of room per house, age : housing before 1940s rate,
        dis : vocational employment center distance , rad: access to highway, tax, ptratio: teacher per students rate,
        b : black man rate , lstat : lower layer ratio, medv : center value for housing price
        """
        # 초기화 된 부분 까지 데이터가 제대로 들어왔는지 확인 필요
        print(f'확률변수 X의 길이: {len(this.train_X)}')
        print(f'확률변수 Y의 길이: {len(this.train_Y)}')
        print(f'확률변수 X[0]: {this.train_X[0]}')
        print(f'확률변수 Y[0]: {this.train_Y[0]}')

    def standardization(self):  # 데이터 전처리. 정규화. 랜덤값이 아니라 실제값이기때문에 중앙값 기준으로 분포를 정규화
        this = self.storage
        x_mean = this.train_X.mean()
        x_std = this.train_X.std()
        this.train_X -= x_mean
        this.train_X /= x_std
        this.test_X -= x_mean
        this.test_X /= x_std
        y_mean = this.train_Y.mean()
        y_std = this.train_Y.std()
        this.train_Y -= y_mean
        this.train_Y /= y_std
        this.test_Y -= y_mean
        this.test_Y /= y_std

    def new_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=52, activation='relu', input_shape=(13, 1)),
            keras.layers.Dense(units=39, activation='relu'),
            keras.layers.Dense(units=26, activation='relu'),
            keras.layers.Dense(units=1)
        ]) # input_shape가 바뀌면 안되기때문에 13개의 feature를 넣고 1개의 값을 출력하라고 지정. relu로 선형 표현
        # 유닛 갯수는 자의적으로 지정가능하고 맨마지막은 항상 1이어야 함. 순차적으로 활성화함수의 층을 쌓는것

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
        # 최적화는 Adam사용. 학습률 0.07로 지정. loss처리는 mse(표준편차). 활성화함수는 relu -> 기본적으로 활용하는 구조
        #print(f' model summary : {model.summary()}')
        return model

    def learn_model(self, model): # 모델을 학습시키기위해 만들어둔 모델을 집어넣어야 함
        this = self.storage
        history = model.fit(this.train_X, this.train_Y, epochs=25, batch_size=32, validation_split=0.25,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])
        # 학습(fit)시킨 기록을 history에 저장. epochs 25회 돌리기. batch_size 는 분할한 데이터의 갯수(32개씩 25회 뽑아서 학습)
        # validation_split 0.25만 검증용으로 사용, 3/4은 학습용으로 사용
        return history


    def eval_model(self, model):
        this = self.storage
        print(model.evaluate(this.test_X, this.test_Y))  # 위에서 0.25로 지정했으니 25%의 데이터를 평가용으로 활용
        return this

