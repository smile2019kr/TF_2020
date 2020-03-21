from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai.util import Storage
from ai.perceptron import Perceptron
from ai.adaline import Adaline

class Iris:
    iris : object # iris를  Iris 클래스 내에서만 사용하기 위함. @statistics와 동일한 역할. object. 객체. (property가 아님)
    def __init__(self):
        self.storage = Storage()
        self.neural_network = None

    @property
    def iris(self) -> object: return self._iris
    @iris.setter
    def iris(self, iris): self._iris = iris

    def initialize(self):  # 돌릴때마다 초기화되지 않도록 5행에서 iris를 object로 지정해주고 초기화하는 함수는 별도로 지정
        self.iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/iris/iris.data', header=None)

        print(self.iris.head())
        print(self.iris.columns)
        """결과값
                     0    1    2    3            4
        0  5.1  3.5  1.4  0.2  Iris-setosa
        1  4.9  3.0  1.4  0.2  Iris-setosa
        2  4.7  3.2  1.3  0.2  Iris-setosa
        3  4.6  3.1  1.5  0.2  Iris-setosa
        4  5.0  3.6  1.4  0.2  Iris-setosa
        Int64Index([0, 1, 2, 3, 4], dtype='int64')
        
        => Iris-setosa 가 Y값 (품종. 정답). 0~3까지의 값이 feature
        
        """
        # setosa 와 versicolor 선택

        """
        타겟 데이터
        setosa, versicolor, virginica의 세가지 붓꽃 종(species)
        feature
        꽃받침 길이(Sepal Length)
        꽃받침 폭(Sepal Width)
        꽃잎 길이(Petal Length)
        꽃잎 폭(Petal Width)
        """
        # setosa 와 versicolor 선택

        temp = self.iris.iloc[0:100, 4].values
        this = self.storage
        this.train_Y = np.where(temp == 'Iris-setosa', -1, 1) # -1부터 1까지 (=all)
        # 꽃받침 길이와 꽃잎 폭 선택

        this.train_X = self.iris.iloc[0:100, [0,2]].values
        self.neural_network = Perceptron(eta=0.1, n_iter=10)
        # eta: lr(learning rate로 진화) n_iter : 몇회전 할지 회전횟수 지정(epoch로 진화)
        # 퍼셉트론 안에는 뉴런(w*x + b)덩어리들이 있음(여러번 돌렸으므로).

    def show_scatter(self):
        this = self.storage
        X = this.train_X
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()

    def show_errors(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y # 여러 y값들중에서 하나이므로 소문자 y
        self.neural_network.fit(X, y) # 어떤 클래스안에 neural_network 가 있는데 앞에 준 값이 feature, 뒤에 준값이 정답
        plt.plot(range(1, len(self.neural_network.errors_) + 1),
                 self.neural_network.errors_, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number or errors')
        plt.show()

    def show_decision_tree(self):
        this = self.storage  # 데이터값을 그대로 유지하므로 show_errors 함수와 동일한 코딩이 많음
        X = this.train_X
        y = this.train_Y
        nn = self.neural_network
        nn.fit(X, y)
        colors = ('red', 'blue', 'rightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))]) # y값에 따라 중복되지않게 하나씩 색깔을 넣으라는 것
        x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max()+1, #min, max 최소/최대범위
        x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max()+1, #min, max 최소/최대범위
        resolution = 0.2
        """
        numpy 모듈의 arange 함수는 반열린 구간 [start, stop] 에서 
        step의 크기만큼 일정하게 떨어져있는 숫자들을 array 형태로 반환해주는 함수
        (시작/끝 사이의 구간을 step크기만큼 여러개로 분할)
        
        meshgrid 명령은 사각형 영역을 구성하는 가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서 
        이 사각형 영역을 이루는 조합을 출력한다
        결과는 그리드 포인트의 x값만을 표시하는 행렬과 
        y값만을 표시하는 행렬 두개로 분리하여 출력한다
        (matrix 이므로 벡터로 받아서 조합을 출력. 이미지 활용시 중요)
        """

        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution),
        ) # 최소/최대값 사이를 0.2 로 분할
        Z = nn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], label=cl, edgecolors=['black'])
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()

    def show_adaline(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        self.neural_network = Adaline(eta=0.01, n_iter=50, random_state=1)
        self.neural_network.fit(X_std, y) #카피본을 활용하여 적합도 검증
        self.show_decision_tree()
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()




