import numpy as np

class Adaline:
    def __init__(self, eta=0.01, n_iter=50, random_state=None, shuffle=True): #랜덤을 섞는것
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def _shuffle(self, X, y): # 외부에서 사용하지 않을때 앞에 _ 를 사용
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weight(self, m):
        """랜덤한 작은 수로 가중치를 초기화"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def activation(self, X):
        return X # 선형활성화

    def _update_weights(self, xi, target):
        """아달린 학습 규칙을 적용하기 위해 가중치 업데이트 함"""
        # eta : 학습률 0.0 ~ 1.0
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def fit(self, X, y):  # 최적화. 대문자 X : 집합. 소문자 y : 특정값
        self._initialize_weight(X.shape[1]) # 여러개의 x값들(집합) 중에서 하나만 뽑아냄
        self.cost_ = []  # error대신에 cost라는 개념을 사용

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = [] # cost를 초기화.
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target)) # 가중치가 계속 누적
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost) # 손실함수 계속 누적
        return self

    def partial_fit(self, X, y): #부분최적화
        """가중치를 다시 초기화 하지 않고 훈련데이터를 학습"""
        if not self.w_initialized:
            self._initialize_weight(X.shape[1]) # 여러개의 x값들(집합) 중에서 하나만 뽑아냄
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target) # 가중치를 계속 수정정        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        """최종입력계산"""
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.01, 1, -1)