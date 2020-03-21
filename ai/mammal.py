import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

"""
[[0,0], -> [1, 0, 0] 기타
 [1,0], -> [0, 1, 0] 포유류
 [1,1], -> [0, 0, 1] 조류
 [0,0], -> [1, 0, 0] 기타
 [0,0], -> [1, 0, 0] 기타
 [0,1]  -> [0, 0, 1] 조류
"""

class Mammal:
    @staticmethod
    def execute():
        # [털, 날개] -> 기타, 포유류, 조류
        x_data = np.array(
            [[0, 0],
             [1, 0],
             [1, 1],
             [0, 0],
             [0, 0],
             [0, 1]
             ]
        )
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )

        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_uniform([2, 3], -1, 1.))
        # -1 all
        # nn은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2,3]
        # tf에는 변수타입 3개. placeholder는 쓰지않게 되었으나 1년은 사용될듯.
        # tf.placeholder 는 개발자가 지정해주는 값. tf.Variable는 tf내부에서 주는값.
        # W 는 내부에서 계속 바뀌는 값. X값은 리스트로 있고 내부에 투입되면 상수처럼 처리됨.
        # 내부에서는 W가 변수처럼 계속 바뀜
        # nn은 W의 수식안에 있는 tf 안에 존재 (tf내부값)

        b = tf.Variable(tf.zeros([3]))
        # b는 편향 bias
        # b는 각 레이어의 아웃풋 갯수로 결정함

        L = tf.add(tf.matmul(X, W), b)  # X와 W의 곱을 b와 합산
        L = tf.nn.relu(L) # activation 함수인 relu추가
        model = tf.nn.softmax(L)
        """
        softmax 함수는 다음 처럼 결과값을 전체 합이 1인 확률로 만들어주는 함수 (scaling)
        예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]                  
        """
        print(f'모델 내부 보기 {model}')
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 경사하강법. 미분으로 기울기가 0이 되는 순간을 잡아내는 것
        train_op = optimizer.minimize(cost)
        #비용함수를 최소화시키면 (= 경사도를 0으로 만들면) 그 값이 최적화 된다
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in range(100):
            sess.run(train_op, {X: x_data, Y: y_data})
            if (step + 1) % 10 == 0:
                print(step +1, sess.run(cost, {X: x_data, Y: y_data}))

        # 결과 확인
        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print(f' 예측값 {sess.run(prediction, {X: x_data})}')
        print(f' 실제값 {target, { Y: y_data}}')
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print(f' 정확도: %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))

if __name__ == '__main__':
    Mammal.execute()
    # static으로 걸어두었으므로 생성자 없이 직접실행 가능