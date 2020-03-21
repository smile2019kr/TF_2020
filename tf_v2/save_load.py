import os
import tensorflow as tf
from tensorflow import keras

class SaveLoad:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.new_model = None
    def execute(self):
        self.download_dataset()
        self.create_model()
        self.train_model()
        self.save_model()
        self.load_model()
    def download_dataset(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) \
            = tf.keras.datasets.mnist.load_data()
        self.train_labels = self.train_labels[:1000]
        self.test_labels = self.test_labels[:1000]
        self.train_images = self.train_images[:1000].reshape(-1, 28 * 28) / 255.0
        self.test_images = self.test_images[:1000].reshape(-1, 28 * 28) / 255.0
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print(self.model.summary())
    def train_model(self):
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # 체크포인트 콜백 만들기
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        self.model.fit(self.train_images, self.train_labels, epochs=10,
                  validation_data=(self.test_images, self.test_labels),
                  callbacks=[cp_callback])  # 훈련 단계에 콜백을 전달합니다
        self.model.load_weights(checkpoint_path) # 가중치 추가
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
        # 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
        checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            # 다섯 번째 에포크마다 가중치를 저장합니다
            period=5)
        self.model.save_weights(checkpoint_path.format(epoch=0))
        self.model.fit(self.train_images, self.train_labels,
                  epochs=50, callbacks=[cp_callback],
                  validation_data=(self.test_images, self.test_labels),
                  verbose=0)
        self.model.fit(self.train_images, self.train_labels, epochs=5)
    def save_model(self):
        # 전체 모델을 HDF5 파일로 저장합니다
        self.model.save('saved/my_model.h5')
        print('======= 모델 훈련 종료 ======')
    def load_model(self):
        self.new_model = keras.models.load_model('saved/my_model.h5')
        self.new_model.summary()
        loss, acc = self.new_model.evaluate(self.test_images,
                                            self.test_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
