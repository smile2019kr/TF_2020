import tensorflow as tf
from tensorflow import keras
# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


class FashionMnist:
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.mode = None

    def show_dataset(self):
        print(f'훈련이미지 : {self.train_images.shape}') #10개카테고리 * 6만개 훈련이미지가 몇행몇열인지 출력
        print(f'훈련이미지 수 : {len(self.train_labels.shape)}') # 6만개
        print(f'훈련라벨 : {self.train_labels}')
        print(f'테스트이미지 : {self.test_images.shape}')
        print(f'테스트이미지 수 : {len(self.test_labels.shape)}')
        print(f'테스트라벨 : {self.test_labels}')
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.show()

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.train_images, self.train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        self.model = model  # 완성된 구조를 저장하는 공간 생성
        print(f'\n 테스트 정확도 {test_acc}')

        """
        결과값
        loss: 0.5860 - accuracy: 0.7897        
         테스트 정확도 0.7896999716758728
         
        """

    def predict_image(self):
        predictions = self.model.predict(self.test_images)
        print(f'예측값 : {predictions[0]}')
        print('가장 신뢰도가 높은 레이블: %s' % np.argmax(predictions[0]))
        return predictions

        """
        테스트 정확도 0.8180000185966492
        예측값 : [6.0337005e-16 5.1523434e-15 2.9218971e-22 2.1057731e-10 2.1181359e-21
         4.9592618e-02 2.5850884e-12 1.6306809e-01 2.5521416e-08 7.8733927e-01]
        가장 신뢰도가 높은 레이블: 9 
          -> 앵클부츠
        """

    def subplot_test(self, predictions):
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_image(i, predictions, self.test_labels, self.test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_value_array(i, predictions, self.test_labels)
        plt.show()

    def plot_value_array(selfself, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0,1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array) # 모델이 예측한 값. 정답인지 아닌지는 모르는 상황
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel('{} {:2.0f}% ({}'.format(self.class_names[predicted_label],
                   100*np.max(predictions_array), self.class_names[true_label], color = color))

    def one_test(self, input):
        img = self.test_images[input]
        img = (np.expand_dims(img, 0))
        predictions_single = self.model.predict(img)
        self.plot_value_array(0, predictions_single, self.test_labels)
        _ = plt.xticks(range(10), self.class_names, rotation=45)
        return np.argmax(predictions_single[0])

        """
        결과값
        10000/10000 - 0s - loss: 0.6113 - accuracy: 0.7891

         테스트 정확도 0.7890999913215637
        3
        """

