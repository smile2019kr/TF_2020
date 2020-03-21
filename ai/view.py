import matplotlib.pyplot as plt
import tensorflow as tf
class View:
    @staticmethod
    def show_hist(rand):
        plt.hist(rand, bins=100)
        plt.show()
    @staticmethod
    def show_line(rand):
        plt.plot(range(100), rand)
        # ValueError: x and y must have same first dimension, but have shapes (20,) and (100,)
        plt.show()
    @staticmethod
    def show_blot(rand):
        plt.plot(range(100), rand, 'bo')
        """
        bo : blue dot
        b- : blue line
        b-- : blue 
        ro : red dot
        """
        plt.show()
    """
    @staticmethod
    def show_population(dic):
        population_inc = dic['population_inc']
        population_old = dic['population_old']
        plt.plot(population_inc, population_old, 'bo')
        plt.xlabel('Population Growth Rate (%)')
        plt.ylabel('Elderly Population Rate (%)')
        plt.show()
    """
    @staticmethod
    def show_population(model, dic):
        X = model.population_inc
        Y = model.population_old
        line_x = dic['line_x']
        line_y = dic['line_y']
        # 붉은색 실선으로 회귀선을 그립니다.
        plt.plot(line_x, line_y, 'r-')
        plt.plot(X, Y, 'bo')
        plt.xlabel('Population Growth Rate (%)')
        plt.ylabel('Elderly Population Rate (%)')
        plt.show()
    @staticmethod
    def show_history(history):
        plt.plot(history.history['loss'], 'b-', label='loss')
        plt.plot(history.history['val_loss'], 'r--', label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    @staticmethod
    def show_sigmoid_tanh_relu(dic):
        x = dic['x']
        sigmoid_x = dic['sigmoid_x']
        tanh_x = dic['tanh_x']
        relu = dic['relu']
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.plot(x, sigmoid_x, 'b-', label='sigmoid')
        plt.plot(x, tanh_x, 'r--', label='tanh')
        plt.plot(x, relu, 'g.', label='relu')
        plt.legend()
        plt.show()
    @staticmethod
    def show_boston(dic):
        model = dic['model']
        this = dic['storage']
        test_X = this.test_X
        test_Y = this.test_Y
        pred_Y = model.predict(test_X)
        plt.figure(figsize=(5, 5))
        plt.plot(this.test_Y, pred_Y, 'b.')
        plt.axis([min(this.test_Y), max(test_Y), min(test_Y), max(test_Y)])
        plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls="--", c=".3")
        plt.xlabel('test_Y')
        plt.ylabel('pred_Y')
        plt.show()
    @staticmethod
    def show_wine_history(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='loss')
        plt.plot(history.history['val_loss'], 'r--', label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'g-', label='accuracy')
        plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0.7, 1)
        plt.legend()
        plt.show()
    @staticmethod
    def show_wine_qty_history(history):
        # 5.17 다항 분류 모델 학습 결과 시각화
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='loss')
        plt.plot(history.history['val_loss'], 'r--', label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'g-', label='accuracy')
        plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0.7, 1)
        plt.legend()
        plt.show()
    @staticmethod
    def show_fashion_history(history):
        # 5.24 Fashion MNIST 분류 모델 학습 결과 시각화
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='loss')
        plt.plot(history.history['val_loss'], 'r--', label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'g-', label='accuracy')
        plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0.7, 1)
        plt.legend()
        plt.show()
    @staticmethod
    def show_fashion_cnn_history(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='loss')
        plt.plot(history.history['val_loss'], 'r--', label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'g-', label='accuracy')
        plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0.7, 1)
        plt.legend()
        plt.show()
    @staticmethod
    def show_cat():
        # No module named 'PIL' => pip install pillow
        image_path = tf.keras.utils.get_file('cat.jpg', 'http://bit.ly/33U6mH9')
        image = plt.imread(image_path)
        titles = ['RGB Image', 'Red channel', 'Green channel', 'Blue channel']
        cmaps = [None, plt.cm.Reds_r, plt.cm.Greens_r, plt.cm.Blues_r]
        from numpy import array, zeros_like
        def channel(image, color):
            if color not in (0, 1, 2): return image
            c = image[..., color]
            z = zeros_like(c)
            return array([(c, z, z), (z, c, z), (z, z, c)][color]).transpose(1, 2, 0)
        colors = range(-1, 3)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3))
        objs = zip(axes, titles, colors)
        for ax, title, color in objs:
            ax.imshow(channel(image, color))
            ax.set_title(title)
            ax.set_xticks(())
            ax.set_yticks(())
        plt.show()