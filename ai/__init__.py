import tensorflow as tf
import tensorflow_hub as hub
from ai.view import View
from ai.neuron import Neuron
from ai.iris import Iris
from ai.boston import Boston
from ai.wine import Wine
from ai.population import Population

if __name__ == '__main__':
    def print_menu():
        print('0. Show version')
        print('1. Random Number')
        print('2. Neuron')
        print('3. Iris')
        print('4. Boston')
        print('5. Wine')
        print('6. Population')
        return input('Select Menu \n')
    view = View()

    while 1:
        menu = print_menu()
        if menu == '1':
            neu = Neuron()
            rand = neu.new_random_uniform_number(100) # 100차원을 선으로 연결
            view.show_hist(rand)
            view.show_line(rand)
            rand = neu.new_random_normal_number(100)
            view.show_hist(rand)
            view.show_blot(rand)

        if menu == '2':
            neu = Neuron()
            neuron = neu.new_neuron()
            dic = neu.sigmoid_tanh_relu()
            view.show_sigmoid_tanh_relu(dic)

        if menu == '3': # 객체만들고 초기화 시킨 다음에 산포도 플롯.
            iris = Iris()
            iris.initialize()
            #iris.show_scatter()
            #iris.show_errors()
            #iris.show_decision_tree() 퍼셉트론용의 decision tree
            iris.show_adaline()  #adaline 내부에 decision tree가 들어가있음

        if menu == '4':
            boston = Boston()
            boston.initialize()
            boston.standardization()
            model = boston.new_model()
            history = boston.learn_model(model)
            storage = boston.eval_model(model)
            view.show_history(history) #history를 전달하는 차트 생성
            view.show_boston({'model': model, 'storage': storage}) # 학습된 것을 보기위해서 설정

        if menu == '5':
            wine = Wine()
            wine.initialize()
            wine.normalization()
            model = wine.new_model()
            history = wine.learn_model(model)
            wine.eval_model(model)

        if menu == '6':
            population = Population()
            population.initialize()
            population.normalization()
            model = population.new_model()
            history = population.learn_model(model)
            population.eval_model(model)

        elif menu == '0':
            print(f'버젼 : {tf.__version__}')
            print(f'즉시실행모드 : {tf.executing_eagerly}')
            """ 
            결과값
            0.Show version
            메뉴 입력 
            0
            버젼 : 2.1.0
            """
            print(f'허브버젼 : {hub.__version__}')
            print(f'GPU', '사용가능' if tf.config.experimental.list_physical_devices('GPU') else '사용불가능')


