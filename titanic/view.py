from titanic.service import Service
from titanic.entity import Entity
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
rc('font', family = font_manager.FontProperties(fname='C:/Windows/Fonts/H2GTRE.ttf').get_name())
#폰트 깨지는것을 방지

class View: #차트 통해서 시각화
    def __init__(self, fname):
        service = Service()
        entity = Entity()
        entity.context = './data/'
        entity.fname = fname
        self._entity = service.new_model(entity)

    def plot_survived_dead(self):
        this = self._entity
        f, ax = plt.subplots(1, 2, figsize=(18, 8)) #자동완성으로 plt.subplot()가 뜨지만 subplots()이므로 주의
        this['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        ax[0].set_title('0.사망자 vs 1.생존자')
        ax[0].set_ylabel('')
        ax[1].set_title('0.사망자 vs 1.생존자')
        sns.countplot('Survived', data=this, ax=ax[1])
        plt.show()

    def plot_sex(self):
        this = self._entity
        f, ax = plt.subplots(1, 2, figsize=(18, 8))  # 자동완성으로 plt.subplot()가 뜨지만 subplots()이므로 주의
        this['Survived'][this['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        this['Survived'][this['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
        #this['Survived'][] 생존자중에서 성별이 남자인 -> 2차원
        # this['Survived'][][] 생존자중에서 성별이 남자이고 $$이 $$인 -> 3차원
        ax[0].set_title('남성의 생존비율 [0.사망자 vs 1.생존자]') # 차원이 하나 추가됨
        ax[1].set_title('여성의 생존비율 [0.사망자 vs 1.생존자]')
        plt.show()

