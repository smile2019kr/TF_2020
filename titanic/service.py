from titanic.entity import Entity
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score  #값을 크로스체크하는 클래스

"""
PassengerId, 고객ID
Survived, 생존여부
Pclass, 승선권 1=1등석, 2=2등석, 3=3등석
Name,
Sex,
Age, 
SibSp,  동반한 형제, 자매, 배우자
Parch,  동반한 부모, 자식
Ticket, 티켓번호
Fare,   요금
Cabin,  객실번호
Embarked  승선한 항구명 C=쉐브루, Q=퀸즈타운, S=사우스햄튼


print(f'결정트리 활용한 검증 정확도 {None}')
print(f'랜덤포레스트 활용한 검증 정확도 {None}')
print(f'나이브베이즈 활용한 검증 정확도 {None}')
print(f'KNN 활용한 검증 정확도 {None}')
print(f'SVM 활용한 검증 정확도 {None}')
"""

class Service:
    def __init__(self):
        self._this = Entity() # 이 안에서 사용하는 entity는 모두 this로 지정. 충돌날수 있으니 언더바 설정

    def new_model(self, payload):
        this = self._this
        this.context = './data/'  #텍스트마이닝에서는 txt파일을 불러들임. 정제되지않은값. csv파일은 dictionary구조. 객체모델로 됨
        this.fname = payload # 22행에서 넘어온 str값이 됨. train.csv
        return pd.read_csv(this.context + this.fname) # csv이므로 바로 객체로 넘겨버림.

    @staticmethod # self를 사용하지않는 경우, self를 삭제하고 @staticmethod를 추가.
    def create_train(this):
        return this.train.drop('Survived', axis=1) #훈련용 객체에서는 생존여부 열을 drop

    @staticmethod
    def create_label(this):
        return this.train['Survived']
    # train과 label이 있으므로 비지도학습을 하겠다는 것

    #feauture 가 너무 많으면 성능 낮아지므로 정리
    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        this.test = this.test.drop([feature], axis=1)
        return this #필요없는 feature를 날린 상태에서 객체를 반환

    @staticmethod
    def embarked_norminal(this) -> object:
        this.train = this.train.fillna({'Embarked': "S"}) # null값이 있다면 그 값을 S로 채우기. 탑승자들 중에 S가 매우 많고 노동자들이 많아서 승선정보체크 제대로 없이 누락된 값이 많다고 함
        this.test = this.test.fillna({'Embarked': "S"})
        #null값을 무엇으로 채울지는 사람이 판단
        this.train['Embarked'] = this.train['Embarked'].map({"S" : 1, "C" : 2, "Q" : 3}) # 컴퓨터가 이해하는 숫자로 명목변수값 변경
        this.test['Embarked'] = this.test['Embarked'].map({"S" : 1, "C" : 2, "Q" : 3}) # 컴퓨터가 이해하는 숫자로 명목변수값 변경
        return this

    @staticmethod
    def fare_ordinal(this) -> object:
        this.train['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4}) # 모든 문자열을 전부 숫자화(int) 시키는 것. 1/4등분으로 나눠줌.ordinal이므로 4가 제일 높은 금액
        this.test['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4}) # 모든 문자열을 전부 숫자화(int) 시키는 것. 1/4등분으로 나눠줌
        return this

    @staticmethod
    def fareBand_ordinal(this) -> object:
        this.train = this.train.fillna({'FareBand' : 1}) # null값은 가장 저렴한 금액대로 치환
        this.test = this.test.fillna({'FareBand' : 1})
        return this


    @staticmethod #self를 쓰지않고 this를 사용
    def title_norminal(this) -> object:
        combine = [this.train, this.test] # Royal 등의 칭호를 이름과 결합
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
            # 알파벳 처음부터 끝까지 1글자 이상으로 . 앞에 있는 영어단어 추출해서 Title이라는 새로운 컬럼에 담기
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Master') # 여러가지의 Title 값을 Rare로 변경
            dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal') # 여러가지의 Title 값을 Royal로 변경
            dataset['Title'] = dataset['Title'].replace('Mile','Mr') # Mile 을 Miss로 변경
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss') # Mile 을 Miss로 변경
            dataset['Title'] = dataset['Title'].replace('Mme', 'Rare') # Mme 을 Mrs로 변경
        print(this.train[['Title', 'Survived']].groupby('Title', as_index=False).mean())
        """결과값
            Title  Survived
            0  Master  0.466667
            1    Miss  0.699454
            2    Mlle  1.000000
            3      Mr  0.156673
            4     Mrs  0.792000
            5    Rare  1.000000
            6   Royal  1.000000
        """
        title_mapping = {'Mr' : 1, 'Miss' : 2, 'Mrs' : 3, 'Master' :4, 'Royal' : 5, 'Rare' : 6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0) # 타이틀 미상인 사람은 0으로 채우기. 전처리 담당자가 판단.
        print(this.train[['Title', 'Survived']].groupby('Title', as_index=False).mean())
        """ 
        결과값
           Title  Survived
            0    0.0  1.000000
            1    1.0  0.156673
            2    2.0  0.699454
            3    3.0  0.792000
            4    4.0  0.466667
            5    5.0  1.000000
            6    6.0  1.000000
        """
        this.train = this.train
        this.test = this.test
        return this # 새로 맵핑된 데이터를 덮어씀

    @staticmethod
    def sex_norminal(this) -> object:
        combine = [this.train, this.test]
        sex_mapping = {'male' : 0, 'female' : 1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
            dataset['Sex'] = dataset['Sex'].fillna(0)
        print(this.train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean())
        """ 결과값
           Sex  Survived
            0    0  0.188908
            1    1  0.742038
        """
        this.train = this.train
        this.test = this.test
        return this


    @staticmethod
    def age_ordinal(this) -> object:
        train = this.train
        test = this.test
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = train['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] #전처리 담당자가 구분연령대 임의로 설정. -1 ~0 까지의 구간, 0~5까지의 구간, ....  으로 구분한것
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels) #컷팅하면 Age에서 새로운 나이가 들어감
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels) #컷팅하면 Age에서 새로운 나이가 들어감
        age_title_mapping = {
            0: 'Unknown', 1: 'Baby', 2: 'Child', 3: 'Teenager', 4: 'Student',  5: 'Young Adult', 6: 'Adult', 7: 'Senior'
        }
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]
        age_mapping = {
            'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        this.train = train
        this.test = test
        return this

    # 머신러닝
    @staticmethod
    def create_k_fold():
        #k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        # 내가 옵션값을 주면 KFold() 생성자가 알아서 생성. 정교해지는건 좋지만 PC성능에 따라 소요시간이 오래걸림. 적절한 폴드 갯수를 개발자의 경험으로 지정해야함.
        #return k_fold #175, 177 행을 178행의 한줄로 합치고 174행의 self를 삭제하고. self를 사용하지않으므로 173행에 @staticmethod를 추가
        return KFold(n_splits=10, shuffle=True, random_state=0)

    @staticmethod
    def create_random_variable(train, X_features, Y_features) -> []: #DB에서는 스키마, ??에서는 컬럼, 기계학습에서 보는건 피쳐. 어느관점에서 보느냐에 따라 다름.
        # X_features : name 등 다양한 feature들 중에서 어떤것이 들어갈 확률. random으로 뽑으니 몇개인지는 모르겠지만 배열로 그 결과가 나올 것.
        the_X_features = X_features
        the_Y_feautres = Y_features
        train2, test2 = train_test_split(train, test_size=0.3, random_state=0) #random_state=0 으로 랜덤값 주지않고 있는그대로 진행
        train_X = train2[the_X_features]
        train_Y = train2[the_Y_feautres]
        test_X = test2[the_X_features]
        test_Y = test2[the_Y_feautres]
        return [train_X, train_Y, test_X, test_Y] # 181행-190행은 지도학습의 전형적인 코드. 비지도학습은 test가 없음.

    def accuracy_by_dtree(self, this):
        return round(np.mean(cross_val_score(DecisionTreeClassifier(),
                                             this.train,
                                             this.label,
                                             cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                             n_jobs=1,
                                             scoring='accuracy')) * 100, 2)


    def accuracy_by_rforest(self, this):
        return round(np.mean(cross_val_score(RandomForestClassifier(),
                                             this.train,
                                             this.label,
                                             cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                             n_jobs=1,
                                             scoring='accuracy')) * 100, 2)

    def accuracy_by_nb(self, this):
        return round(np.mean(cross_val_score(GaussianNB(),
                                         this.train,
                                         this.label,
                                         cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                         n_jobs=1,
                                         scoring='accuracy')) * 100, 2)

    def accuracy_by_knn(self, this):
        return round(np.mean(cross_val_score(KNeighborsClassifier(),
                                this.train,
                                this.label,
                                cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                n_jobs=1,
                                scoring='accuracy'))*100, 2)

    def accuracy_by_svm(self, this):
        return round(np.mean(cross_val_score(SVC(),
                                this.train,
                                this.label,
                                cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                n_jobs=1,
                                scoring='accuracy'))*100, 2)



    """
    3교시 간소화하기전. 2교시에서 완성한 코드. 메모리를 밟아야하므로 퍼포먼스 떨어짐. 
    def accuracy_by_dtree(self, model, dummy):  
    # model, dummy -> dummy는 이미 label이 init에서 정해져있으므로 label로 기재해야하고 this는 label과 train을 포함하도록 되어있으므로 this로 대체 
        #clf = DecisionTreeClassifier()
        #scoring = 'accuracy'  #결정트리 클래스에 값을 넣으면 scoring으로 정확도 점수 출력
        #k_fold = self.create_k_fold() #178행에서 10개로 나눈 폴드가 k_fold에 들어감
        score = cross_val_score(DecisionTreeClassifier(),
                                this.train,
                                this.label,
                                cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                n_jobs=1,
                                scoring='accuracy')
        # 인공지능끼리 분류기를 사용하여 서로 교차로 훈련시킴. cross_val_score는 사이킷런이 가지고 있는 기능. n_jobs 는 너무 크면 시간이 오래걸리므로 시간상 1로 지정함
        return round(np.mean(score)*100, 2) # 두자리수(00%)로 나타나도록 지정
    
    
    def accuracy_by_rforest(self, this):
        clf = RandomForestClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score)*100, 2)
        return accuracy

    def accuracy_by_nb(self, this):
        clf = GaussianNB()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score)*100, 2)
        return accuracy

    def accuracy_by_knn(self, model, label):
        clf = KNeighborsClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score)*100, 2)
        return accuracy

    def accuracy_by_svm(self, model, label):
        clf = SVC() # 클래스 이름을 가져와서 이 분류기를 사용하겠다고 지정
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score)*100, 2)
        return accuracy

    """