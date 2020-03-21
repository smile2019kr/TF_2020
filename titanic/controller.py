from titanic.entity import Entity # 속성
from titanic.service import Service  # 기능
from sklearn.svm import SVC
import pandas as pd
# 속성+기능=객체 이므로 Controller는 객체에 해당

class Controller:
    def __init__(self):
        self.entity = Entity()
        self.service = Service()

    def modeling(self, train, test):
        service = self.service
        this = self.preprocess(train, test)
        this.label = service.create_label(this)
        this.train = service.create_train(this)
        return this # this에는 라벨과 트레인이 들어가있는 상태이므로 def learning에서 print에는 this만 들어감

    def preprocess(self, train, test) -> object: #외부에서 가지고 오는 값이므로 train, test설정
        service = self.service
        this = self.entity
        this.train = service.new_model(train)
        this.test = service.new_model(test)
        this.id = this.test['PassengerId']
        #print(f'트레인 드랍 전 컬럼 : {this.train.columns}') #오류체크용
        this = service.drop_feature(this, 'Cabin')
        this = service.drop_feature(this, 'Ticket')
        #print(f'트레인 드랍 후 컬럼 : {this.train.columns}') #오류체크용

        this = service.embarked_norminal(this) #메소드가 순서대로 작동하도록 순서대로 나열
        this = service.title_norminal(this)
        this = service.drop_feature(this, 'Name') #이름은 생존과 상관없으므로 호칭을 추출하고 난 뒤의 이름은 데이터에서 삭제
        this = service.drop_feature(this, 'PassengerId') #랜덤으로 생성되는 ID는 생존과 상관없으므로 호칭을 추출하고 난 뒤의 이름은 데이터에서 삭제
        this = service.age_ordinal(this)
        this = service.sex_norminal(this)
        this = service.fareBand_ordinal(this)
        this = service.drop_feature(this, 'Fare')
        print(f'전처리 마감 후 컬럼: {this.train.columns}')
        print(f'train널의 수량: {this.train.isnull().sum()}')
        print(f'test널의 수량: {this.test.isnull().sum()}')
        return this

        """
        결과값
        train널의 수량: Survived    0
        Pclass      0
        Sex         0
        Age         0
        SibSp       0
        Parch       0
        Embarked    0
        Title       0
        AgeGroup    0
        dtype: int64
        test널의 수량: Pclass      0
        Sex         0
        Age         0
        SibSp       0
        Parch       0
        Embarked    0
        Title       0
        AgeGroup    0
        dtype: int64
        """


    def learning(self, train, test):
        service = self.service
        this = self.modeling(train, test) #self:전역. global. 전반적으로. this: 지역. 이 클래스안에서만.
        #label = service.create_label()
        print(f'결정트리 활용한 검증 정확도 {service.accuracy_by_dtree(this)}')
        print(f'랜덤포레스트 활용한 검증 정확도 {service.accuracy_by_rforest(this)}')
        print(f'나이브베이즈 활용한 검증 정확도 {service.accuracy_by_nb(this)}')
        print(f'KNN 활용한 검증 정확도 {service.accuracy_by_knn(this)}')
        print(f'SVM 활용한 검증 정확도 {service.accuracy_by_svm(this)}')

        """
        출력결과
        결정트리 활용한 검증 정확도 78.57
        랜덤포레스트 활용한 검증 정확도 80.7
        나이브베이즈 활용한 검증 정확도 81.03
        KNN 활용한 검증 정확도 78.79
        SVM 활용한 검증 정확도 69.59
        """

    def submit(self, train, test): #캐글에 테스트결과를 제출
        # service = self.service
        this = self.modeling(train, test)
        clf = SVC() # SVC의 정확도가 가장 높게나와서 이 분류결과를 submit
        clf.fit(this.train, this.label) # 훈련은 this.train, 답은 this.label
        prediction = clf.predict(this.test)
        submission = pd.DataFrame(
            {'PassengerId': this.id, 'Survived': prediction}
        ).to_csv('./data/submission.csv', index_label=False) #index_label을 지우라는 것. 스키마 삭제.


