import nltk
from textmining.entity import Entity
from textmining.samsung_service import SamsungService

class Controller:
    def __init__(self):
        pass

    def download_dictionary(self):
        nltk.download('all') #영어사전 모두 다운로드. 로직이 아니므로 서비스까지 갈건 없음

    #def data_analysis(self, fname): #외부에서 들어온것, 이름은 fname으로 지정. stopwords 코딩 전.
    def data_analysis(self):  # stopwords도 있으므로 각각 이름으로 상수처리
        entity = Entity() #데이터저장소
        service = SamsungService() # entity, service: 인스턴스.
        entity.fname = 'Kr-report_2018.txt'
        entity.context = './data/' #데이터가 있는 장소
        service.extract_token(entity) # controller에서는 samsung_service에서 지정한 순서대로 내려올 것.
        service.extract_hangeul()
        service.conversion_token()
        service.compound_noun()
        entity.fname = 'stopwords.txt' # 15행에서의 파일명이 대체됨
        service.extract_stopword(entity)
        service.filtering_text_with_stopword()
        service.frequent_text()
        entity.fname = 'D2Coding.ttf' #분석순서에 따라서 순차적으로 파일명이 대체됨
        service.draw_wordcloud(entity)
        """
        구글링: 디자인 패턴 훅 메서드. 작게 모듈화된 알고리즘을 순서대로 배열
        data_analysis(self, fname) 은 작게 분할된 훅 메서드로 조합되어있다는 것
        조합된 틀을 가지고 있으면 데이터만 넣으면 학습 진행
        """


