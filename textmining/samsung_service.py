import re # 정규표현식
from nltk import word_tokenize
from konlpy.tag import Okt
import pandas as pd
from nltk import FreqDist
        #언어 상관없이 빈출단어 추출
from wordcloud import WordCloud
        # 모듈 가져올때는 from 클래스 가져올때는 import
import matplotlib.pyplot as plt


class SamsungService:
    def __init__(self):
        self.texts = [] # 어미 등이 모두 붙어있는 상태로 저장하는 공간
        self.tokens = [] # 명사만 저장하는 공간
        self.okt = Okt() # 편하게 활용할수 있게 인스턴스로 등록
        self.stopwords = []
        self.freqtxt = []

        """
        매번 코딩할 필요없도록 모델을 작은 모듈로 세분화해서 조합
        공통된 기능 있는 메서드는 매번 코딩하지 말고 한번 만들어서 여기저기서 재조합할수 있도록 구조를 만들어두는 것
        """

    def extract_token(self, payload): #payload : 다른 파일에 있는 데이터값을 불러오는 전달자. entity의 값이 들어올 것
        print('>>> text 문서에서 token 추출')
        filename = payload.context + payload.fname # 경로와 파일명을 합친것을 filename으로 설정
        with open(filename, 'r', encoding='utf-8') as f: #filename을 utf-8로 인코딩해서 f라는 객체로 지정
            self.texts = f.read() # 추출된 단어들을 아래의 메서드에서 모두 공유하기 위해서 3행에 토큰들이 모여있는 장소를 지정
        #print(f'{self.texts[:300]}') # 오류확인용. f'{}' : {}안의 단어를 str으로 포맷팅하라는 것. print('{}'.format('self.texts')의 문법이 변형된 것.

    def extract_hangeul(self):
        print('>>> 한글만 추출')
        texts = self.texts.replace('\n', ' ') # 줄바꿈을 제거
        tokenizer = re.compile(r'[^ ㄱ-힣]') # 정규표현식. 한글의 첫 글자와 맨마지막 글자
        self.texts = tokenizer.sub('', texts) # 한글빼고 다 제거한 값을 위에서 공유한 값에 저장
        #print(f'{self.texts[:300]}') # 오류확인용

    def conversion_token(self):
        print('>>> 토큰으로 변환')
        self.tokens = word_tokenize(self.texts)  #pc가 이해할 수 있는 토큰으로 변환
        #print(f'{self.texts[:300]}')  # 오류확인용

    def compound_noun(self):
        print('>>> 복합명사는 묶어서 filtering 으로 출력')
        print('>>> ex) 삼성전자의 스마트폰은 ---> 삼성전자 스마트폰') # 어미, 조사 삭제
        noun_token = [] # 토큰중에서 명사만 뽑아냄
        for token in self.tokens:
            token_pos = self.okt.pos(token)
            temp = [txt_tag[0] for txt_tag in token_pos
                    if txt_tag[1] == 'Noun'] # 텍스트마다 명사, 동사 등으로 tagging 되어있는데 그중에서 명사만 추출
            if len("".join(temp)) > 1: # 단어가 존재하면. (단어길이가 1보다 길면)
                noun_token.append("".join(temp)) # 한글자는 제거. 두글자 이상만 집어넣겠다는 것.
        self.texts = " ".join(noun_token)
        #print(f'{self.texts[:300]}')  # 오류확인용
        #결과 : 삼성전자 가능보고서 보고서 개요 전자 사회환경 가치창 통합 성과 이해관계자 소통 매년 가능보고서 발간 열한 가능보고서 발간 보고기간 보고서 사회환경 성과 활동 일부 정성 성과 대해 자료 포함 정량 연도별 추이 분석 최근 개년 수치 제공 보고범위 보고범위 국내 해외 사업 공급망 포함 재무성 연결기준 작성 사업 환경 정량 국내외 생산 법인 수집 데이터 기준 작성 작성기준 핵심 부합 방법 작성 추가정보 삼성전자 대표 홈페이지 지속가능경영 홈페이지 홈페이지 삼성전자 뉴스룸 작성자 삼성전자 가능사무국 주소 경기도 수원시 영통구 삼성로 이메일

    def extract_stopword(self, payload):
        print('>>> stopwords에서 단어 추출')
        filename = payload.context + payload.fname
        with open(filename, 'r', encoding='utf-8') as f:
            self.stopwords = f.read() # stopwords를 texts와는 다른 공간에 넣기
        self.stopwords = self.stopwords.split(' ')
        #print(f'{self.stopwords[:10]}')

    def filtering_text_with_stopword(self): # 나중에 재조합 할 수 있도록 각 메소드 이름을 명확하게 명칭 부여하는게 좋음
        print('>>> stopwords 필터링')
        self.texts = word_tokenize(self.texts)
        self.texts = [text for text in self.texts
                      if text not in self.stopwords]

    def frequent_text(self):
        print('>>> 빈도수로 정렬')
        self.freqtxt = pd.Series(dict(FreqDist(self.texts)))\
            .sort_values(ascending=False) #빈출빈도 높은 단어부터 순서대로 추출
        print(f'{self.freqtxt[:10]}')

        #워드클라우드에서 표현하기 위한 폰트 주입 D2Conding.ttf 파일을 data폴더에 삽입
    def draw_wordcloud(self, payload): #워드클라우드 폰트를 가져오기위해 payload 설정
        print('>>> 워드클라우드 작성')
        filename = payload.context + payload.fname
        wcloud = WordCloud(filename, relative_scaling=0.2, background_color='white').generate(" ".join(self.texts))
        plt.figure(figsize=(12,12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
