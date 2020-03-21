from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
import requests # 자체적으로 내장으로 가지고 있는 것

class Service:
    def __init__(self):
        pass

    def bugs_music(self, payload): #url을 controller에서 모델에 담아놓은 값을 전달. payload
        soup = BeautifulSoup(urlopen(payload.url), payload.parser) #생성자. 괄호처리.
        n_artist = 0
        n_title = 0
        for i in soup.find_all(name='p', attrs=({'class' : 'title'})): # 곡 - 노래제목, 차트 등 의 1차원값으로 객체값 구성. i 는 노래 1곡이 됨. soup은 모든 값을 가져옴
                # {'class' : 'artist'} dictionary구조. 쌍을 이루고 있으므로 어떤 정보를 긁어오는데 활용
            n_title += 1 # 1씩 계속 증가
            print(str(n_title) + '위') # 순위출력
            print('노래제목: '+i.text) # i번째에 있는 텍스트를 가져와서 출력
        for i in soup.find_all(name='p', attrs=({'class' : 'artist'})):
            n_artist += 1 # 1씩 계속 증가
            print(str(n_artist) + '위') # 순위출력
            print('아티스트: '+i.find('a').text) # i번째 a에 있는 텍스트를 가져와서 출력


    def naver_movie(self, payload):
        driver = webdriver.Chrome(payload.path)
        driver.get(payload.url)
        soup = BeautifulSoup(urlopen(payload.url), payload.parser)
        # print(soup.prettify()) #prettify 를 사용하면 html스타일을 모두 가져오는 것
        all_divs = soup.find_all('div', attrs={'class' : 'tit3'}) # div가 들어간 것 중에서 클래스가 tit3인 값을 모두 긁어오기
        arr=[div.a.string for div in all_divs] #결과값을 리스트안에 담는 것
        for i in arr :
            print(i)
        driver.close

    def wiki(self, payload): # 테이블 구조를 크롤링하는 연습.
        soup = BeautifulSoup(requests.get(payload.url).text, payload.parser) # 생성자. requests의 get방식으로 text(글자)를 가져오기
        table = soup.find('table', {'class' : 'wikitable sortable'}) # 필요한 정보만 긁어오기
        result = [] # result라는 리스트에 결과값을 담기. 아래의 for구문에 따라서.
        for i in table.find_all('tr'):
            infolist = []
            info = '' # for 안에서의 info 값이 누적되고 빠져나올수 있도록 초기값 설정
            for j in i.find_all('td'):
                info = j.get_text()
                infolist.append(info)
            result.append(info)
        print(result)


    def hanbit(self, payload):
        USER = '1026516'
        PASS = '!!ncmh135!!'
        session = requests.session() # 자동로그인 기능 구현
        login_info = {
            'm_id' : USER,
            'm_passwd' : PASS
        } # 해당 홈페이지에서 설정되어있는 값
        res = session.post(payload.url, data=login_info)
        res.raise_for_status()
        mypage = 'http://www.hanbit.co.kr/myhanbit/myhanbit.html'
        res = session.get(mypage)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, payload.parser)
        mileage = soup.select_one('.mileage_section1 span').get_text()
        ecoin = soup.select_one('.mileage_section2 span').get_text()
        print('마일리지: ' + mileage)
        print('이코인: ' + ecoin)