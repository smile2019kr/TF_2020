from webcrawling.service import Service
from webcrawling.model import Model



class Controller:
    def __init__(self):
        self.service = Service() #service와 model의 객체 생성.
        self.model = Model()

    def bugs_music(self, url):
        self.model.url = url #init에서 받은 값이 모델로 주입되고 모델에 입력된 값이 다다음행에서 서비스로 넘어감
        self.model.parser = 'lxml' # 외부에서 주입받는게 아니라 내부에서 개발자가 적절하게 집어넣는 값. 외부에서는 url 만 받음.
        self.service.bugs_music(self.model)

    def naver_movie(self, url):
        self.model.url = url
        self.model.parser = 'html.parser'
        self.model.path = './data/chromedriver.exe'
        self.service.naver_movie(self.model)

    def wiki(self, url):
        self.model.url = url
        self.model.parser = 'html.parser' #html구문이면 html parser가 대부분 가능. 어떤 파서가 좋은지 메시지가 뜨는경우도 있음
        self.service.wiki(self.model)

    def hanbit(self, url):
        self.model.url = url
        self.model.parser = 'html.parser'
        self.service.hanbit(self.model)



    def nex(self, url):
        self.model.url = url
        self.model.parser = 'html.parser'
        self.service.nex(self.model)
