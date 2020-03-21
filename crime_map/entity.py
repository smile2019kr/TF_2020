import pandas as pd
import json
import googlemaps

class Entity:
    def __init__(self):
        self._context = None
        self._fname = None

    @property
    def context(self) -> str:
        return self.context

    @context.setter
    def context(self, context):
        self._context = context


    @property
    def fname(self) -> str:
        return self.fname

    @fname.setter
    def fname(self, fname):
        self._fname = fname


    def new_file(self): #data폴더의 파일들이 많으므로 폴더명+파일명으로 새로운 파일명 지정
        return self._context + self._fname

    def csv_to_dframe(self) -> object: # csv파일을 df로 전환하여 객체로 지정
        return pd.read_csv(self.new_file(), encoding='UTF-8', thousands=',', ) # df로 만들어질 객체이름을 맨앞에 두고, 그 뒤에 구체적인 옵션 지정, 천단위마다 쉼표표기

    def xls_to_dframe(self, header, usecols) -> object:
        return pd.read_excel(self.new_file(), encoding='UTF-8', header=header,
                             usecols=usecols) #피쳐가 많으므로 정리해서 분석속도를 향상시킴.

    def json_load(self) -> object:
        return json.load(open(self.new_file(), encoding='UTF-8'))

    def create_gmaps(self) -> object:
        return googlemaps.Client(key='AIzaSyAhXaPdAteb5wRPiR0qiccupkolviVJog0') #구글에서 받은 API 코드를 '' 안에 넣을 예정
        # 선생님API키로 넣었을때는 잘 돌아갔으나 내 API키로 변경하니 에러남. 미승인 API키 라는 메세지

