class Service:
    def __init__(self):
        self._contacts = [] #여기서 self.는 service를 말함
    # 기능정의. 5행 이하가 알고리즘.
    def add_contact(self, payload): #주소록에 한명씩 추가하기. 값을 전달하는 것(보통, 모델)을 payload로 설정.
        self._contacts.append(payload) # payload에서 들어온 객체 하나를 기존의 contacts에 append하라는 것

    # 모델이 서비스 안에 있는 것. 차원의 문제. matrix 구조. tensor에게 collection을 넣었다가 뺐다가 함.
    def get_contacts(self) ->str: #주소록의 연락처들을 가져오기
        contacts = [] # ._contacts : 파이썬 안에 있는것.  to_string 을 만든 이유는 컴퓨터에 주소값(0/1 바이너리값)으로 저장된 것을 string으로 보여주기위함
        for i in self._contacts: # i -> payload. 전달되는 값.
            contacts.append(i.to_string()) # i 를 그대로 하면 요소가 나옴. 사람이 이해할 수 있는 문자로 출력하라는 것. to_string() 괄호 잊지말기
        return ' '.join(contacts)

    def del_contact(self, name):
        for i, t in enumerate(self._contacts):
            if t.name == name :
                del self._contacts[i] # 컴퓨터에 저장되어있는 i 번째를 삭제


