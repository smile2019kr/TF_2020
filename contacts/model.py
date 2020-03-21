class Model:
    def __init__(self):
        self.name = '' # property 4종으로 선정하고 각각 초기값 부여
        self.phone = ''
        self.email = ''
        self.addr = ''

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def phone(self) -> str:
        return self._phone

    @phone.setter
    def phone(self, phone):
        self._phone = phone

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, email):
        self._email = email

    @property
    def addr(self) -> str:
        return self._addr

    @addr.setter
    def addr(self, addr):
        self._addr = addr

    def to_string(self): #4개의 속성값을 일괄적으로 출력하는 메소드 작성 (일괄출력에 편함)
        return '이름: {}, 전화번호: {}, 이메일: {}, 주소: {}'.format(self._name, self._phone, self._email, self._addr)
        # ''.format() 스트링값이 여러개일때 앞을 지정해주고 format(속성값)사용

