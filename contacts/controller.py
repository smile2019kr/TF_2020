from contacts.service import Service
from contacts.model import Model

class Controller:
    def __init__(self):
        self.service = Service() # 세 객체에서 공유하는 서비스가 됨
    # init에서 표면에 나타나는 세가지 기능

    def register(self, name, phone, email, addr):
        model = Model()
        model.name = name
        model.phone = phone
        model.email = email
        model.addr = addr
        self.service.add_contact(model) # 객체가 어디있느냐에 따라 model로 주고 payload로 주는게 달라짐. payload는 데이터를 전달할때에만 (역할차이의 느낌)

    def list(self):
        return self.service.get_contacts()

    def remove(self, name):
        self.service.del_contact(name)


