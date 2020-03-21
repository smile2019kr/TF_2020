# controller에서는 Model, Service 클래스(첫글자가 대문자) 를 각각 가져옴


from calculator.service import Service
from calculator.model import Model
from calculator.tensor_service import TensorService


class Controller:
    def exec(self, num1, num2, opcode):
        model = Model() # 함수처럼 괄호를 넣으면 인스턴스가 되어서 클래스의 기능을 가지는 객체가 됨
        model.num1 = num1
        model.num2 = num2
        model.opcode = opcode  #10-12행 : 데이터 주입
        # 7행 -> 10행 opcode 가 model.opcode로 들어가서 model.py의 setter에 있는 opcode로 꽂힘
        service = Service(model) # model의 값을 service에 전달해야 연산 가능
        if opcode == '+':
            result = service.add()
        if opcode == '-':
            result = service.subtract()
        if opcode == '*':
            result = service.multiply()
        if opcode == '/':
            result = service.divide()
        return result

    def tensorExec(self, num1, num2, opcode):
        model = Model()  # 함수처럼 괄호를 넣으면 인스턴스가 되어서 클래스의 기능을 가지는 객체가 됨
        model.num1 = num1
        model.num2 = num2
        model.opcode = opcode  # 10-12행 : 데이터 주입
        service = TensorService(model)
        if opcode == '+':
            result = service.add()
        if opcode == '-':
            result = service.subtract()
        if opcode == '*':
            result = service.multiply()
        if opcode == '/':
            result = service.divide()
        return result
