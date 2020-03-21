#init은 controller만 가지고 옴
from calculator.controller import Controller


if __name__ == '__main__':
    def print_menu():
        print('0.Exit')
        print('1.Calculator')
        print('2.TensorCalculator')
        return input('메뉴선택 \n')

    while 1: #true일때
        menu = print_menu()
        if menu == '0':
            break
        if menu == '1':
            app = Controller()  # 1을 누르면 계산기를 실행하라는 기능으로 Controller를 실행
            print('계산기 작동')
            num1 = int(input('첫번째 수 \n')) #input의 return값은 string이므로 int()로 지정해줘야함
            opcode = input('연산자 \n')
            num2 = int(input('두번째 수 \n'))
            # result = num1 + num2  15행을 넣기 전
            result = app.exec(num1, num2, opcode) # app으로 객체를 설정
            print('결과: %d ' % result) # % -> 파싱하는 코드
        if menu == '2':
            app = Controller()  # 1을 누르면 계산기를 실행하라는 기능으로 Controller를 실행
            print('텐서계산기 작동')
            num1 = int(input('첫번째 수 \n')) #input의 return값은 string이므로 int()로 지정해줘야함
            opcode = input('연산자 \n')
            num2 = int(input('두번째 수 \n'))
            # result = num1 + num2  15행을 넣기 전
            result = app.tensorExec(num1, num2, opcode) # app으로 객체를 설정
            print('결과: %d ' % result) # % -> 파싱하는 코드
