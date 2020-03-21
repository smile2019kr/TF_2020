from contacts.controller import Controller

if __name__ == '__main__': # 네개중에서 이 구문이 있는걸 가장 먼저 시작. 나머지 세개의 객체들이 역할을 수행하도록 호출함
    def print_menu():
        print('0.Exit')
        print('1.연락처 추가')
        print('2.연락처 목록')
        print('3.연락처 삭제')
        return input('메뉴 선택 \n')


    """ 
    주석표기도 클래스보다 하위에 있어야 함
        while 1:
            menu = print_menu()
            if menu == '0':
                break

            if menu == '1':
                app = Controller()
                print('연락처 추가정보 입력 \n')
                name = input('이름 : \n')
                phone = input('전화 : \n')
                email = input('이메일 : \n')
                addr = input('주소 : \n')

            if menu == '2':
                app = Controller()
                print('연락처 목록 표시\n')
                print(contacts)

            elif menu == '3':
                break
    """


    app = Controller() # calculator 처럼 while구문안으로 넣으면 매번 새로 없어짐
    while 1:
        menu = print_menu()

        if menu == '1':
            app.register(input('이름 \n'), input('전화번호 \n'), input('이메일 \n'), input('주소 \n'))
        if menu == '2':
            print(app.list()) # 아까 to_string으로 지정해주었으므로 그대로 나오도록 print구문안에 넣어주기
        if menu == '3':
            app.remove(input('삭제할 이름\n'))
        elif menu == '0': #break는 맨마지막에 elif와 함께 오는게 관습적
            break
