from textmining.controller import Controller #service는 분기하고 controller는 하나만 잡기로 함

if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. 사전 다운로드') #Kor, Eng 사전 다운로드해서 단어를 알려줘야 함. 다운로드 작업은 controller에서
        print('2. 삼성 전략보고서 분석')
        return input('Select Menu\n')
    app = Controller()
    while 1:
        menu = print_menu()
        if menu == '1':
            app.download_dictionary() # controller에서 생성한 download_dictionary()를 실행. 절차지향적으로 여기에 다 쓰지말고 객체지향적으로 구조 설정
        if menu == '2':
           #app.data_analysis('kr-Report_2018.txt') # 파일로 제공한 데이터를 분석 stopwords 추출 전
            app.data_analysis() # 파일로 제공한 데이터를 분석 stopwords 추출 전
        elif menu == '0':
            break

