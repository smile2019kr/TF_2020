from webcrawling.controller import Controller

if __name__ == '__main__':  # 네개중에서 이 구문이 있는걸 가장 먼저 시작. 나머지 세개의 객체들이 역할을 수행하도록 호출함
    def print_menu():
        print('0.Exit')
        print('1.Bugs Music ')
        print('2.Naver movie ')
        print('3.Wiki ')
        print('4.Hanbit ')
        print('5.Weather ')
        print('10.nex')
        return input('메뉴 선택 \n')


    app = Controller()  # calculator 처럼 while 구문안으로 넣으면 매번 새로 없어짐
    while 1:
        menu = print_menu()
        if menu == '1':
            app.bugs_music('https://music.bugs.co.kr/chart/track/realtime/total?chartdate=20200201&charthour=16')
        if menu == '2':
            app.naver_movie('https://movie.naver.com/movie/sdb/rank/rmovie.nhn')
        if menu == '3':
            app.wiki('http://dh.aks.ac.kr/Encyves/wiki/index.php/%EC%A1%B0%EC%84%A0_%EC%84%B8%EC%A2%85')
        if menu == '4':
            app.hanbit('http://www.hanbit.co.kr/member/login_proc.php') # 바로 로그인까지 연결되는 화면주소
        if menu == '5':
            app.weather('    ')
        if menu == '10':
            app.nex('   ')
        elif menu == '0':  # break는 맨마지막에 elif와 함께 오는게 관습적
            break
