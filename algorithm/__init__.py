from algorithm.dtree import DTree

if __name__ == '__main__':
    def print_menu():
        print('0.Exit')
        print('1.결정트리')
        return input('메뉴 입력 \n')

    while 1:
        menu = print_menu()
        if menu == '0':
            break
        if menu == '1':
            DTree().breast_cancer()
            DTree().iris()
            #print('유방암 분류기 정확도: %d' % DTree().breast_cancer()


        """
        if menu == '2':
            app.modeling()  # str값을 모두 숫자화
        if menu == '3':
            pass
        if menu == '4':
            pass
        if menu == '5':
            t = FoliumTest()
            t.show_map()
        """


