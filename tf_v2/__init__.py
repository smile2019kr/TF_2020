from tf_v2.fashion_mnist import FashionMnist
from tf_v2.imdb import Imdb

if __name__ == '__main__':
    def print_menu():
        print('0.Exit')
        print('1.Fashion Mnist')
        print('2.Imdb ') # 영화
        print('3.')
        return input('메뉴 입력 \n')


    while 1:
        menu = print_menu()
        if menu == '1':
            fm = FashionMnist()
            #  fm.show_dataset()
            fm.create_model()
            #  fm.predict_image()
            #  fm.subplot_test(fm.predict_image()) # 윗행에서는 앵클부츠 하나만 맞춤. 이렇게 하면 안쪽에서 여러번 돌릴수 있게 하고 시간단축가능
            print(fm.one_test(100)) # 100번째 이미지번호를 맞췄는지 출력
        if menu == '2':
            imdb = Imdb()
            imdb.download_data()
            #print(imdb.create_sample())
        if menu == '3':
            s1 = SaveLoad()
            s1.execute()
        elif menu == '0':
            break
