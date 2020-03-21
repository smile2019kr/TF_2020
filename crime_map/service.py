from dataclasses import dataclass
import folium

from crime_map.entity import Entity
import pandas as pd
import numpy as np
from sklearn import preprocessing

class Service:
    def __init__(self):
        self.e = Entity()
        self.crime_rate_columns = ['살인 검거율', '강간 검거율', '강도 검거율', '절도 검거율', '폭력 검거율']
        self.crime_columns = ['살인', '강간', '강도', '절도', '폭력']

    def save_police_pos(self): # 경찰서의 위치값을 저장하는 공간
        station_names = []
        e = self.e
        e.context = './data/'
        e.fname = 'crime_in_seoul.csv'  # 파일이름
        crime = e.csv_to_dframe()
        for name in crime['관서명']: #해당이름의 API추출
            station_names.append('서울'+str(name[:-1]+'경찰서'))
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = e.create_gmaps()
        for name in station_names: # 제대로 API키값을 인식하는지 확인
            t = gmaps.geocode(name, language='ko') #한국어로 언어설정을 해야 한글로 된 지역명 인식
            station_addrs.append(t[0].get('formatted_address'))
            t_loc = t[0].get('geometry') # 구글API에 있는 옵션값을 일시적으로 저장
            station_lats.append(t_loc['location']['lat']) #위도정보를 차곡차곡 쌓음
            station_lngs.append(t_loc['location']['lng']) #경도정보를 차곡차곡 쌓음
        #    print(name + ' ------> ' + t[0].get('formatted_address')) #API에 있는 정보를 출력
        gu_names = [] # 구 이름만 추출
        for name in station_addrs:
            print(f'--> {name}')
            t = name.split()
            gu_name = [gu for gu in t if gu[-1] == '구'][0]
            gu_names.append(gu_name)
        crime['구별'] = gu_names  # 구와 경찰서 위치가 다른경우 수작업
        crime.loc[crime['관서명'] == '혜화서', ['구별']] == '종로구'  # 구글맵에 '혜화서'인 곳이 구 표시가 잘못되어있었다는 것
        crime.loc[crime['관서명'] == '서부서', ['구별']] == '은평구'  # 구글맵에 '서부서'인 곳이 구 표시가 잘못 되어있었다는 것
        crime.loc[crime['관서명'] == '강서서', ['구별']] == '양천구'  # 구글맵에 '혜화서'인 곳이 구 표시가 잘못 되어있었다는 것
        crime.loc[crime['관서명'] == '종암서', ['구별']] == '성북구'  # 구글맵에 '혜화서'인 곳이 구 표시가 잘못 되어있었다는 것
        crime.loc[crime['관서명'] == '방배서', ['구별']] == '서초구'  # 구글맵에 '혜화서'인 곳이 구 표시가 잘못 되어있었다는 것
        crime.loc[crime['관서명'] == '수서서', ['구별']] == '강남구'  # 구글맵에 '혜화서'인 곳이 구 표시가 잘못 되어있었다는 것
        crime.to_csv('./saved_data/police.csv') # 2/22 추가

        """ 2/22. 삭제
        # print(crime)  # 제대로 들어왔는지 확인
        self.police_pos = crime
        # crime.to_csv('./saved_data/police.csv') #saved_data라는 폴더를 생성한 후 저장공간으로 지정하여 실행. 한번 실행하면 계속저장되지않도록 주석처리
        """
    def save_cctv_pop(self):
        e = self.e
        e.context = './data/'
        e.fname = 'cctv_in_seoul.csv'
        cctv = e.csv_to_dframe()
        e.fname = 'pop_in_seoul.xls'
        pop = e.xls_to_dframe(2, 'B,D,G,J,N')

        cctv.rename(columns={cctv.columns[0]: '구별'}, inplace=True)

        pop.rename(columns={
            pop.columns[0]: '구별',
            pop.columns[1]: '인구수',
            pop.columns[2]: '한국인',
            pop.columns[3]: '외국인',
            pop.columns[4]: '고령자'
        }, inplace=True)

        pop.drop([26], inplace = True) #26번 row에 null값있어서 삭제
        print(pop)
        pop['외국인비율'] = pop['외국인'].astype(int) / pop['인구수'].astype(int) * 100  # str로 인식하므로 int로 변환해서 계산
        pop['고령자비율'] = pop['고령자'].astype(int) / pop['인구수'].astype(int) * 100

        cctv.drop(['2013년도 이전', '2014년', '2015년'], 1, inplace=True) # inplace=True : 대체한 값으로 fix
        cctv_pop = pd.merge(cctv, pop, on='구별')
        cor1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop['소계'])
        cor2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop['소계'])
        print(f'고령자비율과 CCTV의 상관계수 {str(cor1)} \n'
              f'외국인비율과 CCTV의 상관계수 {str(cor2)}')

        """ 결과값
         고령자비율과 CCTV 의 상관계수 [[ 1.         -0.28078554]
                                     [-0.28078554  1.        ]] 
         외국인비율과 CCTV 의 상관계수 [[ 1.         -0.13607433]
                                     [-0.13607433  1.        ]]
        r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
        r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
        r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
        r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
        r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
        r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
        r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
        고령자비율 과 CCTV 상관계수 [[ 1.         -0.28078554] 약한 음적 선형관계
                                    [-0.28078554  1.        ]]
        외국인비율 과 CCTV 상관계수 [[ 1.         -0.13607433] 거의 무시될 수 있는
                                    [-0.13607433  1.        ]]                        
         """

        cctv_pop.to_csv('./saved_data/cctv_pop.csv')

    #def fetch_police(self, payload): # police.csv 로 저장된 파일을 DF로 읽어들이기 #2/22삭제
    #    self.police = payload.csv_to_dframe() #2/22삭제

    def save_police_norm(self):
        e = self.e
        e.context = './saved_data/'
        e.fname = 'police_pos.csv'
        police_pos = e.csv_to_dframe()
        police = pd.pivot_table(self.police, index='구별', aggfunc=np.sum) # 방향을 90도로 변경
        #print(police.columns)
        """결과값
        Index(['Unnamed: 0', '강간 검거', '강간 발생', '강도 검거', '강도 발생', '살인 검거', '살인 발생',
       '절도 검거', '절도 발생', '폭력 검거', '폭력 발생'], dtype='object')
        """
        police['살인 검거율'] = police['살인 검거'].astype(int) / police['살인 발생'].astype(int) * 100
        police['강간 검거율'] = police['강간 검거'].astype(int) / police['강간 발생'].astype(int) * 100
        police['강도 검거율'] = police['강도 검거'].astype(int) / police['강도 발생'].astype(int) * 100
        police['절도 검거율'] = police['절도 검거'].astype(int) / police['절도 발생'].astype(int) * 100
        police['폭력 검거율'] = police['폭력 검거'].astype(int) / police['폭력 발생'].astype(int) * 100

        police.drop(columns={'살인 검거', '강간 검거', '강간 검거', '강간 검거', '강간 검거'}, axis=1)  # 검거율 산출했으므로 불필요한 열 삭제

        for i in self.crime_rate_columns:
            police.loc[police[i] > 100, 1] = 100  # 데이터값의 기간 오류로 100을 넘으면 100으로 계산
        police.rename(columns={
            '살인 발생': '살인',
            '강간 발생': '강간',
            '강도 발생': '강도',
            '절도 발생': '절도',
            '폭력 발생': '폭력'
        }, inplace=True)

        x = police[self.crime_rate_columns].values
        min_max_scalar = preprocessing.MinMaxScaler()  # 자동으로 전처리. 0부터 1사이의 값으로 scaling.
        """
        스케일링은 선형변환을 적용하여 
        전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정
        """

        x_scaled = min_max_scalar.fit_transform(x.astype(float))
        """
        정규화 normalization
        많은 양의 데이터를 처리함에 있어 데이터의 범위(도메인)을 일치시키거나
        분포(스케일)을 유사하게 만드는 작업
        """

        police_norm = pd.DataFrame(x_scaled, columns=self.crime_columns, index=police.index)
        police_norm[self.crime_rate_columns] = police[self.crime_rate_columns] # police에서 범죄율에 관한 내용만 police_norm으로 추가

        """
        def save_police_norm(self, payload):
        police_norm = self.police_norm
        self.cctv_pop = pd.read_csv(payload.new_file(), encoding='UTF-8', sep=',', index_col='구별')
        """
        police_norm['검거'] = np.sum(police_norm[self.crime_rate_columns], axis=1)
        police_norm['범죄'] = np.sum(police_norm[self.crime_columns], axis=1)
        #police_norm = police_norm # 바뀐값을 self.police_norm 에 대입
        police_norm.to_csv('./saved_data/police_norm.csv', sep=',', encoding='UTF-8')


    def draw_crime_map(self):
        e = self.e
        e.context = './saved_data/'
        e.fname = 'police_norm.csv'
        police_norm = e.csv_to_dframe()
        e.context = './data/'
        e.fname = 'geo_simple.json'
        seoul_geo = e.json_load()
        e.fname = 'crime_in_seoul.csv'
        crime = e.csv_to_dframe()
        e.context = './saved_data/'
        e.fname = 'police_pos.csv'
        police_pos = e.csv_to_dframe()
        station_names = []
        for name in crime['관서명']: #해당이름의 API추출
            station_names.append('서울'+str(name[:-1]+'경찰서'))
        station_addrs = []
        station_lats = []
        station_lngs = []
        gmaps = e.create_gmaps()
        for name in station_names:
            t = gmaps.geocode(name, language='ko')
            station_addrs.append(t[0].get('formatted_address'))
            t_loc = t[0].get('geometry') # 구글API에 있는 옵션값을 일시적으로 저장
            station_lats.append(t_loc['location']['lat']) #위도정보를 차곡차곡 쌓음
            station_lngs.append(t_loc['location']['lng']) #경도정보를 차곡차곡 쌓음
        police_pos['lat'] = station_lats
        police_pos['lng'] = station_lngs
        col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
        tmp = police_pos[col] / police_pos[col].max()
        police_pos['검거'] = np.sum(tmp, axis=1)
        m = folium.Map(location=[37.5502, 126.982], zoom_start=12, title='Stamen Toner') # 서울의 위도, 경도값, zoom_start는 단위값지정
        m.choropleth(
            geo_data = seoul_geo,
            name = 'choropleth',
            data = tuple(zip(police_norm['구별'], police_norm['범죄'])),
            key_on = 'feature.id',
            fill_opacity = 0.7,
            fill_color = 'PuRd',
            line_opacity = 0.7,
            legend_name = 'Crime Rate (%)'
        )
        for i in police_pos.index:  #검거율을 원으로 추가표기
            folium.CircleMarker([police_pos['lat'][i], police_pos['lng'][i]],
                                radius=police_pos['검거'][i]*10,
                                fill_color = '#0a0a32').add_to(m)

        m.save('./saved_data/Crime_map.html')

