import pandas as pd
import folium

class FoliumTest:
    def __init__(self):
        pass

    def show_map(self):
        state_geo = './data/us-states.json'
        state_unemployment = './data/us_unemployment.csv'
        state_data = pd.read_csv(state_unemployment)
        m = folium.Map(location=[37, -102], zoom_start=5) # 미국의 위도, 경도값, zoom_start는 단위값지정
        m.choropleth(
            geo_data = state_geo,
            name = 'choropleth',
            data = state_data,
            columns = ['State', 'Unemployment'],
            key_on = 'feature.id',
            fill_opacity = 0.7,
            fill_color = 'YlGn',
            line_opacity = 0.7,
            legend_name = 'Unemployment Rate (%)'
        )
        folium.LayerControl().add_to(m)
        m.save('./saved_data/USA.html')

