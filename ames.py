import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
import kagglehub
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
# 1. 데이터 불러오기
df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")
df = df[['Latitude', 'Longitude']].dropna()

# 2. 지도 중심
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, width='100%', height='600px')

# 3. 마커 추가
marker_cluster = MarkerCluster().add_to(m)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(marker_cluster)

# ✅ 4. HTML로 저장 (폴더는 qmd 기준 상대경로로)
# m.save("./data/house/house_gla.html")
m


# cond_qual_vars = [
#     'KitchenQual', 'FireplaceQu', 'BsmtCond', 'ExterCond', 'GarageCond',
#     'OverallCond', 'BsmtQual', 'ExterQual', 'GarageQual', 'HeatingQC' 'GrLivArea'
# ]
df['GrLivArea'].unique()

# 지역별 최빈 지하실 상태 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 지하실 상태 점수 매핑 (나쁠수록 높은 점수)
bsmt_cond_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['BsmtCondScore'] = df['BsmtCond'].map(bsmt_cond_mapping)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 최빈 지하실 상태 점수 계산
mode_score = (
    df.dropna(subset=['BsmtCondScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['BsmtCondScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
bsmt_color_map = {
    1: 'blue',       # Excellent
    2: 'green',      # Good
    3: 'gold',       # Typical
    4: 'orangered',  # Fair
    5: 'darkred'     # Poor
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='BsmtCondScore',
    category_orders={"BsmtCondScore": sorted(bsmt_color_map.keys())},
    color_discrete_map={str(k): v for k, v in bsmt_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "지하실 상태 점수 (BsmtCondScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})


# ExterCond 시각화
# 지역별 최빈 지하실 상태 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 지하실 상태 점수 매핑 (나쁠수록 높은 점수)
bsmt_cond_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['ExterCondScore'] = df['ExterCond'].map(bsmt_cond_mapping)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 최빈 지하실 상태 점수 계산
mode_score = (
    df.dropna(subset=['ExterCondScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['ExterCondScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
Exter_color_map = {
    1: 'blue',       # Excellent
    2: 'green',      # Good
    3: 'gold',       # Typical
    4: 'orangered',  # Fair
    5: 'darkred'     # Poor
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='ExterCondScore',
    category_orders={"ExterCondScore": sorted(Exter_color_map.keys())},
    color_discrete_map={str(k): v for k, v in Exter_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "외장재 상태 점수 (ExterCondScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})


# GarageCond 시각화
# 지역별 최빈 차고지 상태 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 차고지 상태 점수 매핑 (나쁠수록 높은 점수)
Garage_cond_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['GarageCondScore'] = df['GarageCond'].map(bsmt_cond_mapping)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 최빈 차고지 상태 점수 계산
mode_score = (
    df.dropna(subset=['GarageCondScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['GarageCondScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
Garage_color_map = {
    1: 'blue',       # Excellent
    2: 'green',      # Good
    3: 'gold',       # Typical
    4: 'orangered',  # Fair
    5: 'darkred'     # Poor
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='GarageCondScore',
    category_orders={"GarageCondScore": sorted(Garage_color_map.keys())},
    color_discrete_map={str(k): v for k, v in Garage_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "차고지 상태 점수 (GarageCondScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})


# Year_Remod_Add 시각화
# 1. 데이터 불러오기
df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")

# 2. 10년 단위로 변환
df['RemodDecade'] = (df['YearRemodAdd'] // 10) * 10

# 3. 결측 제거
df = df[['Latitude', 'Longitude', 'RemodDecade']].dropna()

# 4. Plotly 시각화
fig = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='RemodDecade',
    color_continuous_scale='Plasma',  # 또는 Turbo, Viridis 등
    zoom=12,
    mapbox_style='carto-positron',
    title="전체 리모델링 연도 (10년 단위) 분포"
)

fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.show(config={"scrollZoom": True})


# 지역별 지하실 품질 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 지하실 품질 점수 매핑 (나쁠수록 높은 점수)
BsmtQual_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['BsmtQualScore'] = df['BsmtQual'].map(bsmt_cond_mapping)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 지하실 품질 점수 계산
mode_score = (
    df.dropna(subset=['BsmtQualScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['BsmtQualScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
bsmt_color_map = {
    1: 'blue',       # Excellent
    2: 'green',      # Good
    3: 'gold',       # Typical
    4: 'orangered',  # Fair
    5: 'darkred'     # Poor
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='BsmtQualScore',
    category_orders={"BsmtQualScore": sorted(bsmt_color_map.keys())},
    color_discrete_map={str(k): v for k, v in bsmt_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "지하실 품질 점수 (BsmtQualScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})



# 지역별 외장재 품질 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 외장재 품질 점수 매핑 (나쁠수록 높은 점수)
ExterQual_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['ExterQualScore'] = df['ExterQual'].map(ExterQual_mapping)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 외장재 품질 점수 계산
mode_score = (
    df.dropna(subset=['ExterQualScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['ExterQualScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
Exter_color_map = {
    1: 'blue',       # Excellent
    2: 'green',      # Good
    3: 'gold',       # Typical
    4: 'orangered',  # Fair
    5: 'darkred'     # Poor
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='ExterQualScore',
    category_orders={"ExterQualScore": sorted(bsmt_color_map.keys())},
    color_discrete_map={str(k): v for k, v in bsmt_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "외장재 품질 점수 (ExterQualScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})




# 지역별 난방 품질 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 난방 품질 점수 매핑 (나쁠수록 높은 점수)
HeatingQC_mapping = {
    'Ex': 1,  # Excellent
    'Gd': 2,
    'TA': 3,
    'Fa': 4,
    'Po': 5   # Poor
}
df['HeatingQCScore'] = df['HeatingQC'].map(HeatingQC_mapping)
# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지역별 외장재 품질 점수 계산
mode_score = (
    df.dropna(subset=['HeatingQCScore'])
    .groupby(['Lat_bin', 'Lon_bin'])['HeatingQCScore']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)



# 6. 시각화
# 점수 컬럼을 문자열로 변환
mode_score['HeatingQCScore'] = mode_score['HeatingQCScore'].astype(str)

# 색상 매핑 (의미 있는 색상 직접 지정)
HeatingQC_color_map = {
    '1': 'darkred',     # 매우 나쁨
    '2': 'orangered',   # 나쁨
    '3': 'gold',        # 보통
    '4': 'skyblue',     # 좋음
    '5': 'darkblue'     # 매우 좋음
}

# 시각화

fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='HeatingQCScore',
    category_orders={"HeatingQCScore": ['1', '2', '3', '4', '5']},
    color_discrete_map=HeatingQC_color_map,
    zoom=12,
    mapbox_style='carto-positron',
    title="난방 품질 점수 (HeatingQCScore)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(dragmode='zoom')
fig.show(config={"scrollZoom": True})




# 지상 생활공간 점수
# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')
# 2. 지상 생활공간 점수 매핑 (나쁠수록 높은 점수)
df["GrLivArea_Score"] = pd.qcut(df["GrLivArea"], q=4, labels=[1, 2, 3, 4]).astype(int)

# 3. 위경도 구간화 (소수 셋째 자리까지 반올림 → 밀집도 그룹화)
df['Lat_bin'] = (df['Latitude'] * 1000).round() / 1000
df['Lon_bin'] = (df['Longitude'] * 1000).round() / 1000

# 4. 지상 생활공간 점수 계산
mode_score = (
    df.dropna(subset=['GrLivArea_Score'])
    .groupby(['Lat_bin', 'Lon_bin'])['GrLivArea_Score']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# 5. 색상 매핑 (점수 높을수록 더 눈에 띄는 색상)
GrLivArea_color_map = {
    1: 'darkred',     # 매우 좁음
    2: 'orangered',   # 좁음
    3: 'gold',        # 보통
    4: 'skyblue',     # 넓음
    5: 'darkblue'    # 매우 넓음
}

# 6. 시각화
fig = px.scatter_mapbox(
    mode_score,
    lat='Lat_bin',
    lon='Lon_bin',
    color='GrLivArea_Score',
    category_orders={"GrLivArea_Score": sorted(GrLivArea_color_map.keys())},
    color_discrete_map={str(k): v for k, v in GrLivArea_color_map.items()},
    zoom=12,
    mapbox_style='carto-positron',
    title= "지상 생활 공간 점수 (GrLivArea_Score)"
)

fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show(config={"scrollZoom": True})

# # 유지 보수 점수 산출 및 상위 25% 출력
# # 1. 데이터 불러오기
# df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")

# # 2. 사용할 변수
# cond_qual_vars = ['Bsmt_Cond', 'Exter_Cond', 'Garage_Cond', 'Heating_QC', 'Pool_QC', 'Overall_Cond', 'Year_Remod_Add']

# # 3. 문자열 → 점수 매핑
# qual_mapping = {
#     'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,
#     'Poor': 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5,
#     'No_Basement': 0, 'No_Pool': 0, 'No_Garage': 0,
#     'Average': 3, 'Below_Average': 2, 'Above_Average': 4, 'Very_Good': 5, 'Very_Poor': 1
# }

# df_model = df[cond_qual_vars].copy()
# for col in cond_qual_vars:
#     if df_model[col].dtype == object:
#         df_model[col] = df_model[col].map(qual_mapping)

# # 4. 노후도 파생 변수
# df_model['Age'] = 2024 - df_model['Year_Remod_Add']
# df_model.drop(columns='Year_Remod_Add', inplace=True)

# # 5. 결측치 0으로 대체
# df_model.fillna(0, inplace=True)

# # 6. 회귀 학습을 위한 임시 점수 생성
# initial_score = (
#     df_model['Age'] * 0.3 +
#     (6 - df_model['Overall_Cond']) * 0.3 +
#     df_model.drop(columns=['Age', 'Overall_Cond']).sum(axis=1) * (0.4 / (len(df_model.columns) - 2))
# )

# # 7. 표준화
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df_model)

# # 8. Lasso 회귀로 가중치 산출
# lasso = LassoCV(cv=5, max_iter=10000)
# lasso.fit(X_scaled, initial_score)

# # 9. 최종 유지보수 점수 계산
# final_weights = pd.Series(lasso.coef_, index=df_model.columns)
# df['MaintenanceNeedScore'] = X_scaled @ lasso.coef_ + lasso.intercept_

# # 10. 점수 상위 확인
# df[['Latitude', 'Longitude', 'MaintenanceNeedScore']].sort_values(by='MaintenanceNeedScore', ascending=False).head()


# # df.columns












# 라쏘회귀를 통한 condition 점수 산출
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import folium

# 1. 데이터 불러오기
# 1. 데이터 불러오기 및 결측치 제거
df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")
df = df.dropna(subset=['Latitude', 'Longitude'])

# 2. 사용할 변수들
cols = ['BsmtCond', 'ExterCond', 'GarageCond', 'YearRemodAdd']

# 3. 문자열 점수 매핑
qual_mapping = {
    'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,
    'Poor': 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5,
    'No_Basement': 0, 'No_Garage': 0,
    'Average': 3, 'Below_Average': 2, 'Above_Average': 4,
    'Very_Good': 5, 'Very_Poor': 1
}

# 4. 문자열 점수 매핑을 df에 직접 적용
for col in ['BsmtCond', 'ExterCond', 'GarageCond']:
    if df[col].dtype == object:
        df[col] = df[col].map(qual_mapping)

# 5. 리모델링 연도 점수화
def remod_to_score(year):
    if year < 1960: return 6
    elif year < 1970: return 5
    elif year < 1980: return 4
    elif year < 1990: return 3
    elif year < 2000: return 2
    else: return 1

df['RemodScore'] = df['YearRemodAdd'].apply(remod_to_score)

# 6. 상태 점수 뒤집기 (좋음=1 → 나쁨=5)
for col in ['BsmtCond', 'ExterCond', 'GarageCond']:
    df[col] = 5 - df[col]

# 7. 결측치 처리
df[['BsmtCond', 'ExterCond', 'GarageCond', 'RemodScore']] = df[[
    'BsmtCond', 'ExterCond', 'GarageCond', 'RemodScore']].fillna(0)

# 8. 초기 유지보수 필요 점수 계산
df['MaintenanceNeedScore'] = (
    df['RemodScore'] * 0.2 +
    df['GarageCond'] * 0.2 +
    df['ExterCond'] * 0.3 +
    df['BsmtCond'] * 0.3
)

# 9. 상위 200개 추출
top200 = df.sort_values(by='MaintenanceNeedScore', ascending=False).head(200)
top200.to_csv('./data/house/top200.csv', index=True)
top25 = df[df['MaintenanceNeedScore'] >= df['MaintenanceNeedScore'].quantile(0.75)][
    ['Latitude', 'Longitude', 'MaintenanceNeedScore']
].sort_values(by='MaintenanceNeedScore', ascending=False)
top10 = df[df['MaintenanceNeedScore'] >= df['MaintenanceNeedScore'].quantile(0.95)][
    ['Latitude', 'Longitude', 'MaintenanceNeedScore']
].sort_values(by='MaintenanceNeedScore', ascending=False)
top10
top10.to_csv('./data/house/top10.csv', index=True)
top25
center_lat = top10['Latitude'].mean()
center_lon = top10['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

for _, row in top10.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        fill=True,
        color='red',
        fill_opacity=0.7,
        popup=f"Score: {row['MaintenanceNeedScore']:.2f}"
    ).add_to(m)

# 저장
m
# HTML로 저장
m.save("./data/house/top10.html")

# step3. 고가/저가 주택 회귀 계수 비교
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2. 상위 25%, 하위 25% 분할
q_high = df['SalePrice'].quantile(0.75)
q_low = df['SalePrice'].quantile(0.25)
df_high = df[df['SalePrice'] >= q_high].copy()
df_low = df[df['SalePrice'] <= q_low].copy()

# 3. 변수 정의
cond_qual_vars = [
    'KitchenQual', 'FireplaceQu', 'BsmtCond', 'ExterCond', 'GarageCond',
    'OverallCond', 'BsmtQual', 'ExterQual', 'GarageQual', 'HeatingQC'
]

# 4. 품질 매핑
qual_mapping = {
    'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,
    'Poor': 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5,
    'No_Basement': 0, 'No_Garage': 0,
    'Average': 3, 'Below_Average': 2, 'Above_Average': 4,
    'Very_Good': 5, 'Very_Poor': 1
}

# 1. 'OverallCond'는 매핑하지 말고 제외
vars_to_map = [v for v in cond_qual_vars if v != 'OverallCond']

# 2. 매핑 적용
for var in vars_to_map:
    df_high[var] = df_high[var].map(qual_mapping)
    df_low[var] = df_low[var].map(qual_mapping)

# 3. 결측치 평균 대체
df_high[vars_to_map] = df_high[vars_to_map].fillna(df_high[vars_to_map].mean())
df_low[vars_to_map] = df_low[vars_to_map].fillna(df_low[vars_to_map].mean())

# 4. 전체 변수 목록 (매핑한 것 + 그대로 둔 OverallCond)
final_vars = vars_to_map + ['OverallCond']

# 5. 클린 데이터 생성
df_high_clean = df_high[['SalePrice'] + final_vars].dropna()
df_low_clean = df_low[['SalePrice'] + final_vars].dropna()
# 7. 스케일링
scaler_high = StandardScaler()
X_high_scaled = scaler_high.fit_transform(df_high[cond_qual_vars])
y_high = df_high['SalePrice']

scaler_low = StandardScaler()
X_low_scaled = scaler_low.fit_transform(df_low[cond_qual_vars])
y_low = df_low['SalePrice']

# 8. GridSearch 파라미터 정의
param_grid = {
    'alpha': np.arange(0.01, 0.3, 0.01),
    'l1_ratio': np.linspace(0, 1, 20)
}

cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 9. 고가 모델 학습
elastic_high = GridSearchCV(
    estimator=ElasticNet(max_iter=1000),
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
elastic_high.fit(X_high_scaled, y_high)

# 10. 저가 모델 학습
elastic_low = GridSearchCV(
    estimator=ElasticNet(max_iter=1000),
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
elastic_low.fit(X_low_scaled, y_low)

# 11. 결과 출력
print("📈 고가 주택 Best Params:", elastic_high.best_params_)
print("📉 저가 주택 Best Params:", elastic_low.best_params_)

print("\n고가 주택 Feature Coefficients:")
print(pd.Series(elastic_high.best_estimator_.coef_, index=cond_qual_vars).round(2))

print("\n저가 주택 Feature Coefficients:")
print(pd.Series(elastic_low.best_estimator_.coef_, index=cond_qual_vars).round(2))


# 위 결과를 하나로 통합
# 고가/저가 회귀 계수 추출
coef_high = pd.Series(elastic_high.best_estimator_.coef_, index=cond_qual_vars).round(2)
coef_low = pd.Series(elastic_low.best_estimator_.coef_, index=cond_qual_vars).round(2)

# 하나의 테이블로 병합
coef_df = pd.DataFrame({
    '고가 계수': coef_high,
    '저가 계수': coef_low
})

# 출력
coef_df