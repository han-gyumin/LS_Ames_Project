import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

#위도 경도 결측치 제거
df = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
df['Latitude'].isna().sum()
df['Longitude'].isna().sum()


df["ExterQual"].isna().sum()
df["BsmtQual"].isna().sum()
df["HeatingQC"].isna().sum()
df["KitchenQual"].isna().sum()
df["FireplaceQu"].isna().sum()
df["GarageQual"].isna().sum()
df["Fence"].isna().sum()

#각 퀄별 점수 부여 후 score 컬럼 형성 

# 1. 품질 관련 등급 → 점수 매핑 (5점~1점), NaN은 0점 처리
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
quality_cols = [
    "BsmtQual", "ExterQual", "FireplaceQu", "GarageQual",
    "HeatingQC", "KitchenQual"
]

for col in quality_cols:
    df[f"{col}_Score"] = df[col].map(quality_map).fillna(0)

# 2. Fence 등급 → 점수 매핑 (4점~1점), NaN은 0점 처리
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}
df["Fence_Score"] = df["Fence"].map(fence_map).fillna(0)

# 3. 연속형 변수 (GrLivArea, SalePrice)를 4분위로 나눠 점수화 (4~1점)
df["GrLivArea_Score"] = pd.qcut(df["GrLivArea"], q=4, labels=[1, 2, 3, 4]).astype(int)
df["SalePrice_Score"] = pd.qcut(df["SalePrice"], q=4, labels=[1, 2, 3, 4]).astype(int)

# 4. 결과 확인용 미리보기
cols_to_show = (
    quality_cols +
    ["Fence", "GrLivArea", "SalePrice"] +
    [f"{col}_Score" for col in quality_cols] +
    ["Fence_Score", "GrLivArea_Score", "SalePrice_Score"]
)

print(df[cols_to_show].head())
df.columns
# score 생김

df['BsmtQual_Score'].isna().sum()
df['ExterQual_Score'].isna().sum()
df['FireplaceQu_Score'].isna().sum()
df['GarageQual_Score'].isna().sum()
df['HeatingQC_Score'].isna().sum()
df['KitchenQual_Score'].isna().sum()
df['Fence_Score'].isna().sum()
df['GrLivArea_Score'].isna().sum()
df['SalePrice_Score'].isna().sum()

# NaN값 없이 score 점수 잘 형성 됨

# 우선 Quality 관련 컬럼들을 평균을 구하여 'Quality_AvgScore' 컬럼 형성
# 그리고 'Quality_AvgScore' 'GrLivArea_Score' 'SalePrice_Score' 컬럼에 가중치를 부여하여
#  최종 'InteriorBusiness_Score' 를 만들어서
#  이컬럼을 기반으로 지도 시각화 상위권 집 필터링하기

# 1. Quality 관련 평균 점수 계산
quality_cols = [
    'ExterQual_Score', 'KitchenQual_Score', 'BsmtQual_Score',
    'FireplaceQu_Score', 'GarageQual_Score', 'HeatingQC_Score'
]
df['Quality_AvgScore'] = df[quality_cols].mean(axis=1)

# 2. 가중치 기반 최종 점수 계산
df['InteriorBusiness_Score'] = (
    df['Quality_AvgScore'] * 0.40 +
    df['GrLivArea_Score'] * 0.30 +
    df['SalePrice_Score'] * 0.30
)

#############################################
import folium
import pandas as pd
from branca.colormap import linear

dff = pd.read_csv('./data/house/top200.csv')
# 1. 상위 % 기준점 계산
threshold = dff['InteriorBusiness_Score'].quantile(0.75)

# 2. 상위 1%에 해당하는 집 필터링 + 위경도 결측 제거
top_1_percent = dff[(dff['InteriorBusiness_Score'] >= threshold) & 
                   dff['Latitude'].notna() & dff['Longitude'].notna()]

# 3. 지도 생성 (중심 위치: 평균 좌표)
center_lat = top_1_percent['Latitude'].mean()
center_lon = top_1_percent['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# 4. 점수에 따른 색상 매핑 (선택적)
color_scale = linear.PuRd_09.scale(
    top_1_percent['InteriorBusiness_Score'].min(),
    top_1_percent['InteriorBusiness_Score'].max()
)
color_scale.caption = 'Top 1% Interior Business Score'

# 5. 마커 추가 - 빨간색 + 팝업에 점수 표시
for _, row in top_1_percent.iterrows():
    popup_html = f"""
    <b>Interior Score:</b> {row['InteriorBusiness_Score']:.2f}<br>
    <b>Lat:</b> {row['Latitude']:.5f}<br>
    <b>Lon:</b> {row['Longitude']:.5f}
    """
    popup = folium.Popup(popup_html, max_width=250)
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.8,
        popup=popup
    ).add_to(m)
m
m.save("./data/house/final.html")