import pandas as pd
# import matplotlib.pyplot as plt

def load_df():
    path1 = 'dataset/taipower-20190101-20211231.csv'
    path2 = 'dataset/taipower-20210101-20220329.csv'

    df1 = pd.read_csv(path1, usecols=['日期', '備轉容量(萬瓩)'])
    df2 = pd.read_csv(path2, usecols=['日期', '備轉容量(MW)'])

    df1['日期'] = pd.to_datetime(df1['日期'].astype(str), format='%Y-%m-%d')
    df2['日期'] = pd.to_datetime(df2['日期'].astype(str), format='%Y-%m-%d')
    df1['備轉容量(萬瓩)'] = df1['備轉容量(萬瓩)'].apply(lambda x: int(x*10))
    df1.rename(columns={'備轉容量(萬瓩)': '備轉容量(MW)'}, inplace=True)

    df = pd.concat([df1, df2]).drop_duplicates('日期', keep='last').reset_index(drop=True)

    '''plt.style.use('ggplot')
    plt.figure(figsize=(20, 9))
    plt.plot(df['date'], df['operating_reserve(MW)'])
    plt.show()'''

    return df