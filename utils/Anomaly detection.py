import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class IQRDetection:
    ##데이터프레임의 IQR을 탐지해서 하나의 DataFrame 으로 반환합니다.
    ##IQR 을 계산할때 디폴트인 1.5 값을 늘려 이상치를 줄일수 있다.
    def IQR(dff, IQR_Range = 1.5):
        data = pd.DataFrame()
        #데이터 프레임을 컬럼단위로 계산해서 data에 넣습니다.
        for i in dff:
            Q1 = np.percentile(dff[i], 25)
            Q3 = np.percentile(dff[i], 75)
            iqr =  Q3 - Q1
            outlier = IQR_Range * iqr
            minimum = Q1 - outlier
            maximum = Q3 + outlier
            multiple_outliers = dff[(dff[i] < minimum) | (dff[i] > maximum)]
            data = pd.concat([data,multiple_outliers])
        data.drop_duplicates()
        return data


    #특정 컬럼의 IQR을 반환합니다. IQR 함수의를 축소 했습니다. 둘중에 하나를 사용하면 됩니다.
    #데이터 프레임 하나와 데이터프레임에서 본인이 원하는 컬럼의 이상치만을 반환합니다.
    #ascending 이 0 이면 내림차순 1 이면 오름차순 입니다.
    ##IQR 을 계산할때 디폴트인 1.5 값을 늘려 이상치를 줄일수 있다.
    def IQR_column(df, col, ascending = 0, IQR_Range = 1.5):
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        iqr =  Q3 - Q1
        outlier = IQR_Range * iqr
        minimum = Q1 - outlier
        maximum = Q3 + outlier
        multiple_outliers = df[(df[col] < minimum) | (df[col] > maximum)]
        if ascending == 0:
            multiple_outliers = multiple_outliers.sort_values(col,ascending = False)
        else:
            multiple_outliers = multiple_outliers.sort_values(col,ascending = True)
        return multiple_outliers

    #박스플롯을 만드는 함수입니다.
    def make_boxplot(df) :
        try:
            k = len(df.columns)
            plt.figure(figsize=(25,20))
            for i in range(k) : 
                plt.subplot(3,5,i+1) # 꼭 +1 을 붙여줘야한다.
                sns.boxplot(data=df.iloc[:,i])
                plt.title(df.columns[i])
            plt.tight_layout()
            plt.show()
        except:
            print('문자 컬럼 빼주세요.')
    
    
    #df는 데이터 프레임이고 
    # col은 내가 이상값을 빼고 싶은 컬럼. 
    # count는 몇개를 빼고 싶은지,
    #  스타팅 포인트는 어디서 부터 뺄건지,
    #  up는 위에서 뺄건지, up의 기본값은 1이며 0을 넣으면 이상값을 위에서는 빼지 않습니다.
    #  down은 아래서 뺄건지 정하는 겁니다. 기본값이 0이며 1을 넣으면 아래서도 뺍니다.
    def Anomaly_Remove(df, col, count, starting_point = 0, up = 1, down = 0):
        data = pd.DataFrame()
        if up == 1:
            data = pd.concat([data,IQRDetection.IQR_column(df,col)[starting_point:count]])
        if down == 1:
            data = pd.concat([data,IQRDetection.IQR_column(df,col,ascending=1)[starting_point:count]])
        remove = df.drop(data.index)
        return remove
