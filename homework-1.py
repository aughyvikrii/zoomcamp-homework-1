import pandas as pd
import numpy as np

#no 1
#print("Numpy: " + np.__version__)

#no 2
#print("Pandas: " + pd.__version__)

df = pd.read_csv(r'data.csv')

#no 3
#mean = df.groupby('Make').mean()

#no4
#data = df.loc[df['Year']>=2015]
#data.sort_values('Year', inplace=True)
#print(data.isnull().sum())

#no5
#mean_hp_before = df['Engine HP'].mean()
#fill = df.fillna(0)
#mean_hp_after = fill['Engine HP'].mean()
#print('before:', round(mean_hp_before))
#print('after:', round(mean_hp_after))

#no6
data1 = df.loc[df['Make']=='Rolls-Royce']
data2 = data1[['Engine HP', 'Engine Cylinders', 'highway MPG']]
data3 = data2.drop_duplicates()

arr_x = np.array(data3)
arr_xtrans = arr_x.transpose()
xt_m_x = np.dot(arr_xtrans, arr_x)
xt_m_x_inv = np.linalg.inv(xt_m_x)
# print(np.sum(xt_m_x_inv))

#no7
arr_y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
res = np.dot(xt_m_x_inv, arr_xtrans)
res_m_y = np.dot(res, arr_y)
print(res_m_y)