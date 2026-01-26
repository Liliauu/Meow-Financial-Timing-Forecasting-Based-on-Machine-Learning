import pandas as pd

# 假设HDF5文件名为'example.h5'
hdf5_filename = 'cleaned_archive_20230630.h5'

# 使用Pandas的read_hdf函数读取HDF5文件
# 注意：这里不需要指定数据集名，因为Pandas的HDF5格式会存储一些元数据来标识DataFrame
# 假设HDF5文件中有一个名为'my_dataframe'的DataFrame
df = pd.read_hdf(hdf5_filename, key='h5')

# 计算fret12与数值型特征之间的皮尔逊相关系数
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('fret12')
corr_matrix = df[['fret12'] + list(numeric_cols)].corr()

import matplotlib.pyplot as plt
import seaborn as sns

# 假设 df 已经被定义，并且 numeric_cols 已经包含除了 'fret12' 之外的所有数值型特征的列名
corr_subset = df[['fret12'] + list(numeric_cols)].corr()

# 绘制热力图，不显示数字
plt.figure(figsize=(15, 10))
sns.heatmap(corr_subset, annot=False, cmap='coolwarm', center=0)
plt.show()

