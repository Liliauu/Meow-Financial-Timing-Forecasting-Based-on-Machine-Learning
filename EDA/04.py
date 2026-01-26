import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 假设HDF5文件名为'example.h5'
hdf5_filename = 'archive/20230630.h5'

# 使用Pandas的read_hdf函数读取HDF5文件
# 注意：这里不需要指定数据集名，因为Pandas的HDF5格式会存储一些元数据来标识DataFrame
# 假设HDF5文件中有一个名为'my_dataframe'的DataFrame
df = pd.read_hdf(hdf5_filename, key='h5')
# 提取数值型特征

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('fret12')

# 计算相关性并创建热力图
corr_matrix = df[['fret12'] + list(numeric_features)].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 对于分类特征，可以使用箱线图或其他可视化方法
# 例如，查看'symbol'与'fret12'的关系
sns.boxplot(x='symbol', y='fret12', data=df)
plt.xticks(rotation=45)  # 旋转x轴标签以便于阅读
plt.show()

# 散点图矩阵（仅用于数值型特征）
sns.pairplot(df[['fret12'] + list(numeric_features)], vars=list(numeric_features), hue='fret12', palette='coolwarm')
plt.show()