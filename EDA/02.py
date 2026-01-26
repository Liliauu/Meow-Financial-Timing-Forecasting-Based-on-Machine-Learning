import pandas as pd

# 假设HDF5文件名为'example.h5'
hdf5_filename = 'archive/20230630.h5'

# 使用Pandas的read_hdf函数读取HDF5文件
# 注意：这里不需要指定数据集名，因为Pandas的HDF5格式会存储一些元数据来标识DataFrame
# 假设HDF5文件中有一个名为'my_dataframe'的DataFrame
df = pd.read_hdf(hdf5_filename, key='h5')

# 查看DataFrame的前几行
print(df.head(6))
print(df.columns)
print("---------------")
# 假设我们想要填充'cxlSellHigh'和'cxlSellLow'列的NaN值，这里使用0作为填充值
df['cxlSellHigh'] = df['cxlSellHigh'].fillna(0)
df['cxlSellLow'] = df['cxlSellLow'].fillna(0)

# 查看处理后的数据
print(df[['cxlSellHigh', 'cxlSellLow']].head())
print("---------------")
# 查看'fret12'列的统计信息
print(df['fret12'].describe())
print("---------------")
# 对数据进行分组，比如按'symbol'列分组并查看每个'symbol'的平均'fret12'
grouped = df.groupby('symbol')['fret12'].mean()
print(grouped)
print("---------------")

import matplotlib.pyplot as plt

# 假设'interval'是时间戳或可以转换为时间戳的整数
df['interval'] = pd.to_datetime(df['interval'], unit='ms')  # 这里假设interval是以毫秒为单位的时间戳
df.set_index('interval', inplace=True)  # 设置时间为索引

# 绘制'fret12'的折线图
df['fret12'].plot(kind='line')
plt.title('fret12 Over Time')
plt.xlabel('Time')
plt.ylabel('fret12')
plt.show()

#直方图
df['fret12'].plot(kind='hist', bins=30, figsize=(10, 6))
plt.title('Histogram of fret12')
plt.xlabel('fret12')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df['fret12'], df['cxlSellTurnover'])
plt.title('Scatter Plot of fret12 vs cxlSellTurnover')
plt.xlabel('cxlSellTurnover')
plt.ylabel('fret12')
plt.show()