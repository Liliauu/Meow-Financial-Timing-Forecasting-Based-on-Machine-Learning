import pandas as pd

# 假设HDF5文件名为'example.h5'且数据集名为'my_dataframe'
hdf5_filename = 'cleaned_archive_20230630.h5'
df = pd.read_hdf(hdf5_filename)

# 检查缺失值
print(df.isnull().sum())
print("01end------------------------------")
df.isnull().sum().to_csv('missing_values_after.csv')

# # 检查数据类型
# print(df.dtypes)
# print("03end------------------------------")
# # 检查重复值
# print(df.duplicated().sum())
# print("04end------------------------------")
#
# # 如果发现需要清洗的问题，可以对DataFrame进行修改，并保存到新的HDF5文件中
# # 例如，填充缺失值
# df.fillna(method='ffill', inplace=True)  # 前向填充缺失值
#
# # 保存修改后的DataFrame到新的HDF5文件
# df.to_hdf('cleaned_archive_20230630.h5', key='h5')