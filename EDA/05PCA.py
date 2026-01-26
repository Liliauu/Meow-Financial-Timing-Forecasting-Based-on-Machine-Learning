import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 假设HDF5文件名为'example.h5'
hdf5_filename = 'cleaned_archive_20230630.h5'

df = pd.read_hdf(hdf5_filename, key='h5')

numeric_features = [col for col in df.columns if
                    col in ['fret12', 'midpx', 'lastpx', 'open', 'high',
       'low', 'bid0', 'ask0', 'bid4', 'ask4', 'bid9', 'ask9', 'bid19', 'ask19',
       'bsize0', 'asize0', 'bsize0_4', 'asize0_4', 'bsize5_9', 'asize5_9',
       'bsize10_19', 'asize10_19', 'btr0_4', 'atr0_4', 'btr5_9', 'atr5_9',
       'btr10_19', 'atr10_19', 'nTradeBuy', 'tradeBuyQty', 'tradeBuyTurnover',
       'tradeBuyHigh', 'tradeBuyLow', 'buyVwad', 'nTradeSell', 'tradeSellQty',
       'tradeSellTurnover', 'tradeSellHigh', 'tradeSellLow', 'sellVwad',
       'nAddBuy', 'addBuyQty', 'addBuyTurnover', 'addBuyHigh', 'addBuyLow',
       'nAddSell', 'addSellQty', 'addSellTurnover', 'addSellHigh',
       'addSellLow', 'nCxlBuy', 'cxlBuyQty', 'cxlBuyTurnover', 'cxlBuyHigh',
       'cxlBuyLow', 'nCxlSell', 'cxlSellQty', 'cxlSellTurnover', 'cxlSellHigh',
       'cxlSellLow']]

# 确保 DataFrame 中的缺失值在标准化之前被处理
# 例如，使用列的中位数来填充 NaN 值
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_features])

# 应用PCA
pca = PCA()  # 默认保留所有主成分
# 或者，指定要保留的主成分数量，例如 pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 创建一个 DataFrame 来存储 PCA 特征
pca_df = pd.DataFrame(X_pca, columns=[f'pca_feature_{i + 1}' for i in range(X_pca.shape[1])])

# 查看组件向量，即每个新特征是如何由原始特征变换而来的
component_matrix = pd.DataFrame(pca.components_, columns=numeric_features)
component_matrix.index = [f'pca_component_{i + 1}' for i in range(component_matrix.shape[0])]

# 打印第一个主成分（即pca_component_1）的组件向量
print(component_matrix.loc['pca_component_1'])

# 或者打印所有组件向量
print(component_matrix)

# 将组件向量保存到CSV文件中
component_matrix.to_csv('pca_components.csv')