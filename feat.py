import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    @classmethod
    def featureNames(cls):
        return [
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "trade_imb",
            "trade_imbema5",
            "bret12",
            "lagret12",
            "lagret1",
            "lagret2",
            "lagret4",
            "lagret8",

            "movavg_trade_imb",
            "alpha1",
            "alpha2",
            "momentum",
            "volatility",
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "depth_ratio",
            "price_change_rate",
            "trade_volume_ratio",
            "vwap",
            "new_order_cancel_ratio",
            "open_lastpx_diff",
            "high_lastpx_diff",
            "low_lastpx_diff",
            "order_book_bsize0_change",
            "order_book_asize0_change",
            "order_book_bsize04_change",
            "order_book_asize04_change",
            "order_book_bsize59_change",
            "order_book_asize59_change",
            "price_vwap_deviation",
            "range_to_avg_ratio",
            "trade_volume_depth_ratio",


        ]

    def genFeatures(self, df):

        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))

        # 计算订单簿不平衡特征
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        df.loc[:, "ob_imb19"] = (df["asize10_19"] - df["bsize10_19"]) / (df["asize10_19"] + df["bsize10_19"])

        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()

        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)

        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")

        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        df.loc[:, "lagret1"] = df["bret12"].shift(1)
        df.loc[:, "lagret2"] = df["bret12"].shift(2)
        df.loc[:, "lagret4"] = df["bret12"].shift(4)
        df.loc[:, "lagret8"] = df["bret12"].shift(8)
        df.loc[:, "movavg_trade_imb"] = df["trade_imb"].rolling(window=10, min_periods=1).mean()

        df.loc[:, "alpha1"] = (df["lastpx"] - df["open"]) / df["lastpx"]
        df.loc[:, "alpha2"] = (df["high"] - df["low"]) / df["tradeBuyQty"]

        df.loc[:, "momentum"] = df["lastpx"] - df["lastpx"].shift(10)
        df.loc[:, "volatility"] = df["lastpx"].rolling(window=10).std()

        delta = df["lastpx"].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df.loc[:, "rsi"] = 100 - (100 / (1 + rs))

        ema12 = df["lastpx"].ewm(span=12, adjust=False).mean()
        ema26 = df["lastpx"].ewm(span=26, adjust=False).mean()
        df.loc[:, "macd"] = ema12 - ema26

        df["bollinger_mid"] = df["lastpx"].rolling(window=20).mean()
        df["bollinger_std"] = df["lastpx"].rolling(window=20).std()
        df.loc[:, "bollinger_upper"] = df["bollinger_mid"] + (df["bollinger_std"] * 2)
        df.loc[:, "bollinger_lower"] = df["bollinger_mid"] - (df["bollinger_std"] * 2)

        df.loc[:, "depth_ratio"] = df["bid0"] / df["ask0"]
        df.loc[:, "price_change_rate"] = (df["lastpx"] - df["open"]) / df["open"]
        df.loc[:, "trade_volume_ratio"] = df["tradeBuyQty"] / df["tradeSellQty"]
        df.loc[:, "vwap"] = (df["buyVwad"] + df["sellVwad"]) / 2
        df.loc[:, "new_order_cancel_ratio"] = (df["addBuyQty"] - df["addSellQty"]) / (
                    df["addBuyQty"] + df["addSellQty"])

        df.loc[:, "open_lastpx_diff"] = df["open"] - df["lastpx"]
        df.loc[:, "high_lastpx_diff"] = df["high"] - df["lastpx"]
        df.loc[:, "low_lastpx_diff"] = df["low"] - df["lastpx"]
        df.loc[:, "order_book_bsize0_change"] = df["bsize0"] - df["bsize0"].shift(1)
        df.loc[:, "order_book_asize0_change"] = df["asize0"] - df["asize0"].shift(1)
        df.loc[:, "order_book_bsize04_change"] = df["bsize0_4"] - df["bsize0_4"].shift(4)
        df.loc[:, "order_book_asize04_change"] = df["asize0_4"] - df["asize0_4"].shift(4)
        df.loc[:, "order_book_bsize59_change"] = df["bsize5_9"] - df["bsize5_9"].shift(9)
        df.loc[:, "order_book_asize59_change"] = df["asize5_9"] - df["asize5_9"].shift(9)

        df.loc[:, "price_vwap_deviation"] = (df["lastpx"] - df["vwap"]) / df["vwap"]
        df.loc[:, "range_to_avg_ratio"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2)
        df.loc[:, "trade_volume_depth_ratio"] = df["tradeBuyQty"] / (df["bsize0"] + df["asize0"])

        xdf = df[self.mcols + self.featureNames()].copy()
        xdf['date'] = pd.to_datetime(xdf['date'])
        xdf.set_index('date', inplace=True)

        xdf['day_of_week'] = xdf.index.dayofweek
        xdf['day_of_month'] = xdf.index.day
        xdf['month'] = xdf.index.month

        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)

        return xdf.fillna(0), ydf.fillna(0)
