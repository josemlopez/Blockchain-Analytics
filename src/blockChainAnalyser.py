"""

"""
import blockcypher
import numpy as np
import pandas as pd
from datetime import datetime
import time


class BlckChainAnalyser:
    @staticmethod
    def calculate_balance_from_txr(trxs: list) -> int:
        """Calculates the balance from a list of txr. The function doesn't check if the transactions are in
        the same block

        Args:
            trxs (list): List of transactions in the format returned by blockcypher API

        Returns:
            int: Sum of all the values in the list of transactions
        """
        assert trxs is not None
        # starting_balance = trxs[-1]["ref_balance"]-trxs[-1]["value"]
        acc_balance = 0
        for trx in trxs:
            acc_balance = np.sign(trx["tx_output_n"]) * trx["value"] + acc_balance
        return acc_balance

    @staticmethod
    def trxs_filter_by_date(trxs: list, init_date: datetime, end_date: datetime) -> list:
        """Filter the list of BTC transactions passed by argument and obtained using blockcypher API to be between
        init_date end_date

        Args:
            trxs (list): List of transactions in the format returned by blockcypher API
            init_date (datetime): Initial datetime of the range. Needs to be UTC
            end_date (datetime): End datetime of the range. Needs to be UTC

        Returns:
            list: A sublist of the initial list containing transactions above or equal to init_date and strictly bellow
            to end_date. An empty list if there are no
            transactions in that range of dates
        """
        assert init_date.tzname() == "UTC"
        assert end_date.tzname() == "UTC"
        res = [trx for trx in trxs if init_date <= trx["confirmed"] < end_date]
        return res

    @staticmethod
    def trxs_filter_by_address(trxs: list, wallet_address: str) -> list:
        """Filters the list of BTC transactions by address

        Args:
            trxs (list): Lists of transactions in the format returned by blockcypher
            wallet_address (str): Wallet address

        Raises:
            err: [description]

        Returns:
            list: List of transactions filtered by wallet
        """
        assert len(trxs) > 0
        res = [trx for trx in trxs if trx['address'] == wallet_address]
        return res

    @staticmethod
    def get_details_addresses_low_profile(list_addresses: list, wait: float = 0.5) -> list:
        """Make n calls to get_address_details being n the len of list_addresses. It will wait [wait] seconds between
        each call, to avoid ERROR 429 from the server

        Args:
            list_addresses (list): List containing all the addresses to be query
            wait (float): Seconds between each call to avoid error from the server
        Returns:
            list: list of dict containing: {"address": "<address>", "details": {<details returned from the server>}}
        """
        details_addresses = []
        for addr in list_addresses:
            try:
                details_addresses_temp = blockcypher.get_address_details(addr)
            except Exception as err:
                print(len(details_addresses))
                raise err

            details_addresses.append({"address": addr, "details": details_addresses_temp})
            time.sleep(wait)
        return details_addresses


class ToolboxPriceAnalytics(object):
    @staticmethod
    def dataframe_from_historicalklines(data_klines: list) -> pd.DataFrame:
        """Get data list from get_historical_Klines and creates a Dataframe.

        Args:
            data_klines (list): A list of candles obtained using get_historical_Klines

        Returns:
            pd.DataFrame: A dataframe containing these columns
            ["time", "open", "high", "low", "close", "volume", "close_time", "qav",
            "ntrades", "tbav", "tbqav", "ignore"]
        """
        data = pd.DataFrame(data_klines,
                            columns=["time", "open", "high", "low", "close", "volume", "close_time", "qav", "ntrades",
                                     "tbav", "tbqav", "ignore"])
        data = data.apply(pd.to_numeric)
        # Cleaning the columns that won't be needed
        data.drop("close_time", inplace=True, axis=1)
        data.drop("qav", inplace=True, axis=1)
        data.drop("tbav", inplace=True, axis=1)
        data.drop("tbqav", inplace=True, axis=1)
        data.drop("ignore", inplace=True, axis=1)
        return data

    @staticmethod
    def study_max_delta(data: pd.DataFrame, windows: int) -> pd.DataFrame:
        """Includes in the dataframe (data) 4 columns per windows, calculating:
        1) close_w_<#window>              : close price in # window ahead
        2) close_delta_w_<#window>        : delta between the current price and the same we will find in <#window> ahead
        3) close_delta_w_per_<#window>    : delta in percentage
        4) close_low_consecutive_<#window>: 1 when all the previous candles' delta where bellow the threshold

        Args:
            data (pd.DataFrame): A dataframe prepared with, at least, dataframe_from_historicalKlines columns
            windows (int): Number of windows that will be calculated

        Returns:
            pd.DataFrame: [description]
        """
        assert data.empty is False
        assert windows > 0
        for i in range(1, windows + 1):
            data['close_next_cdl_' + str(i)] = data['close'].shift(-i)
            data.fillna(0, inplace=True)
            data['close_delta_next_cdl_' + str(i)] = data['close_next_cdl_' + str(i)] - data['close']
            data['close_delta_next_cdl_pert_' + str(i)] = (data['close_next_cdl_' + str(i)] - data['close']) / data[
                'close'] * 100
            # data['close_low_consecutive_'+str(i)] = np.where(data[])
        return data
