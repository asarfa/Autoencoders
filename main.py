import pandas as pd
import numpy as np
import talib
import pickle
import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Dict, Tuple
from talib.abstract import Function
from sklearn.preprocessing import MinMaxScaler

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

pd.set_option('precision', 4)


# torch.manual_seed(0)


class BaseData:
    """
    This class allows to request data and compute features (i.e factors/characteristics)
    The data used are:
    _ The top 10 stocks of Nasdaq 100 in term of capitalization, and some of there market fields
    _ The nasdaq 100 himself
    _ The US T-Bil 3 months to serve as risk free index in order to calcul excess returns
    _ Daily frequency
    _ From 2011 to 2020
    The data are therefore not the same as in the paper, which are:
    _ Monthly individual stock for all firms listed in the three major exchanges: NYSE, AMEX, and NASDAQ
    _ From 1957 to 2016
    _ 94 characteristics (fields/factors)
    These data were clearly not free and not obtainable
    """

    def __init__(self):
        # self.request()
        self.compute()

    @property
    def risk_free(self) -> str:
        return 'US T-Bill 1-3M'

    @property
    def bench(self) -> str:
        return 'Nasdaq 100'

    @property
    def tickers_equity(self) -> List[str]:
        return ['Apple', 'Microsoft', 'Alphabet', 'Amazon', 'Tesla', 'Meta',
                'Nvidia', 'Broadcom', 'COST', 'Intel']

    @property
    def new_tickers(self) -> List[str]:
        return [self.risk_free] + [self.bench] + self.tickers_equity

    def request(self):
        """
        This method uses a function which requests bloomberg for some tickers
        and some fields and save it into a dataframe
        Tried to request the same characteristic as in the paper, like market cap, beta, leverage, dividend yields
        """
        start = "31/12/2010"
        end = "31/12/2020"
        field = "PX_LAST, PX_HIGH, PX_LOW, PX_OPEN, PX_VOLUME, CUR_MKT_CAP, EQY_RAW_BETA, IVOL, FXOPT_VOLATILITY, TURNOVER," \
                "LQA_LIQUIDITY_SCORE, AVERAGE_BID_ASK_SPREAD, LQA_PRICE_VOLATILITY, EQY_DVD_YLD_EST, PX_TO_BOOK_RATIO, " \
                "OPER_INC_GROWTH, PX_TO_SALES_RATIO, RD_EXPEND_TO_NET_SALES, RETURN_ON_ASSET, FNCL_LVRG, ASSET_GROWTH, " \
                "BS_LT_BORROW"
        options = {"Fill": "B"}
        tickers = ['LD12TRUU Index', 'NDX Index', 'AAPL UW Equity', 'MSFT UW Equity', 'GOOG UW Equity',
                   'AMZN UW Equity', 'TSLA UW Equity',
                   'FB UW Equity', 'NVDA UW Equity', 'AVGO UW Equity', 'COST UW Equity', 'INTC UW Equity']
        # data = request_bloom(start, end, field, tickers, new_tickers, options)
        # with open('data/data.pkl', 'wb') as file:
        #    pickle.dump(data, file)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        with open('data/data.pkl', 'rb') as file:
            data = pickle.load(file)
        return data

    def interpolate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        This method allows to do an upward linear interpolation on data
        Indeed some times there is nan data and somes times the data are monthly (for characteristics)
        """
        for ticker in self.new_tickers:
            data[ticker] = pd.DataFrame(np.where(data[ticker] == '', np.nan, data[ticker]).astype(float),
                                        index=data[ticker].index,
                                        columns=data[ticker].columns)
            data[ticker] = data[ticker].ffill()
        return data

    def compute_talib_fields(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        As there is not enough features (characteristics) from the bloomberg request, used the talib package which
        perform technical analysis of financial market data and help identifying different patterns that stocks follow
        """
        overlap_studies = {'sma': 30, 'dema': 30, 'ema': 30, 'ht_trendline': 30, 'kama': 30, 'ma': 30, 'ma': 90,
                           'midpoint': 14, 't3': 5, 'tema': 30, 'trima': 30, 'wma': 30}
        momentum_indic = {'adx': 14, 'adxr': 14, 'aroonosc': 14, 'cci': 14, 'cmo': 14, 'dx': 14,
                          'mfi': 14, 'mom': 5, 'plus_di': 14, 'plus_dm': 14, 'rsi': 14, 'willr': 14, 'rocp': None}
        volume_indic = {'ad': None, 'adosc': None, 'obv': None}
        volatility_indic = {'atr': 14, 'natr': 14, 'trange': None}
        talib_field = {}
        for d in (overlap_studies, momentum_indic, volume_indic, volatility_indic):
            talib_field.update(d)
        talib_pattern_regonition = talib.get_function_groups()['Pattern Recognition']
        for ticker in [self.bench] + self.tickers_equity:
            inputs = {
                'open': data[ticker]['PX_OPEN'],
                'high': data[ticker]['PX_HIGH'],
                'low': data[ticker]['PX_LOW'],
                'close': data[ticker]['PX_LAST'],
                'volume': data[ticker]['PX_VOLUME']
            }
            for field, timeperiod in talib_field.items():
                if timeperiod is not None:
                    data[ticker][f'{field}_{timeperiod}d'] = Function(field)(inputs, timeperiod=timeperiod)
                else:
                    data[ticker][f'{field}'] = Function(field)(inputs)
            for field in talib_pattern_regonition:
                data[ticker][f'{field}'] = Function(field)(inputs)
            self.risk_free_rocp = data[self.risk_free]['PX_LAST'].pct_change()
            data[ticker]['excess_return'] = data[ticker]['rocp'] - self.risk_free_rocp
        return data, list(talib_field.keys()) + talib_pattern_regonition + ['excess_return']

    def clean_field(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        This method allows to get the equity field that are nan (from bloomberg request) in order to delete them
        """
        na_field = []
        for ticker in self.tickers_equity:
            for col in data[ticker]:
                if data[ticker][col].notna().sum() == 0:
                    na_field.append(col)
        return list(set(na_field))

    def delete_field(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        This method allows to delete the field which are nan, drop the nan values,
        and to inner merged the dataframes in order to have the same shape for all
        """
        na_field = self.clean_field(data)
        list_data = []
        for ticker in [self.bench] + self.tickers_equity:
            data[ticker] = data[ticker].drop(na_field, axis=1)
            if ticker in self.tickers_equity:
                data[ticker].dropna(inplace=True)
            list_data.append(data[ticker])
        all_field = list(data[ticker].columns)
        data_inner_merged = pd.concat(list_data, axis=1, join='inner')
        data_inner_merged.columns = pd.MultiIndex.from_product([[self.bench] + self.tickers_equity, all_field])
        return dict(
            map(lambda ticker: (ticker, data_inner_merged[ticker]), [self.bench] + self.tickers_equity)), all_field

    def compute_bloom_fields(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        This method allows to infer the nan fields of the bench (Nasdaq 100) by taking the mean of the 10 composants fields
        """
        data, all_field = self.delete_field(data)
        for field in data[self.bench]:
            if data[self.bench][field].notna().sum() == 0:
                data[self.bench][field] = pd.concat([data[ticker][field] for ticker in self.tickers_equity],
                                                    axis=1).mean(axis=1)
        return data, all_field

    def compute(self):
        """
        Main method that computes all the steps
        self.data is a dict of tickers
        For each tickers the value is a dataframe of all fields from the start to the end date
        These fields are in self.field
        There is 108 fields in total
        """
        self.data = self.load_data()
        self.data = self.interpolate(self.data)
        self.data, self.field = self.compute_talib_fields(self.data)
        self.data, self.field = self.compute_bloom_fields(self.data)
        self.start = self.data[self.bench].index[-1]
        self.end = self.data[self.bench].index[0]
        self.rows = len(self.data[self.bench])
        self.cols = len(self.data[self.bench].T)
        self.keys = self.data.keys()


class ModelData:
    """
    This class allows to create the Xs data (input 1 being asset characteristics (in t-1) and input 2 being asset
    excess returns (in t))
    Then the Xs are split for the training sample, validation sample and testing sample
    Moreover the training, validation and testing set are scaled (Min Max scaler)
    And conversion to tensor
    """

    def __init__(
            self,
            data_base: BaseData = None,
            training_proportion: float = 0.7,
            validation_proportion: float = 0.15,
            testing_proportion: float = 0.15
    ):
        self.data_base = data_base
        self.training_proportion = training_proportion
        self.validation_proportion = validation_proportion
        self.testing_proportion = testing_proportion

    def init_X_1(self):
        """
        From the dataframe, instantiation of array data that will serve in the beta network i.e characteristics
        """
        features = pd.concat(list(
            map(lambda index: pd.concat(
                list(map(lambda tick: self.data_base.data[tick].iloc[index], self.data_base.keys)),
                axis=1, keys=self.data_base.keys).T, range(self.data_base.rows - 1))))  # [0,len(data)-1]
        return np.array(features).reshape(self.data_base.rows - 1, len(self.data_base.keys), self.data_base.cols)

    def init_X_2(self):
        """
        From the dataframe, instantiation of array data that will serve in the factor network i.e excess returns
        """
        excess_return = pd.concat(
            list(map(lambda tick: self.data_base.data[tick]['excess_return'], self.data_base.keys)),
            axis=1, keys=self.data_base.keys).iloc[1:]  # [1,len(data)]
        return np.array(excess_return).reshape(self.data_base.rows - 1, 1, len(self.data_base.keys))

    def init_scaler(self):
        """
        Type of scaler to rank-normalize asset characteristics into the interval (-1,1)
        The rank normalization is done on each day for the asset characteristic of each tickers
        Excess returns are not rank normalize, as in the paper
        Therefore this is not done per asset (all characteristics of an asset by day) but on all characetrstics for all asset
        """
        return MinMaxScaler(feature_range=(-1, 1))

    def set_X(self, start: int, end: int, type: str):
        """
        Converting raw data array into scaled tensor sample (train, valid, test)
        """
        if type is 'training':
            X_1 = np.stack(list(map(lambda index: self.scaler_X_1.fit_transform(self.X_1[start:end][index]),
                                    range(len(self.X_1[start:end])))))
        else:
            X_1 = np.stack(list(map(lambda index: self.scaler_X_1.transform(self.X_1[start:end][index]),
                                    range(len(self.X_1[start:end])))))
        X_2 = self.X_2[start:end]
        return torch.tensor(X_1, dtype=torch.float32), torch.tensor(X_2, dtype=torch.float32)

    def init_train_data(self) -> int:
        """
        Instantiation of the training data
        """
        self.scaler_X_1 = self.init_scaler()
        start = 0
        end = int(self.data_base.rows * self.training_proportion)
        self.X_1_train, self.X_2_train = self.set_X(start, end, 'training')
        return end

    def init_valid_data(self) -> int:
        """
        Instantiation of the validation data
        """
        start = self.init_train_data()
        end = start + int(self.data_base.rows * self.validation_proportion)
        self.X_1_valid, self.X_2_valid = self.set_X(start, end, 'validation')
        return end

    def init_test_data(self):
        """
        Instantiation of the testing data
        """
        start = self.init_valid_data()
        end = start + int(self.data_base.rows * self.testing_proportion)
        self.X_1_test, self.X_2_test = self.set_X(start, end, 'testing')

    def balance_data(self, excess_return_train: torch.tensor):
        """
        Calcul the proportion of positive excess returns and negative excess returns in the training sample
        Indeed we do not want an imbalanced dataset
        """
        nb_excess_returns = len((excess_return_train).view(-1))
        proportion_pos = (excess_return_train > 0).view(-1).sum() / nb_excess_returns
        proportion_neg = 1 - proportion_pos
        print(f'On the training set')
        print(f'Proportion of positive returns: {proportion_pos}')
        print(f'Proportion of negative returns: {proportion_neg}')
        return proportion_pos, proportion_neg

    def compute(self):
        """
        Main method of the class, allowing to perform the steps and saving the arguments because the computation is long
        """
        assert (0 <= self.training_proportion <= 1)
        assert (0 <= self.validation_proportion <= 1)
        assert (0 <= self.testing_proportion <= 1)
        assert (self.training_proportion + self.validation_proportion + self.testing_proportion == 1)
        self.X_1 = self.init_X_1()
        self.X_2 = self.init_X_2()
        assert (self.X_1.shape[0] == self.X_2.shape[0])
        assert (self.X_1.shape[1] == self.X_2.shape[2])
        self.init_train_data()
        proportion_pos, proportion_neg = self.balance_data(self.X_2_train)
        assert (.40 <= proportion_pos <= .60)  # max to consider the dataset has not imbalanced
        self.init_valid_data()
        self.init_test_data()
        self.save_args()

    def save_args(self):
        """
        As the main method of the class is long in debug mode, saving the data to reload after
        """
        args = {'X_1_train': self.X_1_train, 'X_2_train': self.X_2_train,
                'X_1_valid': self.X_1_valid, 'X_2_valid': self.X_2_valid,
                'X_1_test': self.X_1_test, 'X_2_test': self.X_2_test, 'fields': self.data_base.field}
        with open('data/data_model.pkl', "wb") as file:
            pickle.dump(args, file)

    def load_args(self):
        with open('data/data_model.pkl', "rb") as file:
            args = pickle.load(file)
        self.X_1_train = args['X_1_train']
        self.X_2_train = args['X_2_train']
        self.X_1_valid = args['X_1_valid']
        self.X_2_valid = args['X_2_valid']
        self.X_1_test = args['X_1_test']
        self.X_2_test = args['X_2_test']
        self.fields = args['fields']
        return self


@dataclass
class ParamsIO:
    """
    This class allows to define the input and output of each neural network
    """
    input_dim_beta: int
    output_dim_beta: int
    input_dim_factor: int
    output_dim_factor: int


class CA:
    """
    This class allows to define methods which are commune to conditional autoencoders define in the paper
    """

    @staticmethod
    def _factor_network(input_dim_factor, output_dim_factor):
        return nn.Linear(input_dim_factor, output_dim_factor)

    @staticmethod
    def _forward(beta_network, x_beta: torch.Tensor, factor_network, x_factor: torch.Tensor):
        return beta_network(x_beta) @ factor_network(x_factor).T


class CA0(nn.Module):
    """
    First autoencoder of the paper
    Uses a single linear layer in both the beta and factor networks
    Can be compared to a PCA
    No activation needed
    """

    def __init__(
            self,
            params: ParamsIO = None
    ):
        super().__init__()
        self.params = params
        self.beta_network = nn.Linear(self.params.input_dim_beta, self.params.output_dim_beta)
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA1(nn.Module):
    """
    Second autoencoder of the paper
    Difference between above: one hidden layer in the beta network, therefore using relu activation
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']  # Dropout do not figure in the paper, but can be useful
        self.activation = nn.ReLU()
        self.beta_network_base = [('hidden1', nn.Linear(self.params.input_dim_beta, 32)),
                                  ('dropout1', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm1', nn.BatchNorm1d(32)),
                                  ('relu1', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(32, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA2(nn.Module):
    """
    Third autoencoder of the paper
    Difference between above: one more hidden layer in the beta network
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']
        self.activation = nn.ReLU()
        self.beta_network_base = CA1(self.params, config).beta_network_base + \
                                 [('hidden2', nn.Linear(32, 16)),
                                  ('dropout2', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm2', nn.BatchNorm1d(16)),
                                  ('relu2', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(16, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class CA3(nn.Module):
    """
    Third autoencoder of the paper
    Difference between above: one more hidden layer in the beta network
    """

    def __init__(
            self,
            params: ParamsIO = None,
            config: dict = None
    ):
        super().__init__()
        self.params = params
        self.rate_dropout = config['rate_dropout']
        self.activation = nn.ReLU()
        self.beta_network_base = CA2(self.params, config).beta_network_base + \
                                 [('hidden3', nn.Linear(16, 8)),
                                  ('dropout3', nn.Dropout(self.rate_dropout)),
                                  ('batchnorm3', nn.BatchNorm1d(8)),
                                  ('relu3', self.activation)]
        self.beta_network = nn.Sequential(
            OrderedDict(self.beta_network_base + [('linear1', nn.Linear(8, self.params.output_dim_beta))]))
        self.factor_network = CA._factor_network(self.params.input_dim_factor, self.params.output_dim_factor)

    def forward(self, x_beta, x_factor):
        return CA._forward(self.beta_network, x_beta, self.factor_network, x_factor)


class EarlyStopping:
    """
    This class allow to use early stopping in a model that is not hyper-optimized
    Stop the training as soon as the validation error reaches the minimum
    Took on stackoverflow
    """

    def __init__(
            self,
            tolerance: int = None,
            min_delta: int = 10):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class Finance:
    """
    This class allows to define some strategies and compute their performance and sharpe ratio
    """

    @staticmethod
    def compute_strategy_1(ptf_ret: list, outputs: torch.Tensor, X_factor: torch.Tensor):
        """
        Buying the stock with highest expected return and selling the lowest, strat from the paper
        Weights are 1 and - 1
        """
        ret = X_factor.view(-1)[np.argmax(outputs).item()].item() - X_factor.view(-1)[np.argmin(outputs).item()].item()
        ptf_ret.append(ret)
        return ptf_ret

    @staticmethod
    def compute_strategy_2(ptf_ret: list, outputs: torch.Tensor, X_factor: torch.Tensor):
        """
        Buying the stocks with positive expected return and selling the stocks with negative expected return
        Weights are a positive and croissant function of returns
        The sum of weights is either:
        _1 is all returns expected are positive
        _0 if returns expected are positive and negative
        _-1 if returns expected are negative
        """
        idx_pos_ret = np.argwhere(outputs.view(-1) >= 0)
        idx_neg_ret = np.argwhere(outputs.view(-1) < 0)
        pos_ret = outputs[idx_pos_ret]
        neg_ret = outputs[idx_neg_ret]
        weights = torch.zeros(len(outputs.view(-1)))
        weights[idx_pos_ret] = (pos_ret / pos_ret.sum()).view(-1)
        weights[idx_neg_ret] = (-neg_ret / neg_ret.sum()).view(-1)
        ret = (weights @ X_factor.view(-1)).item()
        ptf_ret.append(ret)
        return ptf_ret

    @staticmethod
    def compute_ptf_base(start_sum: int, rets: list):
        for r in rets:
            v = start_sum * (1 + r)
            yield v
            start_sum = v

    @staticmethod
    def compute_ptf_perf(ptf_base: pd.Series):
        return ptf_base.iloc[-1] / ptf_base.iloc[0] - 1

    @staticmethod
    def compute_sharpe_ratio(ptf_ret: pd.Series):
        annualized_ret = ptf_ret.mean() * 252
        annualized_vol = ptf_ret.std() * 252 ** 0.5
        return annualized_ret / annualized_vol

    @staticmethod
    def compute_ptf_metrics(ptf_ret: list):
        ptf_perf = Finance.compute_ptf_perf(pd.Series(list(Finance.compute_ptf_base(100, ptf_ret))))
        try:
            sharpe = Finance.compute_sharpe_ratio(pd.Series(ptf_ret))
        except ZeroDivisionError:
            sharpe = 0
        return ptf_perf, sharpe


class ModelCA:
    """
    This class allows to:
    _Load the data created and transformed
    _Instantiate a model defined above
    _Define the main args (configuration of hyper parameter, optimizer, criterion)
    _Define the main methods (training, validation, test) for the models defined
    _Define the main metrics (loss, r2, accuracy of direction, etc)
    """

    def __init__(
            self,
            model_type: str = None,
            K_factor: int = None,
            max_epochs: int = None,
            tolerance_es: int = None
    ):
        self.model_type = model_type
        self.K_factor = K_factor
        self.max_epochs = max_epochs
        self.tolerance_es = tolerance_es
        self.data_model = ModelData().load_args()
        self.config = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def set_model(self):
        shape_X_beta_train_i = self.data_model.X_1_train[0].shape
        shape_X_factor_train_i = self.data_model.X_2_train[0].shape
        params = ParamsIO(input_dim_beta=shape_X_beta_train_i[1], input_dim_factor=shape_X_factor_train_i[1],
                          output_dim_beta=self.K_factor, output_dim_factor=self.K_factor)
        if self.model_type == 'CA0':
            self.model = CA0(params)
        elif self.model_type == 'CA1':
            self.model = CA1(params, self.config)
        elif self.model_type == 'CA2':
            self.model = CA2(params, self.config)
        elif self.model_type == 'CA3':
            self.model = CA3(params, self.config)
        else:
            raise Exception(f'{self.model_type} has not been implemented')
        """ 
        #Useless with Natixis laptop
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(device)
        """

    def instantiate_args(self, config: dict):
        self.config = config
        self.set_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = nn.MSELoss()

    def train_one_epoch(self):
        X_beta = self.data_model.X_1_train
        X_factor = self.data_model.X_2_train
        train_loss = 0.0
        self.model.train()  # puts the model in training mode (batch normalization and dropout are used)
        for X_beta_train_i, X_factor_train_i in zip(X_beta, X_factor):
            self.optimizer.zero_grad()
            outputs = self.model(X_beta_train_i, X_factor_train_i)
            loss = self.criterion(outputs, torch.reshape(X_factor_train_i, outputs.shape))
            l1_parameters = torch.cat([parameter.view(-1) for parameter in self.model.parameters() if
                                       len(parameter.view(-1)) != self.K_factor])
            l1_parameters_ = torch.cat([parameter.view(-1) for parameter in self.model.parameters()])
            # print(f'just weights:{torch.norm(l1_parameters, 1)}')
            # print(f'weights + bias:{torch.norm(l1_parameters_, 1)}')
            l1_reg = self.config["l1"] * torch.norm(l1_parameters, 1)
            loss += l1_reg
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / (len(X_factor_train_i) * len(X_factor))

    @staticmethod
    def compute_r2_steps(numerator: list, denumerator: list, outputs: torch.Tensor, X_factor_: torch.Tensor):
        X_factor_ = torch.reshape(X_factor_, (-1,))
        predicted, _ = torch.max(outputs.data, 0)
        numerator.append(((X_factor_ - predicted) ** 2).sum().item())
        denumerator.append((X_factor_ ** 2).sum().item())
        return numerator, denumerator

    @staticmethod
    def compute_accuracy_direction(acc: list, outputs: torch.Tensor, X_factor: torch.Tensor):
        acc.append(((np.sign(outputs.view(-1)) == np.sign(X_factor.view(-1))).sum() / len(outputs.view(-1))).item())
        return acc

    def compute_prediction(self, X_beta: torch.Tensor, X_factor: torch.Tensor):
        sample_loss = 0.0
        numerator_r2, denumerator_r2, accuracy_direction, ptf_1_returns, ptf_2_returns = [], [], [], [], []
        self.model.eval()  # puts the model in testing mode
        for X_beta_i, X_factor_i in zip(X_beta, X_factor):
            outputs = self.model(X_beta_i, X_factor_i)
            loss = self.criterion(outputs, torch.reshape(X_factor_i, outputs.shape))
            sample_loss += loss.item()
            numerator_r2, denumerator_r2 = self.compute_r2_steps(numerator_r2, denumerator_r2, outputs, X_factor_i)
            accuracy_direction = self.compute_accuracy_direction(accuracy_direction, outputs, X_factor_i)
            ptf_1_returns = Finance.compute_strategy_1(ptf_1_returns, outputs, X_factor_i)
            ptf_2_returns = Finance.compute_strategy_2(ptf_2_returns, outputs, X_factor_i)
        r2_total = 1 - np.sum(numerator_r2) / np.sum(denumerator_r2)
        ptf_1_perf, sharpe_1 = Finance.compute_ptf_metrics(ptf_1_returns)
        ptf_2_perf, sharpe_2 = Finance.compute_ptf_metrics(ptf_2_returns)
        sample_loss /= (len(X_factor_i.T) * len(X_factor))  # nb of stocks predicted * nb of days in the sample
        return sample_loss, r2_total, np.mean(accuracy_direction), ptf_1_perf, sharpe_1, ptf_2_perf, sharpe_2

    def validate_one_epoch(self):
        X_beta = self.data_model.X_1_valid
        X_factor = self.data_model.X_2_valid
        valid_loss, r2_total, acc_direction, perf_1, sharpe_1, perf_2, sharpe_2 = self.compute_prediction(X_beta,
                                                                                                          X_factor)
        r2_total = r2_total if r2_total > 0 else 0  # in order to maximize r2 (do not want to maximize negative value)
        return valid_loss, r2_total, acc_direction, perf_1, sharpe_1, perf_2, sharpe_2

    def train(self, config: dict, hyperopt: bool):
        if not hyperopt: early_stopping = EarlyStopping(tolerance=self.tolerance_es, min_delta=10)
        self.instantiate_args(config)
        train_loss, validation_loss = [], []
        for _ in range(self.max_epochs):
            # training
            epoch_train_loss = self.train_one_epoch()
            train_loss.append(epoch_train_loss)
            # validation
            # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()
            with torch.no_grad():
                epoch_validate_loss, epoch_validate_r2, epoch_validate_direction_acc, epoch_perf_strat_1, \
                epoch_sharpe_strat_1, epoch_perf_strat_2, epoch_sharpe_strat_2 = self.validate_one_epoch()
                validation_loss.append(epoch_validate_loss)
            if hyperopt:
                with tune.checkpoint_dir(_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                # Send the current validation R2 back to Tune for the hyperopt
                # Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the best results.
                tune.report(R2=epoch_validate_r2, loss=epoch_validate_loss, acc=epoch_validate_direction_acc,
                            perf_1=epoch_perf_strat_1, sharpe_1=epoch_sharpe_strat_1, perf_2=epoch_perf_strat_2,
                            sharpe_2=epoch_sharpe_strat_2)
            else:
                early_stopping(epoch_train_loss, epoch_validate_loss)
                if early_stopping.early_stop:
                    print("We are at epoch:", _)
                    break
                print("Epoch: {} Train Loss: {} Val Loss: {} Val R2: {}".format(
                    _, epoch_train_loss, epoch_validate_loss, epoch_validate_r2))

    def find_epoch_mode_metric(self, analysis, metric, mode, path):
        if mode == 'max':
            epoch = analysis.trial_dataframes[path][metric].iloc[self.tolerance_es:].idxmax()
        elif mode == 'min':
            epoch = analysis.trial_dataframes[path][metric].iloc[self.tolerance_es:].idxmin()
        else:
            raise Exception('Mode must be min or max')
        return epoch, analysis.trial_dataframes[path][metric].loc[epoch]

    def loop_mode_metric(self, analysis, metric, mode, path_trials, name_trials):
        best_metric = 0 if mode == 'max' else 1
        best_trial, best_config, best_epoch, best_checkpoint = None, {}, 0, None
        for path, name in zip(path_trials, name_trials):
            try:
                epoch, metric_result = self.find_epoch_mode_metric(analysis, metric, mode, path)
            except ValueError:
                epoch = 0  # this trial has not enough epochs for the early stopping
                metric_result = 0 if mode == 'max' else 1
            if (mode == 'max' and metric_result > best_metric) or (mode == 'min' and metric_result < best_metric):
                best_metric = metric_result
                best_epoch = epoch
                best_trial = name
                best_config = analysis.results[name]['config']
                if epoch < 10:
                    str_epoch = '00' + str(epoch)
                elif epoch < 100:
                    str_epoch = '0' + str(epoch)
                else:
                    str_epoch = str(epoch)
                best_checkpoint = path.replace('C:\\Users\\asarfati',
                                               os.getcwd()) + '\\checkpoint_000' + str_epoch + '\\checkpoint'
        return best_metric, best_trial, best_config, best_epoch, best_checkpoint

    def compute_mode_metric(self, analysis=None, metric: str = None, mode: str = None):
        """
        Method that allows to browse all the trials of an analyis and in each trials browse all progress (epochs)
        The aim is to find the trial and the epoch which maximize or minimize the metric wanted
        This metric can be different of the metric use in the scheduler
        Indeed for the epoch that has the min loss, its not necessarily the max R2
        The metric wanted should figure in the tune.report()
        :param analysis:
        :param metric: R2, loss, etc
        :param mode: min or max
        :param tolerance: early stopping min
        :return: the saved weights and bias minimizing or maximizing the metric and the config of the model
        """
        path_trials = list(analysis.trial_dataframes.keys())
        name_trials = list(analysis.results.keys())
        best_metric, best_trial, best_config, best_epoch, best_checkpoint = self.loop_mode_metric(analysis, metric,
                                                                                                  mode, path_trials,
                                                                                                  name_trials)
        if best_config is None:
            self.tolerance_es = None
            print('The Early Stopping tolerance has been set to None')
            print('Indeed all the trials of this model were bad and the termination was too early')
            print('Consequence: not a model adapted to the problem')
            best_metric, best_trial, best_config, best_epoch, best_checkpoint = self.loop_mode_metric(analysis, metric,
                                                                                                      mode,
                                                                                                      path_trials,
                                                                                                      name_trials)
        print(f'Best {metric} on the validation set: {"{:.5f}".format(best_metric)}')
        print(f'Occurs for the trial: {best_trial}')
        print(f'The configuration of this trial is: {best_config}')
        print(f'Early stopping occurs for the epoch: {best_epoch}')
        return best_config, best_checkpoint

    def set_characteristic_to_zero(self, X_beta: torch.tensor = None, index: int = None):
        """
        Method that allows to test the importance of characteristic, defined as the reduction in total R2
        All values of a given characteristic are set to zero while holding the remaining model estimates fixed
        """
        X_beta = X_beta.clone().detach()
        X_beta[:, :, index] = 0
        print(f'All values of {self.data_model.fields[index]} has been set to 0')
        return X_beta

    def test(self, analysis=None, metric=None, mode=None, char_idx=None):
        if analysis:
            config, checkpoint = self.compute_mode_metric(analysis, metric, mode)
            if not config:
                print('****************Error on the hyper-optimization of this model****************')
                return 0, 0, 0, 0, 0
            self.instantiate_args(config)
            self.model.load_state_dict(torch.load(checkpoint))
        else:
            print('No hyper-optimization')
        with torch.no_grad():
            X_beta = self.data_model.X_1_test if char_idx is None else self.set_characteristic_to_zero(
                self.data_model.X_1_test, char_idx)
            X_factor = self.data_model.X_2_test
            test_loss, r2_total, acc_direction, perf_1, sharpe_1, perf_2, sharpe_2 = self.compute_prediction(X_beta,
                                                                                                             X_factor)
        print(f'******************************************')
        print(f'************* On testing set ************* ')
        print(f'******************************************')
        print(f'Total R2: {"{:.1f}".format(r2_total * 100)}%')
        print(f'Total accuracy of returns directions: {"{:.2f}".format(acc_direction * 100)}%')
        print(f'Total loss: {"{:.5f}".format(test_loss)}')
        print(
            f'For the first strategy the performance is {"{:.1f}".format(perf_1 * 100)}% and the sharpe ratio is {"{:.1f}".format(sharpe_1)}')
        print(
            f'For the second strategy the performance is {"{:.1f}".format(perf_2 * 100)}% and the sharpe ratio is {"{:.1f}".format(sharpe_2)}')
        return r2_total, perf_1, sharpe_1, perf_2, sharpe_2


@dataclass
class Analysis:
    """
    This class allows to get the wanted attributes from the analysis of the tune.run method
    Indeed saving (with a pickle) directly the analysis creates error when using other laptop with different paths
    """
    trial_dataframes: dict
    results: dict


class MultiCAModel:
    """
    This class hyper-optimize the training and compute steps of testing of each model defined
    The aim is to obtain results of test sample metrics (R2, sharpe ratio) as in the paper
    """

    def __init__(
            self,
            factor: range = range(1, 7),
            max_epochs: int = 1000,
            trials_hopt: int = 50,  # Number of Trials of hyper optimization
            tolerance_es: int = 10,  # Tolerance of early stopping i.e early stopping is at epoch 10 minimum
            objective_func: str = 'loss'  # Could be also R2, accuracy of direction, a strat perf or a strat sharpe
    ):
        self.factor = factor
        self.max_epochs = max_epochs
        self.trials_hopt = trials_hopt
        self.tolerance_es = tolerance_es
        self.objective_func = objective_func

    @property
    def models(self):
        """
        Defining the conditional autoencoders (CA) as in the paper and as above
        """
        return ['CA0', 'CA1', 'CA2', 'CA3']

    @property
    def search_space(self):
        """
        Defining the search space of the hyper-parameters
        """
        return {
            "lr": tune.loguniform(1e-4, 1e-1),
            "l1": tune.loguniform(1e-1, 9e-1),
            "rate_dropout": tune.loguniform(1e-1, 9e-1)
        }

    @property
    def mode(self):
        return 'max' if self.objective_func != 'loss' else 'min'

    @property
    def scheduler(self):
        """
        ASHAScheduler terminate bad performing trials early
        Uses a metric as the training result objective value attribute
        Uses a mode as whether objective is minimizing or maximizing the metric attribute
        """
        return ASHAScheduler(metric=self.objective_func, mode=self.mode, max_t=self.max_epochs, grace_period=1,
                             reduction_factor=2)

    @property
    def reporter(self):
        """
        Reporter of the hyper-optimization results on the validation sample
        """
        return CLIReporter(
            metric_columns=["loss", "R2", "training_iteration", 'perf_1', 'sharpe_1', 'perf_2', 'sharpe_2'])

    def compute(self, mode: str = None, metric: str = None):
        """
        This main method allows to compute several steps in order to reimplement the paper
        For each model defined:
        _Run an hyper-optimization and then test the model with the config obtained
        _Save the results of the metrics on the testing sample
        """
        if mode is None: mode = self.mode
        if metric is None: metric = self.objective_func
        r2_results = pd.DataFrame(index=self.models, columns=self.factor)
        perf_1_results = pd.DataFrame(index=self.models, columns=self.factor)
        sharpe_1_results = pd.DataFrame(index=self.models, columns=self.factor)
        perf_2_results = pd.DataFrame(index=self.models, columns=self.factor)
        sharpe_2_results = pd.DataFrame(index=self.models, columns=self.factor)
        dict_results = {'R2': r2_results, 'Perf_1': perf_1_results, 'Sharpe_1': sharpe_1_results,
                        'Perf_2': perf_2_results, 'Sharpe_2': sharpe_2_results}
        for name_model in self.models:
            for nb_factor in self.factor:
                key = f'analysis_{name_model}_{nb_factor}_{mode}_{metric}'
                print('-----------------------------------------------------------------------')
                print('-----------------------------------------------------------------------')
                print(f'Hyperoptimization of {name_model} with {nb_factor} factor(s)')
                print('-----------------------------------------------------------------------')
                print('-----------------------------------------------------------------------')
                model = ModelCA(name_model, nb_factor, self.max_epochs, self.tolerance_es)
                """
                Training and Validation
                Running the package Ray Tune in order to iteratively search for hyperparameters that optimize 
                the validation objective (Sum of MSE / (Nb_stocks * Nb_days))
                """
                try:
                    with open(f'data/{key}.pkl', 'rb') as file:
                        analysis = pickle.load(file)
                except FileNotFoundError:
                    analysis = tune.run(
                        tune.with_parameters(model.train, hyperopt=True),
                        num_samples=self.trials_hopt,
                        config=self.search_space,
                        scheduler=self.scheduler,
                        progress_reporter=self.reporter,
                        name=key
                    )
                    my_analysis = Analysis(analysis.trial_dataframes, analysis.results)
                    with open(f'data/{key}.pkl', 'wb') as file:
                        pickle.dump(my_analysis, file)
                """
                Testing
                """
                r2_total, perf_1, sharpe_1, perf_2, sharpe_2 = model.test(analysis, metric, mode)
                list_result = [r2_total, perf_1, sharpe_1, perf_2, sharpe_2]
                """
                Saving the metrics of the test sample
                """
                for name_results, results, result in zip(dict_results.keys(), dict_results.values(), list_result):
                    results[nb_factor].loc[name_model] = result
        print('-----------------------------------------------------------------------')
        print('-----------------------------------------------------------------------')
        print(f'               Results of metrics on the test sample                  ')
        print('-----------------------------------------------------------------------')
        for name, results in dict_results.items():
            print('-----------------------------------------------------------------------')
            print(name)
            print(results)
            results.to_pickle(f'data/{name}.pkl')


def features_importance():
    """
     Analysis the importance of each characteristics (features)
     As in the paper for this analysis, we focus on the five-factor specification of each model
     Loading the model saved during the training and validation (metric='loss' and mode='min')
     And testing with each asset characteristic set to 0
     Search the R2 contribution of each asset characteristic:
     For each model (5 factors specifications):
     1 - (Total R2 of the model with this characteristic set to 0 / Total R2 of the model with all the characteristics)
     And then making the mean of each model result
    """
    try:
        r2_contribution = pd.read_pickle('data/R2_contribution.pkl')
    except FileNotFoundError:
        models, fields = ['CA0', 'CA1', 'CA2', 'CA3'], ModelData().load_args().fields
        factor, metric, mode = 5, 'loss', 'min'
        true_r2_results, r2_results = pd.read_pickle('R2.pkl'), pd.DataFrame(index=models, columns=fields)
        r2_contribution = r2_results.copy()
        for name_model in models:
            key = f'analysis_{name_model}_{factor}_{mode}_{metric}'
            model = ModelCA(name_model, factor)
            for idx_fields in range(len(fields)):
                with open(f'{key}.pkl', 'rb') as file:
                    analysis = pickle.load(file)
                r2_total, _, _, _, _ = model.test(analysis, metric, mode, idx_fields)
                r2_results[fields[idx_fields]].loc[name_model] = r2_total
            r2_contribution.loc[name_model] = 1 - r2_results.loc[name_model] / true_r2_results[factor].loc[name_model]
        r2_contribution.to_pickle('data/R2_contribution.pkl')
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('   Top 20 characteristics sorted by total R2 contribution importance:  ')
    print('-----------------------------------------------------------------------')
    for model in ['CA0', 'CA1', 'CA2', 'CA3']:
        print('-----------------------------------------------------------------------')
        print(f'{model} Model: ')
        top_20 = r2_contribution.loc[model].sort_values(ascending=False)[:20]
        print(top_20)


if __name__ == '__main__':
    """
    ********************************************************************************************************************
                                          Autoencoder asset pricing models
                                                                
                                                                                 Shihao Gu a, Bryan Kelly b, Dacheng Xiu
    ********************************************************************************************************************
    3. An empirical study of US equity
    Steps of the paper's implementations:
    3.1. Data
    3.2. Models comparison set
    3.3. Statistical performance evaluation
    3.5. Risk premia vs. mispricing                 /// not done \\\
    3.4. Economic performance evaluation
    3.7. Robustness check                           /// not done \\\
    3.6. Characteristics importance
    ********************************************************************************************************************
    """

    """
    #Instantiation and computation of the data already done
    ModelData(BaseData()).compute()
    #To debug a model
    model = ModelCA(model_type='CA2', K_factor=2, max_epochs=50, tolerance_es=10)
    model.train(config={'l1':0.5,'lr':0.001,'rate_dropout':0}, hyperopt=False)
    model.test()
    """

    MultiCAModel().compute()
    features_importance()
