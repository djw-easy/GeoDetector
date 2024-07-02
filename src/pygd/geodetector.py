import warnings
import numpy as np
import pandas as pd
from typing import Sequence
from scipy.stats import f, ncf
import matplotlib.pyplot as plt


from pathlib import Path
def load_example_data():
    file_path = Path(__file__).parent / "example_data" / "disease.csv"
    df = pd.read_csv(file_path)
    return df


def _plot_value(ax, interaction_df, ecological_df, value_fontsize=10):
    length = len(interaction_df.index)
    for i in range(length):
        for j in range(length):
            if not pd.isna(interaction_df.iloc[i, j]):
                num = str(round(interaction_df.iloc[i, j], 2))
                mark = num[-2:] if 3 == len(num) else num[-3:]
                if 'Y'==ecological_df.iloc[i, j]:
                    ax.text(j, i, mark, ha="center", va="center", color="r", fontsize=value_fontsize)
                else:
                    ax.text(j, i, mark, ha="center", va="center", color="k", fontsize=value_fontsize)


class GeoDetector(object):
    def __init__(self, df: pd.DataFrame, y: str, factors: Sequence[str]):
        self.df = df
        self.y = y
        self.factors = factors
        self._check_data(df, y, factors)
        self.interaction_df, self.ecological_df = None, None

    def _check_data(self, df, y, factors):
        for factor in factors:
            if not factor in df.columns:
                raise ValueError('Factor [{}] is not in data')
            
        for factor in factors:
            # 检查列的数据类型
            if df[factor].dtype not in ['int64', 'int32', 'int16', 'int8', 
                                        'uint64', 'uint32', 'uint16', 'uint8', 
                                        'object', 'string']:
                # 如果数据类型不是整型或字符型，发出警告
                warnings.warn(f"Factor '{factor}' is not of type 'int' or 'str'.")
        
        if y not in df.columns:
            raise ValueError('Factor [{}] is not in data')
            
        for factor in factors:
            if y==factor:
                raise ValueError("Y variable should not in Factor variables. ")
        
        has_null = df.isnull().values.any()
        if has_null:
            raise ValueError("data hava some objects with value NULL")

    @classmethod
    def _cal_ssw(self, df: pd.DataFrame, y, factor, extra_factor=None):
        def cal_ssw(df: pd.DataFrame, y):
            length = df.shape[0]
            if length==1:
                strataVar = 0
                lamda_1st = np.square(df[y].values[0])
                lamda_2nd = df[y].values[0]
            else:
                strataVar = (length-1) * df[y].var(ddof=1)

                lamda_1st = np.square(df[y].values.mean())
                lamda_2nd = np.sqrt(length) * df[y].values.mean()
            return strataVar, lamda_1st, lamda_2nd
        if extra_factor==None:
            df2 = df[[y, factor]].groupby(factor).apply(cal_ssw, y=y)
        else:
            df2 = df[[y]+list(set([factor, extra_factor]))].groupby([factor, extra_factor]).apply(cal_ssw, y=y)
        df2 = df2.apply(pd.Series)
        df2 = df2.sum()
        strataVarSum, lamda_1st_sum, lamda_2nd_sum = df2.values
        return strataVarSum, lamda_1st_sum, lamda_2nd_sum

    @classmethod
    def _cal_q(self, df, y, factor, extra_factor=None):
        strataVarSum, lamda_1st_sum, lamda_2nd_sum = self._cal_ssw(df, y, factor, extra_factor)
        TotalVar = (df.shape[0]-1)*df[y].var(ddof=1)
        q = 1 - strataVarSum/TotalVar
        return q, lamda_1st_sum, lamda_2nd_sum

    def factor_dector(self):
        out_df = pd.DataFrame(index=["q statistic", "p value"], columns=self.factors, dtype="float32")
        N_var = self.df[self.y].var(ddof=1)
        N_popu = self.df.shape[0]
        for factor in self.factors:
            N_stra = self.df[factor].unique().shape[0]
            q, lamda_1st_sum, lamda_2nd_sum = self._cal_q(self.df, self.y, factor)

            #lamda value
            lamda = (lamda_1st_sum - np.square(lamda_2nd_sum) / N_popu) / N_var
            # F value
            F_value = (N_popu - N_stra)* q / ((N_stra - 1)* (1 - q))
            #p value
            p_value = ncf.sf(F_value, N_stra - 1, N_popu - N_stra, nc=lamda)

            out_df.loc["q statistic", factor] = q
            out_df.loc["p value", factor] = p_value
        return out_df
    
    @classmethod
    def _interaction_relationship(self, df):
        out_df = pd.DataFrame(index=df.index, columns=df.columns)
        length = len(df.index)
        for i in range(length):
            for j in range(i+1, length):
                factor1, factor2 = df.index[i], df.index[j]
                i_q = df.loc[factor2, factor1]
                q1 = df.loc[factor1, factor1]
                q2 = df.loc[factor2, factor2]

                if (i_q <= q1 and i_q <= q2):
                    outputRls = "Weaken, nonlinear"
                if (i_q < max(q1, q2) and i_q > min(q1, q2)):
                    outputRls = "Weaken, uni-"
                if (i_q == (q1 + q2)):
                    outputRls = "Independent"
                if (i_q > max(q1, q2)):
                    outputRls = "Enhance, bi-"
                if (i_q > (q1 + q2)):
                    outputRls = "Enhance, nonlinear"

                out_df.loc[factor2, factor1] = outputRls
        return out_df

    def interaction_detector(self, relationship=False):
        self.interaction_df = pd.DataFrame(index=self.factors, columns=self.factors, dtype="float32")
        length = len(self.factors)
        for i in range(0, length):
            for j in range(0, i+1):
                q, _, _ = self._cal_q(self.df, self.y, self.factors[i], self.factors[j])
                self.interaction_df.loc[self.factors[i], self.factors[j]] = q

        if relationship:
            self.interaction_relationship_df = self._interaction_relationship(self.interaction_df)
            return self.interaction_df, self.interaction_relationship_df
        return self.interaction_df
    
    def ecological_detector(self):
        self.ecological_df = pd.DataFrame(index=self.factors, columns=self.factors, dtype="float32")
        length = len(self.factors)
        for i in range(1, length):
            ssw1, _, _ = self._cal_ssw(self.df, self.y, self.factors[i])
            dfn = self.df[self.factors[i]].notna().sum()-1
            for j in range(0, i):
                ssw2, _, _ = self._cal_ssw(self.df, self.y, self.factors[j])
                dfd = self.df[self.factors[j]].notna().sum()-1
                fval = (dfn*(dfd-1)*ssw1)/(dfd*(dfn-1)*ssw2)
                if fval<f.ppf(0.05, dfn, dfn):
                    self.ecological_df.loc[self.factors[i], self.factors[j]] = 'Y'
                else:
                    self.ecological_df.loc[self.factors[i], self.factors[j]] = 'N'
        return self.ecological_df

    def plot(self, tick_fontsize=10, value_fontsize=10, colorbar_fontsize=10, show=True):
        if isinstance(self.interaction_df, type(None)):
            self.interaction_detector()
        if isinstance(self.ecological_df, type(None)):
            self.ecological_detector()

        fig, ax = plt.subplots(constrained_layout=True)

        im = ax.imshow(self.interaction_df.values, cmap="YlGnBu", vmin=0, vmax=1)
        _plot_value(ax, self.interaction_df, self.ecological_df, value_fontsize=value_fontsize)

        ax.set_xticks(np.arange(len(self.factors)))
        ax.set_yticks(np.arange(len(self.factors)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticklabels(self.factors, fontsize=tick_fontsize)
        ax.set_yticklabels(self.factors, rotation=45, fontsize=tick_fontsize)
        ax.tick_params(axis='y', pad=0.1)

        colorbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, aspect=25, extend="both")
        colorbar.ax.tick_params(labelsize=colorbar_fontsize)

        if show:
            plt.show()
            return ax
        else:
            return ax