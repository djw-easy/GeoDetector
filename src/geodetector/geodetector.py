import warnings
import numpy as np
import pandas as pd
from typing import Sequence, Union, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.stats import f, levene, ncf, ttest_ind
from pathlib import Path
from pandas.api.types import is_integer_dtype, is_string_dtype, is_object_dtype


def load_example_data() -> pd.DataFrame:
    """Load example disease dataset."""
    file_path = Path(__file__).parent / "example_data" / "disease.csv"
    df = pd.read_csv(file_path)
    return df


class GeoDetector:
    """
    GeoDetector class for spatial statistics.
    
    References:
        Wang, J. F., Li, X. H., Christakos, G., Liao, Y. L., Zhang, T., Gu, X., & Zheng, X. Y. (2010).
        Geographical detectors-based health risk assessment and its application in the neural tube defects study of the Heshun Region, China.
        International Journal of Geographical Information Science, 24(1), 107-127.
    """
    def __init__(self, df: pd.DataFrame, y: str, factors: Optional[Sequence[str]] = None, alpha: float = 0.05):
        """
        Initialize the GeoDetector instance.

        Args:
            df (pd.DataFrame): The dataset containing both dependent variable and factors.
            y (str): The column name of the dependent variable (numerical).
            factors (Sequence[str], optional): A list of column names for the factors (categorical/stratified).
                                               If None, automatically detects discrete columns as factors.
            alpha (float, optional): The significance level for hypothesis testing. Defaults to 0.05.
        """
        self.df = df
        self.y = y
        self.alpha = alpha
        
        if factors is None:
            # Automatically detect factors: must be discrete (int, str, or object) and not y
            self.factors = []
            for col in df.columns:
                if col == y:
                    continue
                dtype = df[col].dtype
                if is_integer_dtype(dtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
                    self.factors.append(col)
        else:
            self.factors = list(factors)
        
        self._check_data(df, y, self.factors)

    def _is_discrete(self, factor: str) -> bool:
        """Check if a factor is of discrete type."""
        dtype = self.df[factor].dtype
        return is_integer_dtype(dtype) or is_string_dtype(dtype) or is_object_dtype(dtype)

    def _check_discrete_factors(self, factors: Sequence[str]):
        """Ensure all provided factors are discrete."""
        for factor in factors:
            if not self._is_discrete(factor):
                raise ValueError(f"Factor '{factor}' must be a discrete type (int, str, or object). "
                                 f"Current type: {self.df[factor].dtype}. Please discretize it first.")

    def _check_data(self, df: pd.DataFrame, y: str, factors: Sequence[str]):
        """Check data validity."""
        if y not in df.columns:
            raise ValueError(f"Y variable [{y}] is not in data")

        for factor in factors:
            if factor not in df.columns:
                raise ValueError(f"Factor [{factor}] is not in data")
            
            if y == factor:
                raise ValueError(f"Y variable [{y}] should not be in Factor variables.")

        # Check column data types for provided factors
        self._check_discrete_factors(factors)
        
        if df.isnull().values.any():
            raise ValueError("Data contains NULL values")

    @classmethod
    def _cal_ssw(cls, df: pd.DataFrame, y: str, factor: Union[str, list], extra_factor: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Calculate the Within Sum of Squares (SSW) and other statistics for the q-statistic using vectorization.
        """
        group_cols = [factor] if isinstance(factor, str) else list(factor)
        if extra_factor:
            group_cols.append(extra_factor)

        agg_df = df.groupby(group_cols)[y].agg(['var', 'mean', 'count'])
        agg_df['var'] = agg_df['var'].fillna(0)
        
        strataVarSum = ((agg_df['count'] - 1) * agg_df['var']).sum()
        lamda_1st_sum = (agg_df['mean'] ** 2).sum()
        lamda_2nd_sum = (np.sqrt(agg_df['count']) * agg_df['mean']).sum()
        
        return strataVarSum, lamda_1st_sum, lamda_2nd_sum

    @classmethod
    def _cal_q(cls, df: pd.DataFrame, y: str, factor: str, extra_factor: Optional[str] = None) -> Tuple[float, float, float]:
        """Calculate q-statistic."""
        strataVarSum, lamda_1st_sum, lamda_2nd_sum = cls._cal_ssw(df, y, factor, extra_factor)
        total_var = (df.shape[0] - 1) * df[y].var(ddof=1)
        q = 1 - strataVarSum / total_var
        return q, lamda_1st_sum, lamda_2nd_sum

    def factor_detector(self, factors: Optional[Union[str, Sequence[str]]] = None) -> Union[pd.DataFrame, Tuple[float, float]]:
        """
        Factor detector: detects the spatial stratification heterogeneity of Y.
        
        Args:
            factors (str or list, optional): Factors to detect. If None, use all factors.
                                             If a single string, returns (q, p).
        """
        target_factors = factors if factors is not None else self.factors
        if isinstance(target_factors, str):
            target_factors = [target_factors]
        
        # Check if factors are discrete
        self._check_discrete_factors(target_factors)

        res_df = pd.DataFrame(index=["q statistic", "p value"], columns=target_factors, dtype="float64")
        n_popu = self.df.shape[0]
        y_var = self.df[self.y].var(ddof=1)

        for factor in target_factors:
            n_stra = self.df[factor].nunique()
            q, lamda_1st_sum, lamda_2nd_sum = self._cal_q(self.df, self.y, factor)

            nc_param = (lamda_1st_sum - np.square(lamda_2nd_sum) / n_popu) / y_var
            f_val = (n_popu - n_stra) * q / ((n_stra - 1) * (1 - q))
            p_val = ncf.sf(f_val, n_stra - 1, n_popu - n_stra, nc=nc_param)

            res_df.loc["q statistic", factor] = q
            res_df.loc["p value", factor] = p_val
            
        if isinstance(factors, str):
            return res_df.iloc[0, 0], res_df.iloc[1, 0]
        return res_df

    @staticmethod
    def _interaction_relationship(df: pd.DataFrame) -> pd.DataFrame:
        """Determine the type of interaction relationship."""
        out_df = pd.DataFrame(index=df.index, columns=df.columns)
        factors = df.index
        for i, f1 in enumerate(factors):
            for j in range(i + 1, len(factors)):
                f2 = factors[j]
                i_q = df.loc[f2, f1]
                q1 = df.loc[f1, f1]
                q2 = df.loc[f2, f2]

                if i_q <= q1 and i_q <= q2:
                    rel = "Weaken, nonlinear"
                elif q1 < i_q < q2 or q2 < i_q < q1:
                    rel = "Weaken, uni-"
                elif i_q == (q1 + q2):
                    rel = "Independent"
                elif i_q > max(q1, q2):
                    rel = "Enhance, bi-"
                
                if i_q > (q1 + q2):
                    rel = "Enhance, nonlinear"
                
                out_df.loc[f2, f1] = rel
        return out_df

    def interaction_detector(self, factor1: Optional[str] = None, factor2: Optional[str] = None, relationship: bool = False, factors: Optional[Sequence[str]] = None):
        """
        Interaction detector.
        
        Args:
            factor1, factor2 (str, optional): If both provided, returns interaction q for just this pair.
            relationship (bool): If True, returns relationship type.
            factors (Sequence[str], optional): Custom list of factors for full matrix calculation.
        """
        # If any specific factor is provided, BOTH must be provided
        if factor1 or factor2:
            if not (factor1 and factor2):
                raise ValueError("Both factor1 and factor2 must be provided for pairwise interaction detection.")
            
            if factor1 == factor2:
                raise ValueError("factor1 and factor2 must be different for interaction detection.")
            self._check_discrete_factors([factor1, factor2])
            
            q, _, _ = self._cal_q(self.df, self.y, factor1, factor2)
            if not relationship:
                return q
            
            q1, _, _ = self._cal_q(self.df, self.y, factor1)
            q2, _, _ = self._cal_q(self.df, self.y, factor2)
            temp_df = pd.DataFrame({factor1: [q1, q], factor2: [np.nan, q2]}, index=[factor1, factor2])
            rel_df = self._interaction_relationship(temp_df)
            return q, rel_df.loc[factor2, factor1]

        # Full matrix calculation
        target_factors = factors if factors is not None else self.factors
        self._check_discrete_factors(target_factors)
        
        inter_df = pd.DataFrame(index=target_factors, columns=target_factors, dtype="float64")
        for i, f1 in enumerate(target_factors):
            for j in range(i + 1):
                f2 = target_factors[j]
                q, _, _ = self._cal_q(self.df, self.y, f1, f2)
                inter_df.loc[f1, f2] = q

        if relationship:
            return inter_df, self._interaction_relationship(inter_df)
        return inter_df

    def ecological_detector(self, factor1: Optional[str] = None, factor2: Optional[str] = None, factors: Optional[Sequence[str]] = None) -> Union[pd.DataFrame, str]:
        """Ecological detector."""
        # If any specific factor is provided, BOTH must be provided
        if factor1 or factor2:
            if not (factor1 and factor2):
                raise ValueError("Both factor1 and factor2 must be provided for pairwise ecological detection.")
                
            if factor1 == factor2:
                raise ValueError("factor1 and factor2 must be different for ecological detection.")
            self._check_discrete_factors([factor1, factor2])
            
            ssw1, _, _ = self._cal_ssw(self.df, self.y, factor1)
            ssw2, _, _ = self._cal_ssw(self.df, self.y, factor2)
            dfn = self.df[factor1].count() - 1
            dfd = self.df[factor2].count() - 1
            fval = (dfn * (dfd - 1) * ssw1) / (dfd * (dfn - 1) * ssw2)
            return 'Y' if fval < f.ppf(self.alpha, dfn, dfd) else 'N'

        target_factors = factors if factors is not None else self.factors
        self._check_discrete_factors(target_factors)
        
        eco_df = pd.DataFrame(index=target_factors, columns=target_factors, dtype="object")
        for i, f1 in enumerate(target_factors):
            ssw1, _, _ = self._cal_ssw(self.df, self.y, f1)
            dfn = self.df[f1].count() - 1
            for j in range(i):
                f2 = target_factors[j]
                ssw2, _, _ = self._cal_ssw(self.df, self.y, f2)
                dfd = self.df[f2].count() - 1
                fval = (dfn * (dfd - 1) * ssw1) / (dfd * (dfn - 1) * ssw2)
                eco_df.loc[f1, f2] = 'Y' if fval < f.ppf(self.alpha, dfn, dfd) else 'N'
        
        return eco_df

    def risk_detector(self, factor: Optional[str] = None) -> Union[Dict, Dict[str, Dict]]:
        """Risk detector."""
        target_factors = [factor] if factor else self.factors
        self._check_discrete_factors(target_factors)
        
        risk_result = {}
        for f_name in target_factors:
            risk_mean = self.df.groupby(f_name)[self.y].mean()
            strata = np.sort(self.df[f_name].unique())
            t_test_strata = pd.DataFrame(index=strata, columns=strata, dtype=bool)
            
            for i in range(len(strata)):
                for j in range(i + 1, len(strata)):
                    y_i = self.df.loc[self.df[f_name] == strata[i], self.y].values
                    y_j = self.df.loc[self.df[f_name] == strata[j], self.y].values
                    
                    _, p_levene = levene(y_i, y_j)
                    equal_var = p_levene >= self.alpha
                    _, p_ttest = ttest_ind(y_i, y_j, equal_var=equal_var)
                    t_test_strata.loc[strata[j], strata[i]] = p_ttest <= self.alpha

            risk_result[f_name] = {"risk": risk_mean, "ttest_stra": t_test_strata}

        return risk_result[factor] if factor else risk_result

    def _plot_text_labels(self, ax, interaction_df, ecological_df, value_fontsize=10):
        """Internal plotting helper."""
        for i, row_idx in enumerate(interaction_df.index):
            for j, col_idx in enumerate(interaction_df.columns):
                val = interaction_df.iloc[i, j]
                if not pd.isna(val):
                    mark = f"{val:.2f}"
                    # Use ecological_df to determine color if available
                    color = "k"
                    if ecological_df is not None and i < ecological_df.shape[0] and j < ecological_df.shape[1]:
                        if ecological_df.iloc[i, j] == 'Y':
                            color = "r"
                    ax.text(j, i, mark, ha="center", va="center", color=color, fontsize=value_fontsize)

    def plot(self, factors: Optional[Sequence[str]] = None, tick_fontsize=10, value_fontsize=10, colorbar_fontsize=10, show=True):
        """
        Plot interaction and ecological detector results.
        
        Args:
            factors (Sequence[str], optional): Factors to include in the plot. Defaults to self.factors.
        """
        target_factors = factors if factors is not None else self.factors
        self._check_discrete_factors(target_factors)
        
        inter_df = self.interaction_detector(factors=target_factors)
        eco_df = self.ecological_detector(factors=target_factors)

        fig, ax = plt.subplots(constrained_layout=True)
        im = ax.imshow(inter_df.values, cmap="YlGnBu", vmin=0, vmax=1)
        self._plot_text_labels(ax, inter_df, eco_df, value_fontsize=value_fontsize)

        ax.set_xticks(np.arange(len(target_factors)))
        ax.set_yticks(np.arange(len(target_factors)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticklabels(target_factors, fontsize=tick_fontsize)
        ax.set_yticklabels(target_factors, rotation=45, fontsize=tick_fontsize, va='top')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02, aspect=25, extend="both")
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

        if show:
            plt.show()
        return ax
