import pandas as pd
from scipy.stats import f

def factor_dector(df, y: str, factors: list, out_index_name :str ="q-value"):
    out_df = pd.DataFrame(index=[out_index_name], columns=factors, dtype="float32")
    for factor in factors:
        ssw = df[y, factor].groupby(factor).apply(lambda x:x.shape[0]*x.var(ddof=0))[y].sum()
        sst = (df[y].size*df[y].var(ddof=0))
        q = 1-ssw/sst
        out_df.loc[out_index_name, factor] = q
    return out_df

def interaction_detector(df, y, factors):
    out_df = pd.DataFrame(index=factors, columns=factors, dtype="float32")
    length = len(factors)
    for i in range(0, length):
        for j in range(0, i+1):
            ssw = df[y, factors[i], factors[j]].groupby([factors[i], factors[j]]).apply(lambda x:x.shape[0]*x.var(ddof=0))[y].sum()
            sst = (df[y].size*df[y].var(ddof=0))
            q = 1-ssw/sst
            out_df.loc[factors[i], factors[j]] = q
    return out_df

def ecological_detector(df, y, factors):
    out_df = pd.DataFrame(index=factors, columns=factors, dtype="float32")
    length = len(factors)
    for i in range(1, length):
        ssw1 = df[y, factors[j]].groupby(factors[i]).apply(lambda x:x.shape[0]*x.var(ddof=0))[y].sum()
        dfn = df[factors[i]].notna().sum()-1
        for j in range(0, i):
            ssw2 = df[y, factors[j]].groupby(factors[j]).apply(lambda x:x.shape[0]*x.var(ddof=0))[y].sum()
            dfd = df[factors[j]].notna().sum()-1
            fval = (dfn*(dfd-1)*ssw1)/(dfd*(dfn-1)*ssw2)
            if fval<f.ppf(0.05, dfn, dfn):
                out_df.loc[factors[i], factors[j]] = 'Y'
            else:
                out_df.loc[factors[i], factors[j]] = 'N'
    return out_df