from sklearn.base import BaseEstimator, TransformerMixin
from .binning.utils import woe_transform

class CattoWoe(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    label : the label column name
    Attributes
    ----------
    woe_dict : dict of intervals,example {'col1':{'xx':0.235}}
    Examples
    --------
    please refer to the readme example
    """

    def __init__(self,label,self_woedict=None):
        self.label=label
        self.self_woedict=self_woedict

    def fit(self, df):
        """
        df : data only dataframe type
        """
        self.df=df
        self.woe_dict=woe_transform(df,self.label)
        return self

    # @classmethod
    def transform(self, X=None):
        """Transform X using woe encoding.
        Parameters
        ----------
        X : dataframe, if you not input it will use fit data, 
            the data not contain label column
        self_woedict: the woe dict by this model fit and save to the file
        Returns
        -------
        df : type dataframe,woe data
        """
        df= X if X is not None else self.df
        woe_dict= self.self_woedict if self.self_woedict !=None else self.woe_dict
        cols=filter(lambda item:item not in [self.label,'num'],df.columns)
        for attr in cols:
            df[attr] = df[attr].map(woe_dict[attr])
        if X is None:
            df.drop(['num'],axis=1,inplace=True)
        return df





