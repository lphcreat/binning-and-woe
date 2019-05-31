from sklearn.base import BaseEstimator, TransformerMixin
from utils import woe_transform

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

    def __init__(self,label):
        self.label=label

    def fit(self, df):
        """
        df : data only dataframe type
        """
        self.df=df
        self.woe_dict=woe_transform(df,self.label)
        return self

    def transform(self, X=None,self_woedict=None):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : dataframe, if you not input it will use fit data, 
            the data not contain label column
        self_woedict: the woe dict by this model fit and save to the file
        Returns
        -------
        df : type dataframe,woe data
        """
        df= X if X!=None else self.df
        woe_dict= self_woedict if self_woedict !=None else self.woe_dict
        cols=list(filter(lambda item:item not in [self.label,'num'],self.df.columns))
        for attr in cols:
            df[attr] = df[attr].map(woe_dict[attr])
        df.drop(['num'],axis=1,inplace=True)
        return df
        

if __name__ == "__main__":
    #加载数据
    from sklearn.datasets import load_iris
    import pandas as pd
    from binning.sklearn_bin import NumtoCategorical as nc
    iris = load_iris()
    df=pd.concat([pd.DataFrame(iris.data),pd.DataFrame(iris.target)],ignore_index=True,axis=1)
    df.columns=iris.feature_names+['target']
    df=df[df['target'].isin([1,2])]
    #分割数据
    Sp=nc(bins_num=5)
    clf=Sp.fit(df,'target',split_func='chi')
    dff=clf.transform()

    Cw=CattoWoe('target')
    wclf=Cw.fit(dff)
    wdf=wclf.transform()
    print(wdf.head())

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    cols=list(filter(lambda item:item !='target',wdf.columns))
    X,x_test,Y,y_test=train_test_split(wdf[cols],wdf['target'],test_size=0.33,shuffle=True)
    clf = LogisticRegression()
    clf.fit(X, Y)
    score_test = classification_report(y_test, clf.predict(x_test))
    print(score_test)
    X,x_test,Y,y_test=train_test_split(df[cols],df['target'],test_size=0.33,shuffle=True)
    clf = LogisticRegression()
    clf.fit(X, Y)
    score_test = classification_report(y_test, clf.predict(x_test))
    print(score_test)





