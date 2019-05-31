from sklearn.base import BaseEstimator, TransformerMixin
from utils import get_interval

class NumtoCategorical(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    bins_num : number type, the bins num
    self_thres : dict type, you can input your split dict. example {'col1':[[0,2],[2,5]]}
    Attributes
    ----------
    threshold_list : dict of intervals,example {'col1':[[0,2],[2,5]]}
    Examples
    --------
    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df=pd.concat([pd.DataFrame(iris.data),pd.DataFrame(iris.target)],ignore_index=True,axis=1)
    df.columns=iris.feature_names+['target']
    #split data
    Sp=NumtoCategorical(bins_num=5)
    clf=Sp.fit(df,'target',split_func='tree')
    dff=clf.transform()
    dff=pd.concat([dff,df],axis=1)
    """

    def __init__(self,bins_num=15,self_thres=None):
        self.bins_num = bins_num
        self.self_thres=self_thres   

    def fit(self, df_all, label,split_func):
        """
        df : data only contain num and label columns,cant contain categeory columns
        label : the label column name
        split_func : the split func you can select from ['tree','chi']
        """
        cols=list(filter(lambda item:item !=label,df_all.columns))
        if label==None:
            # import warnings
            # warnings.warn("only split num features,can not calculate woe",Warning)
            raise ValueError("you need confirm input label column name, got error")
        
        #spilt num
        self.threshold_list=get_interval(df_all,label,split_func,bins_num=self.bins_num,
            self_thres=self.self_thres)
        self.df=df_all
        self.cols=cols
        return self

    def transform(self, X=None):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : dataframe, if you not input it will use fit data, 
            the data not contain label column
        Returns
        -------
        df : type dataframe,split data
        """
        threshold_list=self.threshold_list
        if X==None:
            df=self.df
        else:
            df=X
        assert len(df.columns)-1==len(threshold_list.keys())
        def split(x,col):
            for index,item in enumerate(threshold_list[col]):
                if item[0] <= x < item[1]:
                    return col+'_'+str(index+1)
                elif x<threshold_list[col][0][0]:
                    return col+'_0'
                elif x>=threshold_list[col][-1][1]:
                    return col+'_'+str(len(threshold_list[col]))
        for col in self.cols:
            df.loc[:, col] = df.loc[:, col].map(lambda x:split(x,col)) 
        return df
        

if __name__ == "__main__":
    #加载数据
    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df=pd.concat([pd.DataFrame(iris.data),pd.DataFrame(iris.target)],ignore_index=True,axis=1)
    df.columns=iris.feature_names+['target']
    #分割数据
    Sp=NumtoCategorical(bins_num=5)
    clf=Sp.fit(df,'target',split_func='tree')
    dff=clf.transform()
    dff=pd.concat([dff,df],axis=1)
    print(dff.head())
