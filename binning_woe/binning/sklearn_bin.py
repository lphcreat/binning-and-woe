from sklearn.base import BaseEstimator, TransformerMixin
import math
from .utils import get_interval

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

    def __init__(self,bins_num=15,self_thres=None,num_cols=None):
        self.bins_num = bins_num
        self.self_thres=self_thres
        self.num_cols=num_cols   

    def fit(self, df_all, label,split_func):
        """
        df : data only contain num and label columns,cant contain categeory columns
        label : the label column name
        split_func : the split func you can select from ['tree','chi']
        """
        cols=self.num_cols+[label]
        if label==None:
            # import warnings
            # warnings.warn("only split num features,can not calculate woe",Warning)
            raise ValueError("you need confirm input label column name, got error")
        
        #spilt num
        self.threshold_list=get_interval(df_all[cols],label,split_func,bins_num=self.bins_num,
            self_thres=self.self_thres)
        self.df=df_all
        return self

    def transform(self, X=None,cat_style=True):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : dataframe, if you not input it will use fit data, 
            the data not contain label column
        Returns
        -------
        df : type dataframe,split data
        """
        threshold_list= self.self_thres if self.self_thres !=None else self.threshold_list
        if X is not None:
            df=X
        else:
            df=self.df
        df=df.fillna('-99')
        # assert len(self.num_cols)==len(threshold_list.keys())
        if cat_style:
            def split(x,col):
                for _,item in enumerate(threshold_list[col]):
                    if x=='-99':
                        return '_null'
                    elif item[0] <= x < item[1]:
                        return str(item[0])+'_'+str(item[1])
                    #可修改
                    elif x<threshold_list[col][0][0]:
                        return '<'+'first'
                    elif x>=threshold_list[col][-1][1]:
                        return '>='+'last'
        else:
            def split(x,col):
                for index,item in enumerate(threshold_list[col]):
                    if x=='-99':
                        return col+'_null'
                    elif item[0] <= x < item[1]:
                        return col+'_'+str(index+1)
                    elif x<threshold_list[col][0][0]:
                        return col+'_0'
                    elif x>=threshold_list[col][-1][1]:
                        return col+'_'+str(len(threshold_list[col]))
                    
        for col in df.columns:
            if col in self.num_cols:
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
    Sp=NumtoCategorical(num_cols=iris.feature_names,bins_num=5)
    clf=Sp.fit(df,'target',split_func='tree')
    dff=clf.transform()
    dff=pd.concat([dff,df],axis=1)
    print(dff.head())