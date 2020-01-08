if __name__ == "__main__":
    #分割数据
    #加载数据
    from sklearn.datasets import load_iris
    import pandas as pd
    from binning_woe.binning.sklearn_bin import NumtoCategorical as nc
    from binning_woe.sklearn_woe import CattoWoe
    iris = load_iris()
    df=pd.concat([pd.DataFrame(iris.data),pd.DataFrame(iris.target)],ignore_index=True,axis=1)
    df.columns=iris.feature_names+['target']
    df=df[df['target'].isin([1,2])]
    #分割数据
    Sp=nc(bins_num=3,num_cols=iris.feature_names)
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