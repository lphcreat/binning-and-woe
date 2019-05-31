# binning-and-woe
this module is to preprocess data by segmentation and WOE transform, and the style is sklearn-style.
if you will use/reference it,please click star ^_^.

Example code

from sklearn.datasets import load_iris
import pandas as pd
from sklearn_bin import NumtoCategorical as nc
from sklearn_woe import CattoWoe
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
