from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
import pandas as pd
def get_interval(df,label,split_func,bins_num=None,self_thres=None):
    """
    df : the need process dataframe data
    label : the column name of label data
    split_func : the method of getting threshold list
    bin_num : author specify num of interval
    self_thres : if you select method not in [chi,tree] you should specif your threshold list by dict
    """
    cols=list(filter(lambda item:item !=label,df.columns))
    y=df[label]
    if split_func=='chi':
        threshold_list=[chi_merge(df,item,y,label,bins_num=bins_num) for item in cols]
        return dict(zip(cols,threshold_list))
    elif split_func=='tree':
        threshold_list=[dtree_threshold(df[item],y,bins_num=bins_num) for item in cols]
        return dict(zip(cols,threshold_list))
    else:
        if isinstance(self_thres,dict):
            return self_thres
        else:
            raise ValueError("you need input yourself threshold_list")

def woe_transform(df,label):
    #目前只能处理两类问题，对于多类的可以考虑计算WOE后乘以类别的占比，相当于加入先验概率。
    labels=df[label].unique()
    label_one=labels[0]
    label_two=labels[1]
    df['num']=df.index
    def woe_(attr):
        pt = pd.pivot_table(df, index=label,columns=attr, values='num', aggfunc='count').T
        pt['WOEi'] = np.log((pt[label_one] / pt[label_one].sum()) /
                        (pt[label_two] / pt[label_two].sum())).round(4)
        # pt['IVi'] = pt.WOEi.mul((pt[label_one] / pt[label_one].sum()) -
        #                 (pt[label_two] / pt[label_two].sum())).round(3)
        # iv = pt.IVi.sum()
        pt = pt.fillna(0)
        key = pt.index.tolist()
        value = pt.WOEi.tolist()
        dict_v = dict(zip(key, value))
        return dict_v
    cols=list(filter(lambda item:item not in [label,'num'],df.columns))
    woe_list=[woe_(item) for item in cols]
    df.drop(['num'],axis=1)
    return dict(zip(cols,woe_list))

def dtree_threshold(X,y,bins_num=None):
    clf = DecisionTreeClassifier(max_leaf_nodes=bins_num)
    X=np.array(X).reshape(-1,1)
    clf.fit(X,y)
    interval=list(clf.tree_.threshold[clf.tree_.feature == 0])
    interval.append(X.min())
    interval.append(X.max())
    interval=sorted(interval)
    intervals=[[interval[i], interval[i+1]] for i in range(len(interval)-1)]
    new_intervals=check_length_interval(X,intervals)
    return new_intervals

def check_length_interval(X,intervals):
    #default percent is 8%
    threshold_num=X.shape[0]*0.08
    new_intervals=[]
    big_set=set([X.min()])
    for index in range(len(intervals)):
        count_interval= len(np.where(np.logical_and(X>=intervals[index][0], X<intervals[index][1]))[0])
        if count_interval<threshold_num: # Merge the intervals
            if index==len(intervals)-1:
                t = intervals[index-1] + intervals[index]
            else:
                t = intervals[index] + intervals[index+1]
            append_item=[min(t), max(t)]
        else:
            append_item=intervals[index]
        if min(append_item)>=max(big_set):
            big_set.add(max(append_item))
            new_intervals.append(append_item)
    return new_intervals


def chi_merge(data,attr,y,label,bins_num=15):
    distinct_vals = sorted(set(data[attr])) # Sort the distinct values
    labels = sorted(set(y)) # Get all possible labels
    empty_count = {l: 0 for l in labels} # A helper function for padding the Counter()
    intervals = [[distinct_vals[i], distinct_vals[i+1]] for i in range(len(distinct_vals)-1)] # Initialize the intervals for each attribute
    while len(intervals) > bins_num: # While loop
        chi = []
        for i in range(len(intervals)-1):
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i+1][0], intervals[i+1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
            count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
            count_total = count_0 + count_1
            expected_0 = count_total*sum(count_0)/total
            expected_1 = count_total*sum(count_1)/total
            chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
            chi_ = np.nan_to_num(chi_) # Deal with the zero counts
            chi.append(sum(chi_)) # Finally do the summation for Chi2
        sort_chi=sorted(enumerate(chi),key=lambda x:x[1],reverse=True)
        step=len(intervals)-bins_num
        min_chi=sort_chi[-step:] # Find the minimal Chi2 for current iteration
        min_chi_index=[item[0] for item in min_chi]
        new_intervals = [] # Prepare for the merged new data array
        big_set=set([min(distinct_vals)])
        for index in range(len(intervals)):
            #check eve interval num
            if index in min_chi_index: # Merge the intervals
                t = intervals[index] + intervals[index+1]
                append_item=[min(t), max(t)]
            else:
                append_item=intervals[index]
            if min(append_item)>=max(big_set):
                big_set.add(max(append_item))
                new_intervals.append(append_item)
        intervals=new_intervals
    intervals=check_length_interval(np.array(data[attr]),intervals)
    return intervals