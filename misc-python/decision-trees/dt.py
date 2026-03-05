# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

# %%
data = pd.read_csv("titanic.csv")


# %%
data.head(n=10)


# %%
data.info()


# %%
columns_to_drop = ["PassengerId","Name","Ticket","Cabin","Embarked"]


# %%
data_clean = data.drop(columns_to_drop,axis=1)


# %%
data_clean.head(n=5)

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_clean["Sex"] = le.fit_transform(data_clean["Sex"])

# %%
data_clean.head()

# %%
data_clean.info()

# %%
data_clean = data_clean.fillna(data_clean["Age"].mean())

# %%
data_clean.loc[1]

# %%
input_cols = ['Pclass',"Sex","Age","SibSp","Parch","Fare"]
output_cols = ["Survived"]

X = data_clean[input_cols]
Y = data_clean[output_cols]

print(X.shape,Y.shape)
print(type(X))


# %%
def entropy(col):
    
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    
    ent = 0.0
    
    for ix in counts[1]:
        p  = ix/N
        ent += (-1.0*p*np.log2(p))
    
    return ent
    


# %%
def divide_data(x_data, fkey, fval):
    x_left = x_data[x_data[fkey] <= fval]
    x_right = x_data[x_data[fkey] > fval]
    return x_left, x_right



# %%
def information_gain(x_data,fkey,fval):
    
    left,right = divide_data(x_data,fkey,fval)
    
    #% of total samples are on left and right
    l = float(left.shape[0])/x_data.shape[0]
    r = float(right.shape[0])/x_data.shape[0]
    
    #All examples come to one side!
    if left.shape[0] == 0 or right.shape[0] ==0:
        return -1000000 #Min Information Gain
    
    i_gain = entropy(x_data.Survived) - (l*entropy(left.Survived)+r*entropy(right.Survived))
    return i_gain


# %%
for fx in X.columns:
    print(fx)
    print(information_gain(data_clean,fx,data_clean[fx].mean()))


# %%
class DecisionTree:
    
    #Constructor
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
        
    def train(self,X_train):
        
        features = ['Pclass','Sex','Age','SibSp', 'Parch', 'Fare']
        info_gains = []
        
        for ix in features:
            i_gain = information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
            
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        print("Making Tree Features is",self.fkey)
        
        #Split Data
        data_left,data_right = divide_data(X_train,self.fkey,self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
         
        #Truly a left node
        if data_left.shape[0]  == 0 or data_right.shape[0] ==0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        #Stop earyly when depth >=max depth
        if(self.depth>=self.max_depth):
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        
        #Recursive Case
        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left)
        
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right)
        
        #You can set the target at every node
        if X_train.Survived.mean() >= 0.5:
            self.target = "Survive"
        else:
            self.target = "Dead"
        return
    
    def predict(self,test):
        if test[self.fkey]>self.fval:
            #go to right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


# %%
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
test_data = test_data.reset_index(drop=True)

# %%
print(train_data.shape,test_data.shape)


# %%
dt = DecisionTree()


# %%
dt.train(train_data)


# %%
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
test_data = test_data.reset_index(drop=True)

# %%
print(train_data.shape,test_data.shape)

# %%
dt = DecisionTree()

# %%
dt.train(train_data)

# %%
print(dt.fkey)
print(dt.fval)
print(dt.left.fkey)
print(dt.right.fkey)

# %%
y_pred = []
for ix in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[ix]))

# %%
y_pred

# %%
y_actual = test_data[output_cols]

# %%
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)

# %%
print(y_pred)

# %%
y_pred = np.array(y_pred).reshape((-1,1))
print(y_pred.shape)

# %%
acc = np.sum(y_pred==y_actual)/y_pred.shape[0]

# %%
acc = np.sum(np.array(y_pred)==np.array(y_actual))/y_pred.shape[0]

# %%
print(acc)

# %% [markdown]
# ## Using SKLearn

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
sk_tree = DecisionTreeClassifier(criterion='gini',max_depth=5)

# %%
sk_tree.fit(train_data[input_cols],train_data[output_cols])

# %%
sk_tree.predict(test_data[input_cols])

# %%
sk_tree.score(test_data[input_cols],test_data[output_cols])

# %%
import pydotplus

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

# %%
dot_data = StringIO()
export_graphviz(sk_tree,out_file=dot_data,filled=True,rounded=True)

# %%
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# %%
X_train = train_data[input_cols]
Y_train = np.array(train_data[output_cols]).reshape((-1,))
X_test = test_data[input_cols]
Y_test = np.array(test_data[output_cols]).reshape((-1,))

# %%
sk_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
sk_tree.fit(X_train,Y_train)
sk_tree.score(X_train,Y_train)

# %%
sk_tree.score(X_test,Y_test)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5)

# %%
rf.fit(X_train,Y_train)

# %%
rf.score(X_train,Y_train)

# %%
rf.score(X_test,Y_test)

# %%
from sklearn.model_selection import cross_val_score
acc = cross_val_score(RandomForestClassifier(n_estimators=40,max_depth=5,criterion='entropy'),X_train,Y_train,cv=5).mean()

# %%
print(acc)

# %%
acc_list = []
for i in range(1,50):
    acc = cross_val_score(RandomForestClassifier(n_estimators=i,max_depth=5),X_train,Y_train,cv=5).mean()
    acc_list.append(acc)

# %%
print(acc_list)

# %%
import matplotlib.pyplot as plt
plt.plot(acc_list)
plt.show()

# %%
print(np.argmax(acc_list))

# %%
rf = RandomForestClassifier(n_estimators=22,max_depth=5,criterion='entropy')
rf.fit(X_train,Y_train)

# %%
rf.score(X_train,Y_train)

# %%
rf.score(X_test,Y_test)
