#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn . tree import DecisionTreeClassifier
from sklearn . ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split , GridSearchCV, KFold, cross_val_score
#Performance metrices
from sklearn . model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve,auc,log_loss,confusion_matrix,classification_report
import warnings
warnings.simplefilter("ignore")
plt.style.use('seaborn')


# In[95]:


data = pd.read_csv (r'C:\Users\iran\Pictures\Screenshots/train.csv')
data


# In[96]:


df = pd.DataFrame(data)
df


# In[97]:


df.describe().T


# In[98]:


df.info()


# In[99]:


df.isnull().sum()


# In[100]:


numerical = ['battery_power', 'clock_speed', 'fc', 'int_memory' ,'m_dep' ,'mobile_wt' ,'n_cores' ,'pc' , 'px_height', 'px_width', 'ram', 'sc_h' , 'sc_w', 'talk_time']


# In[101]:


categorical = ['blue', 'dual_sim', 'four_g', 'three_g' ,'touch_screen' ,'wifi' , 'price_range']


# In[102]:


i = 0
while i<12:
    fig = plt.figure(figsize = [12,3])
    plt.subplot(1,2,1)   #(one row, two plots, first one)
    sns.boxplot(x = 'price_range',y=numerical[i], data=df) # read each one, from Data
    plt.xticks(fontsize=15)
    plt.xlabel('price_range',fontsize=10)
    i += 1
    
    plt.subplot(1,2,2)
    sns.boxplot(x = 'price_range',y=numerical[i], data=df) # row one , second plot
    plt.xticks(fontsize=15)
    plt.xlabel('price_range',fontsize=10)
    i += 1
    plt.show()


# In[103]:


i = 0 
while i < 14 :
    fig = plt . figure(figsize=[15 , 3])
    plt . subplot(1 ,3 ,1)
    sns . boxplot(x =numerical[i] , data =df ,color='plum')
   

    plt . xlabel(numerical[i] )
    i+=1
    if i== 14 :
        break
    plt . subplot(1 , 3 , 2)
    sns . boxplot(x =numerical[i] , data = df ,color='plum' )
    i += 1


# In[104]:


#### RAM , battery_power  are  an important parameter for price_range  ####


# In[105]:


fig=plt.figure(figsize=(25,30))
for i,col in enumerate(categorical):
    ax=fig.add_subplot(4 , 4,i+1)
    sns.scatterplot(x=col, y ='ram' , hue='price_range',data=df)


# In[106]:


fig=plt.figure(figsize=(25,30))
for i,col in enumerate(numerical):
    ax=fig.add_subplot(4 , 4,i+1)
    sns.scatterplot(x=col, y ='ram' , hue='price_range',data=df)


# In[107]:


plt . subplot(2 , 2 , 1)
sns.distplot(df['ram'][df['price_range']==0],hist=True , kde =True , color = 'darkgrey')
plt . subplot(2 , 2 , 2)
sns.distplot(df['ram'][df['price_range']==1],hist=True , kde =True , color = 'dimgray')
plt . subplot(2 , 2 , 3)
sns.distplot(df['ram'][df['price_range']==2],hist=True , kde =True, color = 'dimgrey')
plt . subplot(2 , 2 , 4)
sns.distplot(df['ram'][df['price_range']==3],hist=True , kde =True, color = 'black')

# important parameter for price_range


# In[108]:


plt . subplot(2 , 2 , 1)
sns.distplot(df['battery_power'][df['price_range']==0],hist=True , kde =True  , color = 'aqua')
plt . subplot(2 , 2 , 2)
sns.distplot(df['battery_power'][df['price_range']==1],hist=True , kde =True  , color = 'darkturquoise')
plt . subplot(2 , 2 , 3)
sns.distplot(df['battery_power'][df['price_range']==2],hist=True , kde =True, color = 'teal')
plt . subplot(2 , 2 , 4)
sns.distplot(df['battery_power'][df['price_range']==3],hist=True , kde =True  , color = 'darkslategrey')

# important parameter for price_range


# In[109]:


plt . subplot(2 , 2 , 1)
sns.distplot(df['int_memory'][df['price_range']==0],hist=True , kde =True)
plt . subplot(2 , 2 , 2)
sns.distplot(df['int_memory'][df['price_range']==1],hist=True , kde =True)
plt . subplot(2 , 2 , 3)
sns.distplot(df['int_memory'][df['price_range']==2],hist=True , kde =True, color = 'b')
plt . subplot(2 , 2 , 4)
sns.distplot(df['int_memory'][df['price_range']==3],hist=True , kde =True, color = 'b')


# In[110]:


plt . subplot(2 , 2 , 1)
sns.distplot(df['clock_speed'][df['price_range']==0],hist=True , kde =True)
plt . subplot(2 , 2 , 2)
sns.distplot(df['clock_speed'][df['price_range']==1],hist=True , kde =True)
plt . subplot(2 , 2 , 3)
sns.distplot(df['clock_speed'][df['price_range']==2],hist=True , kde =True, color = 'm')
plt . subplot(2 , 2 , 4)
sns.distplot(df['clock_speed'][df['price_range']==3],hist=True , kde =True, color = 'm')


# In[111]:


a=  df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='battery_power' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='battery_power' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='battery_power' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='battery_power' , y='ram' , hue = 'price_range' ,shade= True, data =d)

# important parameter for price_range


# In[112]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='int_memory' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='int_memory' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='int_memory' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='int_memory' , y='ram' , hue = 'price_range' ,shade= True, data =d)
# important parameter for price_range


# In[113]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='clock_speed' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='clock_speed' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='clock_speed' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='clock_speed' , y='ram' , hue = 'price_range' ,shade= True, data =d)


# In[114]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='sc_h' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='sc_h' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='sc_h' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='sc_h' , y='ram' , hue = 'price_range' ,shade= True, data =d)
# important parameter for price_range


# In[115]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='sc_w' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='sc_w' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='sc_w' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='sc_w' , y='ram' , hue = 'price_range' ,shade= True, data =d)


# In[116]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='n_cores' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='n_cores' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='n_cores' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='n_cores' , y='ram' , hue = 'price_range' ,shade= True, data =d)
# important parameter for price_range


# In[117]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='px_height' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='px_height' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='px_height' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='px_height' , y='ram' , hue = 'price_range' ,shade= True, data =d)


# In[118]:


a = df[df['price_range']==0]
b = df[df['price_range']==1]
c = df[df['price_range']==2]
d = df[df['price_range']==3]


plt . subplot(2 , 2 , 1)
sns . kdeplot(x='px_width' , y='ram' , hue = 'price_range' ,shade= True, data =a)
plt . subplot(2 , 2 , 2)
sns . kdeplot(x='px_width' , y='ram' , hue = 'price_range' ,shade= True, data =b)
plt . subplot(2 , 2 , 3)
sns . kdeplot(x='px_width' , y='ram' , hue = 'price_range' ,shade= True, data =c)
plt . subplot(2 , 2 , 4)
sns . kdeplot(x='px_width' , y='ram' , hue = 'price_range' ,shade= True, data =d)
# important parameter for price_range


# In[119]:


plt.figure(figsize =(20 , 10))
hm = sns.heatmap(df.corr() , annot=True )
hm.set(title = "Crrelation matrix of the data")
plt.show()


# In[1]:


##################       Decision Tree       ###############
from sklearn.tree import DecisionTreeRegressor


# In[2]:


x = df.drop(['price_range'] , axis = 1)
y = df['price_range']


# In[122]:


x_train , x_test , y_train , y_test  = train_test_split (x , y , test_size=0.2 , random_state = 1)


# In[123]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)


# In[124]:


print("Accuracy :" , metrics.accuracy_score(y_test , y_pred))
print('train_score :',clf . score(x_train , y_train))


# In[125]:


#### model is overfit  ####


# In[126]:


criterion =["squared_error" , 'entropy' , '']
listt = pd . DataFrame()
for i in criterion :
    x = df . drop('price_range' , axis=1)
    y = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.2 , random_state=0)

    clf = DecisionTreeClassifier(criterion = i , max_depth = 30)
    
    
    clf . fit(x_train , y_train)
    y_pred = clf . predict(x_test)
    acc=metrics . accuracy_score(y_test , y_pred)
    dict = {'criterion':i , 'acc':acc , 'score':clf . score(x_train , y_train)}
    listt = listt . append(dict, ignore_index =True)
    def highlight_max (s) :
        is_max = s == s.max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[ ]:


listt . style . apply (highlight_max)

### entropy is better


# In[127]:


max_depth = [1 , 2 ,3 ,4 ,5, 6, 7, 8, 9 ]

listt = pd . DataFrame()
for i in max_depth :
    x = df . drop('price_range' , axis=1)
    y = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.2 , random_state=0)

    clf = DecisionTreeClassifier(criterion = 'entropy' , max_depth = i)
    clf . fit(x_train , y_train)
    y_pred = clf . predict(x_test)
    acc=metrics . accuracy_score(y_test , y_pred)
    dict = {'max_depth':i , 'acc':acc , 'score':clf . score(x , y)}
    listt = listt . append(dict, ignore_index =True)
    def highlight_max (s) :
        is_max = s == s.max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[128]:


listt . style . apply (highlight_max)


# In[129]:


x = df.drop(['price_range'] , axis = 1)
y = df['price_range']
x_train , x_test , y_train , y_test  = train_test_split (x , y , test_size=0.2 , random_state = 0)

clf = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 9 )
clf = clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)

print("Accuracy :" , metrics.accuracy_score(y_test , y_pred))
print('train_score :',clf . score(x_train , y_train))


# In[130]:


plt . figure(figsize=(100,50))

feature_names = df.columns[:21]

target_names = ['0','1', '2', '3']

dt = tree . plot_tree(clf , 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True )


# In[131]:


############    Random Forest   ###########


# In[132]:


max_depth = [5 ,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
listt = pd . DataFrame()
for i in max_depth :
    xR = df . drop('price_range' , axis=1)
    yR = df . price_range. values . reshape(-1 , 1)
    x_train , x_test , y_train , y_test = train_test_split(xR , yR ,test_size=0.2 , random_state=0)
    
    RF = RandomForestClassifier(max_depth = i , random_state = 0) 
    RF . fit(x_train , y_train)
    y_pred = RF . predict(x_test)
    acc=metrics . accuracy_score(y_test , y_pred)
    dict = {'max_depth':i , 'acc':acc , 'score':RF . score(xR , yR)}
    listt = listt . append(dict, ignore_index =True)
    def highlight_max (s) :
        is_max = s == s.max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[133]:


listt . style . apply(highlight_max)


# In[134]:


test_size =[.1 ,.15 ,.2 ,.25 ,.3 , .35 , .4]
listt = pd . DataFrame()
for i in test_size :
    xR = df . drop('price_range' , axis=1)
    yR = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=i , random_state=0)
    RF = RandomForestClassifier(max_depth= 10 , min_samples_leaf= 2 , n_estimators= 100 , criterion='entropy' , min_samples_split = 4 ) 
    RF . fit(x_train , y_train)
    y_pred = RF . predict(x_test)
    acc=metrics . accuracy_score(y_test , y_pred)
    dict = {'test_size':i , 'acc':acc , 'score':RF . score(xR , yR)}
    listt = listt . append(dict, ignore_index =True)
    def highlight_max (s) :
        is_max = s == s.max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[135]:


listt . style . apply(highlight_max)


# In[136]:


n_estimators = [10 ,20 ,50 ,100 , 200,500 , 1000]
listt = pd . DataFrame()
for i in n_estimators :
    xR = df . drop('price_range' , axis=1)
    yR = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(xR , yR ,test_size=0.2 , random_state=0)
    RF = RandomForestClassifier(max_depth= 10 , n_estimators= i ) 
    RF . fit(x_train , y_train)
    y_pred = RF . predict(x_test)
    acc=metrics . accuracy_score(y_test , y_pred)
    dict = {'n_estimators':i , 'acc':acc , 'score':RF . score(xR , yR)}
    listt = listt . append(dict, ignore_index =True)
    def highlight_max (s) :
        is_max = s == s.max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[137]:


listt . style . apply(highlight_max)


# In[138]:


xR = df.drop(['price_range'] , axis = 1)
yR = df['price_range'].values .reshape(-1,1)


# In[139]:


x_train , x_test , y_train , y_test = train_test_split(xR , yR ,test_size=0.15 , random_state=70)

RF = RandomForestClassifier(max_depth= 10 ,  n_estimators= 100 , criterion='gini')  
  

RF . fit(x_train , y_train)
y_pred = RF . predict(x_test)
acc=metrics . accuracy_score(y_test , y_pred)
print('acc',acc)
print('score',RF . score(xR , yR))


# In[140]:


#######################   SVM AND KERNELS    ##################################


# In[141]:


xS = df.drop(['price_range'] , axis = 1)
yS = df['price_range'].values .reshape(-1,1)
x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.15 , random_state=0)
svm = SVC ()
svm . fit(x_train , y_train)
y_pred = svm . predict(x_test)
acc = metrics . accuracy_score(y_test , y_pred)


# In[142]:


kernel = ['linear', 'poly', 'rbf', 'sigmoid']
listt = pd . DataFrame()
for i in kernel :
    xS = df . drop('price_range' , axis=1)
    yS = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.15 , random_state=0)
    
    svm = SVC (kernel = i)
    svm . fit(x_train , y_train)
    y_pred = svm . predict(x_test)
    acc = metrics . accuracy_score(y_test , y_pred)
    dict ={'kernel' : i , 'acc' : acc , 'score' : svm . score(xS , yS)}
    listt = listt . append(dict , ignore_index =True )
    def highlight_max (s) :
        is_max = s == s . max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[143]:


listt . style . apply(highlight_max)


# In[144]:


test_size =[.1 ,.15 ,.2 ,.25 ,.3 , .35 , .4]
listt = pd . DataFrame()
for i in test_size :
    xS = df . drop('price_range' , axis=1)
    yS = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= i , random_state=0)
    
    svm = SVC (kernel = 'linear' , C = 10)
    svm . fit(x_train , y_train)
    y_pred = svm . predict(x_test)
    acc = metrics . accuracy_score(y_test , y_pred)
    dict ={'test_size' : i , 'acc' : acc , 'score' : svm . score(xS , yS)}
    listt = listt . append(dict , ignore_index =True )
    def highlight_max (s) :
        is_max = s == s . max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[145]:


listt . style . apply(highlight_max)


# In[146]:


C = [0.001 , 0.1 , 1 , 10 , 100 , 1000]
listt = pd . DataFrame()
for i in C :
    xS = df . drop('price_range' , axis=1)
    yS = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.2 , random_state=0)
    
    svm = SVC (kernel = 'linear' , C = i)
    svm . fit(x_train , y_train)
    y_pred = svm . predict(x_test)
    acc = metrics . accuracy_score(y_test , y_pred)
    dict ={'C' : i , 'acc' : acc , 'score' : svm . score(xS , yS)}
    listt = listt . append(dict , ignore_index =True )
    def highlight_max (s) :
        is_max = s == s . max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[147]:


listt . style . apply(highlight_max)


# In[148]:


gamma=['scale', 'auto']
listt = pd . DataFrame()
for i in gamma :
    xS = df . drop('price_range' , axis=1)
    yS = df . price_range. values . reshape(-1 , 1)

    x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.2 , random_state=0)
    
    svm = SVC (kernel = 'poly' , C = 10 , degree = 4 , gamma= i)
    svm . fit(x_train , y_train)
    y_pred = svm . predict(x_test)
    acc = metrics . accuracy_score(y_test , y_pred)
    dict ={'gamma' : i , 'acc' : acc , 'score' : svm . score(xS , yS)}
    listt = listt . append(dict , ignore_index =True )
    def highlight_max (s) :
        is_max = s == s . max()
        return ['background-color : yellow' if v else '' for v in is_max]


# In[149]:


listt . style . apply(highlight_max)


# In[150]:


xS = df . drop('price_range' , axis=1)
yS = df . price_range. values . reshape(-1 , 1)
x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.2 , random_state=2)
svm = SVC (kernel = 'linear' , C = 10 ,gamma ='auto'  )
svm . fit(x_train , y_train)
y_pred = svm . predict(x_test)
acc = metrics . accuracy_score(y_test , y_pred)
print('acc',acc)


# In[151]:


###################     SVM is the best model      ########################


# In[153]:


data_test = pd.read_csv (r'C:\Users\iran\Desktop\dataset.exel\test.csv')
data_test


# In[154]:


df_test = pd.DataFrame (data_test)
df_test.drop('id',axis = 1 , inplace =True)


# In[155]:


df_test


# In[156]:


x_test = df_test


# In[157]:


svm = SVC (kernel = 'linear' , C = 10 ,gamma ='auto' )
svm . fit(x_train , y_train)
y_pred = svm . predict(x_test)


# In[158]:


y_pred


# In[ ]:





# In[ ]:




