#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("C:\\Users\\Asus\\OneDrive\\File apps\\Downloads\\heart.csv")


# In[3]:


data.isnull().sum()


# In[4]:


data_dup = data.duplicated().any()


# In[5]:


data_dup


# In[6]:


data = data.drop_duplicates()


# In[7]:


data_dup = data.duplicated().any()


# In[8]:


data_dup


# In[9]:


cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
        


# In[10]:


cate_val


# In[11]:


cont_val


# In[12]:


cate_val


# In[13]:


data['cp'].unique()


# In[14]:


print(cate_val)


# In[15]:


if 'sex' in cate_val:
    cate_val.remove('sex')
if 'target' in cate_val:
    cate_val.remove('target')
data = pd.get_dummies(data,columns=cate_val,drop_first=True)


# In[16]:


print(cate_val)


# In[17]:


print(type(cate_val))


# In[18]:


data.head()


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[21]:


data.head()


# In[22]:


X = data.drop('target',axis=1)


# In[23]:


y = data['target']


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[26]:


X_train


# In[27]:


X_test


# In[28]:


y_train


# In[29]:


y_test


# In[30]:


data.head()


# In[31]:


get_ipython().system('pip install scikit-learn')


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[33]:


log = LogisticRegression()
log.fit(X_train,y_train)


# In[34]:


y_pred1 = log.predict(X_test)


# In[35]:


from sklearn.metrics import accuracy_score


# In[36]:


accuracy_score(y_test,y_pred1)


# In[37]:


from sklearn import svm


# In[38]:


svm = svm.SVC()


# In[39]:


svm.fit(X_train,y_train)


# In[40]:


y_pred2 = svm.predict(X_test)


# In[41]:


accuracy_score(y_test,y_pred2)


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


knn = KNeighborsClassifier()


# In[44]:


knn.fit(X_train,y_train)


# In[45]:


y_pred3=knn.predict(X_test)


# In[46]:


accuracy_score(y_test,y_pred3)


# In[47]:


score = []

for k in range (1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score (y_test,y_pred))


# In[48]:


score


# In[49]:


knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)


# In[50]:


data = pd.read_csv("C:\\Users\\Asus\\OneDrive\\File apps\\Downloads\\heart.csv")


# In[51]:


data.head()


# In[52]:


data = data.drop_duplicates()


# In[53]:


data.shape


# In[54]:


X = data.drop('target',axis=1)
y=data['target']


# In[55]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[56]:


from sklearn.tree import DecisionTreeClassifier


# In[57]:


dt = DecisionTreeClassifier()


# In[58]:


dt.fit(X_train,y_train)


# In[59]:


y_pred4=dt.predict(X_test)


# In[60]:


accuracy_score(y_test,y_pred4)


# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf = RandomForestClassifier()


# In[63]:


rf.fit(X_train,y_train)


# In[64]:


y_pred5=rf.predict(X_test)


# In[65]:


accuracy_score(y_test,y_pred5)


# In[66]:


from sklearn.ensemble import GradientBoostingClassifier


# In[67]:


gbc = GradientBoostingClassifier()


# In[68]:


gbc.fit(X_train,y_train)


# In[69]:


y_pred6=gbc.predict(X_test)


# In[70]:


accuracy_score(y_test,y_pred6)


# In[71]:


final_data = pd. DataFrame ({'Models':['LR','SVM','KNN','DT','RF','GB'],
                           'ACC':[accuracy_score(y_test,y_pred1),
                            accuracy_score(y_test,y_pred2),
                            accuracy_score(y_test,y_pred3),
                            accuracy_score(y_test,y_pred4),
                            accuracy_score(y_test,y_pred5),
                            accuracy_score(y_test,y_pred6)]})


# In[72]:


final_data


# In[73]:


import seaborn as sns


# In[74]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[75]:


X = data.drop('target',axis=1)
y=data['target']


# In[76]:


X.shape


# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


rf = RandomForestClassifier()
rf.fit(X,y)


# In[79]:


import pandas as pd


# In[80]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'threstbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])        


# In[81]:


new_data


# In[82]:


new_data = new_data.rename(columns={'threstbps': 'trestbps'})
p = rf.predict(new_data)
if p[0] == 0:
    print("No Disease")
else:
    print("disease")


# In[83]:


expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

new_data = new_data[expected_features]  # Reorder columns to match training
p = rf.predict(new_data)


# In[84]:


import joblib


# In[85]:


joblib.dump(rf,'model_joblib_heart')


# In[86]:


model = joblib.load('model_joblib_heart')


# In[87]:


model.predict(new_data)


# In[88]:


from tkinter import *
import joblib


# In[89]:


from tkinter import *
import joblib

def show_entry_fields():
    p1 = int(e1.get())
    p2 = int(e2.get())
    p3 = int(e3.get())
    p4 = int(e4.get())
    p5 = int(e5.get())
    p6 = int(e6.get())
    p7 = int(e7.get())
    p8 = int(e8.get())
    p9 = int(e9.get())
    p10 = float(e10.get())
    p11 = int(e11.get())
    p12 = int(e12.get())
    p13 = int(e13.get())

    model = joblib.load('model_joblib_heart')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]])

    if result[0] == 0:
        Label(master, text="No Heart Disease", fg="green").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease", fg="red").grid(row=31)

# Create GUI Window
master = Tk()
master.title("Heart Disease Prediction System")

# Heading Label
Label(master, text="Heart Disease Prediction System", bg="black", fg="white").grid(row=0, columnspan=2)

# Input Labels
Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)

# Input Fields
e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

# Grid Placement for Input Fields
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)

# Predict Button
Button(master, text='Predict', command=show_entry_fields).grid(row=14, columnspan=2)

# Run GUI
mainloop()


# In[ ]:




