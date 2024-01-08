#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


credit_data = pd.read_csv("C:\\Users\\Ivin\\OneDrive\\Desktop\\creditcard.csv")


# In[3]:


credit_data


# In[4]:


credit_data.head()


# In[5]:


credit_data.info()


# In[6]:


credit_data.shape


# In[7]:


credit_data.columns


# ##### count the fraud and non-fradu values

# In[8]:


count_fraud = credit_data["Class"].value_counts()[1]
non_fraud = credit_data["Class"].value_counts()[0] 


# In[9]:


print(count_fraud)
print(non_fraud)


# ##### sample data

# In[10]:


fraud_data = credit_data[credit_data["Class"]==1]
not_fraud_data = credit_data[credit_data["Class"]==0]


# In[11]:


fraud_cound = 492
non_fraud_count = 10000


# In[12]:


sample_fraud = fraud_data.sample(n=fraud_cound,random_state=42)


# In[13]:


sample_fraud


# In[14]:


sample_non_fraud = not_fraud_data.sample(n=non_fraud_count,random_state=42)


# In[15]:


sample_non_fraud 


# In[16]:


random_sample_data = pd.concat([sample_fraud,sample_non_fraud])


# In[17]:


random_sample_data = random_sample_data.sample(frac=1, random_state=42).reset_index(drop=True)


# In[18]:


random_sample_data


# ##### count the null values

# In[19]:


credit_data.isna().sum()


# In[20]:


credit_data.isna().sum()/len(credit_data)*100


# In[21]:


total_duplicates = random_sample_data.duplicated().sum()


# In[22]:


total_duplicates


# In[23]:


random_sample_data.drop_duplicates(inplace=True)


# In[24]:


print(random_sample_data)


# ###### descriptive Statistics

# In[25]:


descriptive = random_sample_data.describe()
print(descriptive)


# ##### count of fraud and non fraud records

# In[26]:


class_counts_values = random_sample_data['Class'].value_counts()


# In[27]:


class_counts_values


# ##### Split features and label

# In[28]:


x = random_sample_data.drop(["Class"],axis = 1)
y = random_sample_data["Class"]


# In[29]:


x


# In[30]:


y


# ###### split train and test data

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size = 0.3, random_state = 3)


# In[33]:


X_train,X_test,Y_train,Y_test


# In[34]:


X_train,X_test


# In[35]:


Y_train,Y_test


# ###### decision tree 

# In[36]:


from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[37]:


DT = DecisionTreeClassifier(random_state = 3, max_depth = 6)
DT.fit(X_train, Y_train)


# In[38]:


Y_pred = DT.predict(X_test)


# In[39]:


Y_pred


# In[40]:


plt.figure(figsize=(15,10))
plot_tree(DT, filled=True, feature_names=x.columns, class_names=['Non-Fraud', 'Fraud'], rounded=True, fontsize=10)
plt.show()


# In[41]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


# In[42]:


accuracy = accuracy_score(Y_test, Y_pred)
classification_report_result = classification_report(Y_test, Y_pred)
confusion_matrix_result = confusion_matrix(Y_test, Y_pred)


# In[43]:


accuracy


# In[44]:


classification_report_result


# In[45]:


confusion_matrix_result


# In[46]:


precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')


# In[47]:


precision, recall, f1_score, _


# In[48]:


print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# ###### Tree model confusion matrix

# In[50]:


get_ipython().system(' pip install scikit-plot')


# In[51]:


import scikitplot as skplt


# In[52]:


skplt.metrics.plot_confusion_matrix(Y_test, Y_pred)


# In[53]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, accuracy_score, f1_score
import numpy as np


# In[54]:


precision_scorer = make_scorer(precision_score, zero_division=1)
num_folds = 10
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[60]:


precision_scores = cross_val_score(DT, x, y, cv=cv, scoring=precision_scorer)
accuracy_scores = cross_val_score(DT, x, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(DT, x, y, cv=cv, scoring='f1')
recall_scores = cross_val_score(DT, x, y, cv=cv,scoring='recall')


# In[61]:


print(f'Accuracy: {np.mean(accuracy_scores)}')
print(f'Precision: {np.mean(precision_scores)}')
print(f'Recall: {np.mean(recall_scores)}')
print(f'F1 Score: {np.mean(f1_scores)}')


# ###### SVM Model

# In[62]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[63]:


svm = SVC()


# In[64]:


from sklearn.preprocessing import StandardScaler


# In[65]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[67]:


svm.fit(X_train_scaled, Y_train)
pred = svm.predict(X_test_scaled)


# In[68]:


print("Classification Report:\n", classification_report(Y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred))


# In[69]:


precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, pred, average='weighted')


# In[70]:


print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


CM = confusion_matrix(Y_test, pred)
plt.figure(figsize=(4, 2))
sns.heatmap(CM, annot=True, fmt="d", cmap="Blues", xticklabels=svm.classes_, yticklabels=svm.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ###### Logistic Regression Model Training 

# In[77]:


from sklearn.linear_model import LogisticRegression


# In[78]:


LR = LogisticRegression(max_iter=500)

LR.fit(X_train, Y_train)
L_pred = LR.predict(X_test)


# In[79]:


accuracy = accuracy_score(Y_test, Y_pred)
classification_report_result = classification_report(Y_test, Y_pred)
confusion_matrix_result = confusion_matrix(Y_test, Y_pred)


# In[80]:


accuracy = accuracy_score(Y_test, Y_pred)
classification_report_result = classification_report(Y_test, Y_pred)
confusion_matrix_result = confusion_matrix(Y_test, Y_pred)


# In[81]:


precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')


# In[82]:


print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# ###### SVM cross check

# In[86]:


precision_scores = cross_val_score(svm, x, y, cv=cv, scoring=precision_scorer)
accuracy_scores = cross_val_score(svm, x, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(svm, x, y, cv=cv, scoring='f1')
recall_scores = cross_val_score(svm, x, y, cv=cv,scoring='recall')

# Print the average scores
print(f'Accuracy: {np.mean(accuracy_scores)}')
print(f'Precision: {np.mean(precision_scores)}')
print(f'Recall: {np.mean(recall_scores)}')
print(f'F1 Score: {np.mean(f1_scores)}')


# ###### Logistic Regression Model Confusion Matrix Plotting

# In[83]:


skplt.metrics.plot_confusion_matrix(Y_test, Y_pred)


# ###### LR cross check

# In[89]:


precision_scores = cross_val_score(LR, x, y, cv=cv, scoring=precision_scorer)
accuracy_scores = cross_val_score(LR, x, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(LR, x, y, cv=cv, scoring='f1')
recall_scores = cross_val_score(LR, x, y, cv=cv,scoring='recall')

# Print the average scores
print(f'Accuracy: {np.mean(accuracy_scores)}')
print(f'Precision: {np.mean(precision_scores)}')
print(f'Recall: {np.mean(recall_scores)}')
print(f'F1 Score: {np.mean(f1_scores)}')


# ###### KNN 

# In[90]:


from sklearn.neighbors import KNeighborsClassifier


# In[91]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)


# In[92]:


y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
classification_report_result = classification_report(Y_test, y_pred)
confusion_matrix_result = confusion_matrix(Y_test, y_pred)


# In[93]:


print("\nClassification Report:")
print(classification_report_result)
print("\nConfusion Matrix:")
print(confusion_matrix_result)


# In[94]:


precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, y_pred, average='weighted')


# In[95]:


print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[96]:


cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[98]:


precision_scores = cross_val_score(knn, x, y, cv=cv, scoring=precision_scorer)
accuracy_scores = cross_val_score(knn, x, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(knn, x, y, cv=cv, scoring='f1')
recall_scores = cross_val_score(knn, x, y, cv=cv,scoring='recall')


print(f'Accuracy: {np.mean(accuracy_scores)}')
print(f'Precision: {np.mean(precision_scores)}')
print(f'Recall: {np.mean(recall_scores)}')
print(f'F1 Score: {np.mean(f1_scores)}')


# In[ ]:




