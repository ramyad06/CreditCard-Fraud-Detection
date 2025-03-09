import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\ramya\Downloads\creditcard.csv")

x=data.drop(columns='Class',axis=1)
y=data.Class

from imblearn.over_sampling import RandomOverSampler
import collections
oversample=RandomOverSampler(sampling_strategy='minority',random_state=42)
xs,ys=oversample.fit_resample(x,y)
counter=collections.Counter(ys)
print(counter)

from sklearn.model_selection import train_test_split

xs_train,xs_test,ys_train,ys_test=train_test_split(xs,ys,random_state=42,test_size=0.2,stratify=ys)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xs_train,ys_train)

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report
yspredict=model.predict(xs_test)
accuracy=accuracy_score(ys_test,yspredict)
f1=f1_score(ys_test,yspredict)
recall=recall_score(ys_test,yspredict)
precision=precision_score(ys_test,yspredict)

print(f"Accuracy:{accuracy:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")
print(f"F1-score:{f1:.4f}")

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ys_test,yspredict)
label=["0","1"]
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=label,yticklabels=label)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

cr=classification_report(ys_test,yspredict)
print(cr)

from imblearn.over_sampling import SMOTE
import collections
smote=SMOTE(random_state=42)
x_resampled,y_resampled=smote.fit_resample(x,y)
counter=collections.Counter(y_resampled)
counter

xg_train,xg_test,yg_train,yg_test=train_test_split(x_resampled,y_resampled,random_state=42,test_size=0.2,stratify=y_resampled)


import xgboost as xgb 
model=xgb.XGBClassifier()
model.fit(xg_train, yg_train)

ygpredict=model.predict(xg_test)
accuracy=accuracy_score(yg_test,ygpredict)
f1=f1_score(yg_test,ygpredict)
recall=recall_score(yg_test,ygpredict)
precision=precision_score(yg_test,ygpredict)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

cr=classification_report(yg_test,ygpredict)
print(cr)

from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(yg_test,ygpredict)
label= ["0", "1"]
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=label,yticklabels=label)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()