import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"C:\Users\ramya\Downloads\creditcard.csv")
data.head()

data.info()

data.duplicated().sum()

data.isnull().sum()

data.describe().T

data.Class.value_counts()

data.hist(bins=30,figsize=(30,30));

plt.figure(figsize=(10,5))
sns.boxplot(x="Class",y="Amount",data=data)
plt.title("Comparision of Transaction Amounts(Fraud vs Non-Fraud)")
plt.xlabel("class(0=Non-Fraud,1=Fraud)")
plt.ylabel("Transaction Amount($)")
plt.show()

data.drop_duplicates(inplace=True)

x=data.drop(columns='Class',axis=1)
y=data.Class