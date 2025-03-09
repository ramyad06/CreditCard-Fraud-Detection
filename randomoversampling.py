import pandas as pd 
data=pd.read_csv(r"C:\Users\ramya\Downloads\creditcard.csv")

x=data.drop(columns='Class',axis=1)
y=data.Class

from imblearn.over_sampling import RandomOverSampler
import collections
oversample=RandomOverSampler(sampling_strategy='minority',random_state=42)
xs,ys=oversample.fit_resample(x,y)
counter=collections.Counter(ys)
print(counter)