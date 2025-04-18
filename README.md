## EXNO:3

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
REG NO: 212224110045
NAME: A PRANEYA
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![Screenshot 2025-04-18 170147](https://github.com/user-attachments/assets/c86eb89a-2d55-48cd-ac23-b83090fe7888)

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','Cold']
```
```
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-18 170155](https://github.com/user-attachments/assets/d60a733e-aa26-4e75-a590-ffa4efd492ab)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-18 170204](https://github.com/user-attachments/assets/f335feb4-8ec2-486a-b2bd-eac737e40a5c)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-18 170212](https://github.com/user-attachments/assets/bd6b374c-e67e-423a-9637-b9a6c13613b2)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
```
```
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-18 170219](https://github.com/user-attachments/assets/dea567f5-b570-41ae-9af1-297aab9f681a)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-18 170229](https://github.com/user-attachments/assets/f68ecdb1-4cd5-44e1-8ca2-cb6ddac4dceb)

```
pip install --upgrade category_encoders
```
![Screenshot 2025-04-18 170243](https://github.com/user-attachments/assets/955f1a0e-b3fa-41d4-b9f9-ecff15431c2d)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```
![Screenshot 2025-04-18 170251](https://github.com/user-attachments/assets/72f006b6-6768-4e62-99e8-7e891257908c)

```
be=BinaryEncoder()
```
```
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2025-04-18 170301](https://github.com/user-attachments/assets/054cb019-9ff7-4607-8e6c-b168dbb7de13)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
```
```
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2025-04-18 170309](https://github.com/user-attachments/assets/3d91af02-561c-41d6-a9a2-2968c0e3b8ff)

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![Screenshot 2025-04-18 170319](https://github.com/user-attachments/assets/2a3055c0-8708-4772-85e2-d2bd56b01488)

```
df.skew()
```
![Screenshot 2025-04-18 170325](https://github.com/user-attachments/assets/a84c2f33-22eb-42c8-a6ea-260cebe855a6)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 170331](https://github.com/user-attachments/assets/0c27eeab-6d2e-4998-9038-f84a434fa7e0)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-18 170345](https://github.com/user-attachments/assets/3e9171e3-5a09-4414-8b61-6f02197188bf)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 170352](https://github.com/user-attachments/assets/71dbe7a4-5c01-426d-8b3d-d9e4fdf46958)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 170400](https://github.com/user-attachments/assets/0d09a109-1668-435d-831f-682cda5ef65d)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-04-18 170414](https://github.com/user-attachments/assets/ba43e6c3-8295-4abc-b46b-f3e77d5fc472)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![Screenshot 2025-04-18 170421](https://github.com/user-attachments/assets/0d06af89-cc32-41c7-bd18-3b46cbb02f8d)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-18 170427](https://github.com/user-attachments/assets/1fb281a5-72d7-4ffb-acd0-a66fac70e388)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
```
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-18 170440](https://github.com/user-attachments/assets/795dc947-b20f-42fd-a13f-225e0dcfbcba)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-18 170455](https://github.com/user-attachments/assets/4023d4e3-5816-4717-a2e5-6a6d41cf0cf1)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-18 170503](https://github.com/user-attachments/assets/c171e4c3-28a3-40cc-ab6e-16be111ef9d5)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-18 170511](https://github.com/user-attachments/assets/96867cbb-c70d-446b-8aa4-4e3afa9130c7)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-18 170521](https://github.com/user-attachments/assets/d03b2a8c-17f9-403e-817a-0cde0b6f40b6)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![Screenshot 2025-04-18 170529](https://github.com/user-attachments/assets/e7d3529e-dc16-486f-8a67-30fb80229dfc)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-18 170538](https://github.com/user-attachments/assets/97c8129c-4034-4a4d-8c14-634946b2a3a9)


# RESULT:
Thus for the given data, Feature Encoding, Transformation process was performed successfully.

       
