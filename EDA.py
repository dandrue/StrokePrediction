"""
EDA

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Line to delete the copy warning from pandas
pd.options.mode.chained_assignment = None  # default='warn'

# Importing the data
data = pd.read_csv('dataset/health_data.csv')

# First look to the data
print(data.head())
print(data.info())
print(data.describe())
print(data.isna().sum().sum())
print(data['age'].sort_values())

# The data has inconsistencies in the age column, below 2 years exists values like 0.08 or 1.8 so I decided
# to drop this values

# It exists some nan values only on the bmi column, before of looking for a relationship with stroke,
# we see if the nan values also be on the same row of positive stroke
data_na = data[data['bmi'].isna() & data['stroke'] == 1]
# print(data_na)
total_stroke = data[data['stroke'] == 1]['stroke'].count()
print('Numero total de strokes ', total_stroke)
# print(data_na.shape[0])
# print(data_na.shape[0]*100/total_stroke)

# Looking for the participation of females and males in the data
num_m = data[data['gender'] == 'Male']['gender'].count()
num_f = data[data['gender'] == 'Female']['gender'].count()
# print(num_m, num_f)
example = data[data['stroke'] == 1]
print(tabulate(example.tail()))
data_by_gender = data.groupby('gender').agg({'stroke': 'sum'})
# print(data_by_gender)
data = data[data['gender'] != 'Other']

# Want to establish a Pmf - probability mass function
age = data[data['age'] >= 2].sort_values('age')
print(age)
age_d = age['age']

# age_d.hist(bins=10, rwidth=0.7)
# plt.xticks(range(0, 85), rotation=90)
# plt.show()

sns.set()
# The number of bins is the sqr of data points
n_bins = int(np.sqrt(data.shape[0]))
sns.histplot(data=data, x=data['age'], bins=n_bins)
plt.show()

sns.set()
sns.scatterplot(data=data, x='age', y='avg_glucose_level', hue='stroke', alpha=0.5)
plt.show()


# Correlation
print(data['gender'].unique())
print(data['ever_married'].unique())
print(data['work_type'].unique())
print(data['Residence_type'].unique())
print(data['smoking_status'].unique())
print(tabulate(data.corr()))
print(data[data['gender'] == 'Other'])
print(data[data['smoking_status'] == 'Unknown'])

columns_to = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level',
              'bmi', 'stroke']

data_m = data[columns_to]
print(data_m.dtypes)
data_m['gender_cat'] = data_m['gender'].astype('category').cat.codes
data_m['ever_married_cat'] = data_m['ever_married'].astype('category').cat.codes
data_m['Residence_type_cat'] = data_m['Residence_type'].astype('category').cat.codes
# data_m['gender_cat_codes'] = data_m['gender_cat']
print(data_m.dtypes)
print(tabulate(data_m.head()))
print(tabulate(data_m.corr()))
print(data_m.isna().sum().sum())
columns_tof = ['age', 'hypertension', 'heart_disease', 'ever_married_cat',
               'avg_glucose_level']

input_data = data_m[columns_tof]
output = data_m['stroke']
print(tabulate(input_data.corr(), headers=input_data.columns))
print(output)

###
nan = np.nan
input_np = input_data.to_numpy()
output_np = output.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(input_np, output_np, test_size=0.3)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)

# for row in range(y_test.shape[0]):
#     print(y_test[row], y_pred[row])


print(input_data.shape)
print(y_test.shape[0]*score)

# Lasso for feature selection
# X = data.drop('stroke', axis=1).values
# y = data['stroke'].values
# names = data.drop('stroke', axis=1).columns
names = input_data.columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(input_data, output).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()

