## Project Description:
- This project aims to develop a predictive model for estimating used car prices using machine learning techniques. The model leverages feature engineering, selection, and scaling to improve accuracy. I employ linear regression, polynomial regression, and Lasso regression algorithms to analyze the relationships between car attributes (e.g., mileage, age, make, model) and market price.

### Key Objectives:
- Explore and preprocess the dataset using feature scaling (Min-Max Scaler) and selection techniques.
- Develop and compare the performance of linear, polynomial, and Lasso regression models.
- Evaluate the impact of feature engineering on model accuracy.
- Identify the most influential factors affecting used car prices.

References
- https://www.kaggle.com/code/starlitlolith/train-and-evaluate-a-regression-model-4



#### 2.  Library import and data preparation


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# So that the plot will be saved within the jupyter notebook
%matplotlib inline

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression # use the regression since it is for regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso 

from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
# load csv
df = pd.read_csv('dataset/Lab2_prepared_SPe177.csv')

# have a peak at data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>sqrt_price</th>
      <th>dr_fwd</th>
      <th>dr_rwd</th>
      <th>ts_lien</th>
      <th>ts_missing</th>
      <th>ts_rebuilt</th>
      <th>ts_salvage</th>
      <th>...</th>
      <th>mk_kia</th>
      <th>mk_nissan</th>
      <th>mk_toyota</th>
      <th>ex_black</th>
      <th>ex_blue</th>
      <th>ex_grey</th>
      <th>ex_other</th>
      <th>ex_red</th>
      <th>ex_silver</th>
      <th>ex_white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>8</td>
      <td>83797</td>
      <td>118.300465</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>4</td>
      <td>76119</td>
      <td>104.857045</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>4</td>
      <td>43325</td>
      <td>146.625373</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>4</td>
      <td>73000</td>
      <td>104.880885</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>4</td>
      <td>142000</td>
      <td>80.622577</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
# about the dataset
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5711 entries, 0 to 5710
    Data columns (total 46 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   year          5711 non-null   int64  
     1   cylinders     5711 non-null   int64  
     2   odometer      5711 non-null   int64  
     3   sqrt_price    5711 non-null   float64
     4   dr_fwd        5711 non-null   int64  
     5   dr_rwd        5711 non-null   int64  
     6   ts_lien       5711 non-null   int64  
     7   ts_missing    5711 non-null   int64  
     8   ts_rebuilt    5711 non-null   int64  
     9   ts_salvage    5711 non-null   int64  
     10  fl_electric   5711 non-null   int64  
     11  fl_gas        5711 non-null   int64  
     12  fl_hybrid     5711 non-null   int64  
     13  fl_other      5711 non-null   int64  
     14  tr_manual     5711 non-null   int64  
     15  tr_other      5711 non-null   int64  
     16  ty_hatchback  5711 non-null   int64  
     17  ty_mini_van   5711 non-null   int64  
     18  ty_pickup     5711 non-null   int64  
     19  ty_sedan      5711 non-null   int64  
     20  ty_suv        5711 non-null   int64  
     21  ty_truck      5711 non-null   int64  
     22  ty_van        5711 non-null   int64  
     23  co_excellent  5711 non-null   int64  
     24  co_fair       5711 non-null   int64  
     25  co_good       5711 non-null   int64  
     26  co_like_new   5711 non-null   int64  
     27  co_new        5711 non-null   int64  
     28  co_salvage    5711 non-null   int64  
     29  mk_bmw        5711 non-null   int64  
     30  mk_chevrolet  5711 non-null   int64  
     31  mk_dodge      5711 non-null   int64  
     32  mk_ford       5711 non-null   int64  
     33  mk_gmc        5711 non-null   int64  
     34  mk_honda      5711 non-null   int64  
     35  mk_hyundai    5711 non-null   int64  
     36  mk_kia        5711 non-null   int64  
     37  mk_nissan     5711 non-null   int64  
     38  mk_toyota     5711 non-null   int64  
     39  ex_black      5711 non-null   int64  
     40  ex_blue       5711 non-null   int64  
     41  ex_grey       5711 non-null   int64  
     42  ex_other      5711 non-null   int64  
     43  ex_red        5711 non-null   int64  
     44  ex_silver     5711 non-null   int64  
     45  ex_white      5711 non-null   int64  
    dtypes: float64(1), int64(45)
    memory usage: 2.0 MB
    


```python
# Remove/drop any remaining null values in the dataset if there is any
df.isnull().sum().sum()
```




    0




```python

```


```python

```

#### 3.Exploratory Data Analysis


```python
# [3.a] Print out the summary statistics of the dataset
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>sqrt_price</th>
      <th>dr_fwd</th>
      <th>dr_rwd</th>
      <th>ts_lien</th>
      <th>ts_missing</th>
      <th>ts_rebuilt</th>
      <th>ts_salvage</th>
      <th>...</th>
      <th>mk_kia</th>
      <th>mk_nissan</th>
      <th>mk_toyota</th>
      <th>ex_black</th>
      <th>ex_blue</th>
      <th>ex_grey</th>
      <th>ex_other</th>
      <th>ex_red</th>
      <th>ex_silver</th>
      <th>ex_white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5711.00000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>...</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.05218</td>
      <td>5.481877</td>
      <td>100239.466118</td>
      <td>123.641223</td>
      <td>0.410086</td>
      <td>0.168447</td>
      <td>0.002802</td>
      <td>0.000175</td>
      <td>0.022238</td>
      <td>0.011031</td>
      <td>...</td>
      <td>0.025039</td>
      <td>0.068114</td>
      <td>0.123796</td>
      <td>0.167746</td>
      <td>0.092103</td>
      <td>0.093679</td>
      <td>0.077570</td>
      <td>0.085274</td>
      <td>0.130625</td>
      <td>0.240063</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.67445</td>
      <td>1.661023</td>
      <td>62651.465057</td>
      <td>51.175808</td>
      <td>0.491892</td>
      <td>0.374295</td>
      <td>0.052861</td>
      <td>0.013233</td>
      <td>0.147469</td>
      <td>0.104458</td>
      <td>...</td>
      <td>0.156258</td>
      <td>0.251964</td>
      <td>0.329378</td>
      <td>0.373674</td>
      <td>0.289197</td>
      <td>0.291407</td>
      <td>0.267517</td>
      <td>0.279314</td>
      <td>0.337019</td>
      <td>0.427159</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1991.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.00000</td>
      <td>4.000000</td>
      <td>46142.500000</td>
      <td>86.573668</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2013.00000</td>
      <td>5.000000</td>
      <td>97215.000000</td>
      <td>118.317370</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017.00000</td>
      <td>6.000000</td>
      <td>142324.500000</td>
      <td>158.113883</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021.00000</td>
      <td>12.000000</td>
      <td>347000.000000</td>
      <td>299.998333</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 46 columns</p>
</div>




```python
# [3.b] Print out the correlation of the features.
df.corr()['sqrt_price'].abs().sort_values(ascending=False)
```




    sqrt_price      1.000000
    year            0.581001
    odometer        0.511765
    dr_fwd          0.374211
    fl_gas          0.307148
    ty_pickup       0.275042
    ty_sedan        0.274499
    ty_truck        0.264982
    tr_other        0.225503
    co_fair         0.188672
    fl_other        0.182126
    cylinders       0.168766
    ex_white        0.129638
    mk_gmc          0.127669
    mk_honda        0.115135
    co_excellent    0.114861
    tr_manual       0.104211
    ty_mini_van     0.103410
    mk_ford         0.098498
    mk_nissan       0.092323
    ex_silver       0.086153
    ty_suv          0.083768
    ty_hatchback    0.076417
    mk_hyundai      0.072486
    ts_rebuilt      0.071273
    ex_other        0.070136
    ex_black        0.061131
    mk_dodge        0.059438
    ex_grey         0.055128
    mk_kia          0.054845
    ts_salvage      0.053743
    ex_blue         0.053187
    co_salvage      0.052432
    fl_hybrid       0.050873
    mk_chevrolet    0.048553
    dr_rwd          0.042774
    fl_electric     0.039660
    ty_van          0.029107
    mk_toyota       0.023673
    ts_missing      0.021959
    ts_lien         0.020974
    co_new          0.020436
    co_like_new     0.020355
    co_good         0.012867
    mk_bmw          0.010503
    ex_red          0.008492
    Name: sqrt_price, dtype: float64




```python
#[3.c] Plot the heatmap of the correlation.
ind_heatmap = df.corr()['sqrt_price'].abs().sort_values(ascending=False)[:10].index
df_heatmap = df[ind_heatmap]

# Heatmap of correlation
plt.figure()
sns.heatmap(df[ind_heatmap].corr(), cmap = 'coolwarm', annot=True)
plt.title('Correlation Heatmap of Top 10 Features')

plt.show()
```


    
![output_11_0](https://github.com/user-attachments/assets/488105c2-359e-4c89-a8f9-9e444baf96f8)




```python
# [3.d] Perform multicollinearity analysis and display the VIF data. Which column/feature is the best candidate to be dropped? 

#step 1: the independent variable set
df_independent = df.select_dtypes(include=np.number).drop('sqrt_price',axis=1)

# step 2: calculate VIF vor each features
VIF = [
    variance_inflation_factor(df_independent.values, i) 
    for i in range(len(df_independent.columns))
]

# step 3: create df_VIF dataframe
df_VIF = pd.DataFrame(
    {
        'feature': df_independent.columns,
        'VIF':VIF
    }
)

df_VIF.sort_values(by=['VIF'],ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>year</td>
      <td>107.583087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cylinders</td>
      <td>20.422535</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fl_gas</td>
      <td>16.812231</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ty_sedan</td>
      <td>16.501521</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ty_suv</td>
      <td>13.394460</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ty_pickup</td>
      <td>9.518341</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ty_truck</td>
      <td>9.262888</td>
    </tr>
    <tr>
      <th>2</th>
      <td>odometer</td>
      <td>4.901743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dr_fwd</td>
      <td>4.640323</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ty_hatchback</td>
      <td>4.135793</td>
    </tr>
    <tr>
      <th>31</th>
      <td>mk_ford</td>
      <td>3.718014</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ex_white</td>
      <td>3.298581</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ty_van</td>
      <td>3.083632</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mk_chevrolet</td>
      <td>2.974494</td>
    </tr>
    <tr>
      <th>24</th>
      <td>co_good</td>
      <td>2.616517</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ex_black</td>
      <td>2.547815</td>
    </tr>
    <tr>
      <th>37</th>
      <td>mk_toyota</td>
      <td>2.366687</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ty_mini_van</td>
      <td>2.348849</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ex_silver</td>
      <td>2.210687</td>
    </tr>
    <tr>
      <th>12</th>
      <td>fl_other</td>
      <td>2.182350</td>
    </tr>
    <tr>
      <th>22</th>
      <td>co_excellent</td>
      <td>2.115814</td>
    </tr>
    <tr>
      <th>33</th>
      <td>mk_honda</td>
      <td>1.928014</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tr_other</td>
      <td>1.914250</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ex_grey</td>
      <td>1.882997</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ex_blue</td>
      <td>1.856776</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dr_rwd</td>
      <td>1.845924</td>
    </tr>
    <tr>
      <th>42</th>
      <td>ex_red</td>
      <td>1.802551</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ex_other</td>
      <td>1.734430</td>
    </tr>
    <tr>
      <th>36</th>
      <td>mk_nissan</td>
      <td>1.721858</td>
    </tr>
    <tr>
      <th>32</th>
      <td>mk_gmc</td>
      <td>1.685805</td>
    </tr>
    <tr>
      <th>28</th>
      <td>mk_bmw</td>
      <td>1.479945</td>
    </tr>
    <tr>
      <th>30</th>
      <td>mk_dodge</td>
      <td>1.435856</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fl_hybrid</td>
      <td>1.402196</td>
    </tr>
    <tr>
      <th>34</th>
      <td>mk_hyundai</td>
      <td>1.381606</td>
    </tr>
    <tr>
      <th>35</th>
      <td>mk_kia</td>
      <td>1.271771</td>
    </tr>
    <tr>
      <th>25</th>
      <td>co_like_new</td>
      <td>1.237730</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fl_electric</td>
      <td>1.171751</td>
    </tr>
    <tr>
      <th>23</th>
      <td>co_fair</td>
      <td>1.130813</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tr_manual</td>
      <td>1.110062</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ts_salvage</td>
      <td>1.064234</td>
    </tr>
    <tr>
      <th>27</th>
      <td>co_salvage</td>
      <td>1.053266</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ts_rebuilt</td>
      <td>1.049561</td>
    </tr>
    <tr>
      <th>26</th>
      <td>co_new</td>
      <td>1.022183</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ts_lien</td>
      <td>1.016102</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ts_missing</td>
      <td>1.003956</td>
    </tr>
  </tbody>
</table>
</div>




```python
# because the VIF is quite high, I will try to scale it down
 #use standard scaler
ss = StandardScaler()
scaled_features = ss.fit_transform(df_independent)

#remake the dataframe
scaled_independent = pd.DataFrame(scaled_features, columns=df_independent.columns)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = df_independent.columns
vif_data["VIF"] = [variance_inflation_factor(scaled_independent.values, i)
                  for i in range(len(scaled_independent.columns))]

vif_data.sort_values(by='VIF',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>ty_sedan</td>
      <td>11.670101</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ty_suv</td>
      <td>10.372902</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ty_pickup</td>
      <td>8.154083</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ty_truck</td>
      <td>8.007260</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ty_hatchback</td>
      <td>3.948922</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ty_van</td>
      <td>3.016818</td>
    </tr>
    <tr>
      <th>31</th>
      <td>mk_ford</td>
      <td>2.829409</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dr_fwd</td>
      <td>2.741129</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ex_white</td>
      <td>2.508497</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mk_chevrolet</td>
      <td>2.459546</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fl_gas</td>
      <td>2.431925</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ty_mini_van</td>
      <td>2.313298</td>
    </tr>
    <tr>
      <th>0</th>
      <td>year</td>
      <td>2.126064</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ex_black</td>
      <td>2.122557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>odometer</td>
      <td>2.077502</td>
    </tr>
    <tr>
      <th>37</th>
      <td>mk_toyota</td>
      <td>2.073743</td>
    </tr>
    <tr>
      <th>12</th>
      <td>fl_other</td>
      <td>2.061368</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ex_silver</td>
      <td>1.922961</td>
    </tr>
    <tr>
      <th>24</th>
      <td>co_good</td>
      <td>1.867396</td>
    </tr>
    <tr>
      <th>33</th>
      <td>mk_honda</td>
      <td>1.766780</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cylinders</td>
      <td>1.747669</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tr_other</td>
      <td>1.724292</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ex_grey</td>
      <td>1.707025</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ex_blue</td>
      <td>1.686879</td>
    </tr>
    <tr>
      <th>42</th>
      <td>ex_red</td>
      <td>1.651708</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ex_other</td>
      <td>1.612114</td>
    </tr>
    <tr>
      <th>36</th>
      <td>mk_nissan</td>
      <td>1.611895</td>
    </tr>
    <tr>
      <th>32</th>
      <td>mk_gmc</td>
      <td>1.597638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dr_rwd</td>
      <td>1.551412</td>
    </tr>
    <tr>
      <th>22</th>
      <td>co_excellent</td>
      <td>1.486292</td>
    </tr>
    <tr>
      <th>28</th>
      <td>mk_bmw</td>
      <td>1.421750</td>
    </tr>
    <tr>
      <th>30</th>
      <td>mk_dodge</td>
      <td>1.384719</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fl_hybrid</td>
      <td>1.379892</td>
    </tr>
    <tr>
      <th>34</th>
      <td>mk_hyundai</td>
      <td>1.337386</td>
    </tr>
    <tr>
      <th>35</th>
      <td>mk_kia</td>
      <td>1.245072</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fl_electric</td>
      <td>1.165905</td>
    </tr>
    <tr>
      <th>25</th>
      <td>co_like_new</td>
      <td>1.162504</td>
    </tr>
    <tr>
      <th>23</th>
      <td>co_fair</td>
      <td>1.127252</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tr_manual</td>
      <td>1.093387</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ts_salvage</td>
      <td>1.052589</td>
    </tr>
    <tr>
      <th>27</th>
      <td>co_salvage</td>
      <td>1.052387</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ts_rebuilt</td>
      <td>1.026295</td>
    </tr>
    <tr>
      <th>26</th>
      <td>co_new</td>
      <td>1.020059</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ts_lien</td>
      <td>1.013484</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ts_missing</td>
      <td>1.005127</td>
    </tr>
  </tbody>
</table>
</div>




```python
# [3.d continue] drop some column I think the VIF is too high
df = df.drop(columns=['ty_sedan'])
```


```python
# I will copy the dataset and use df2 from this point.
df2 = df.copy()
```


```python
# [3.e] plot the distribution plot and boxplot of the sqrt_price column

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)

# sns.histplot(df.sqrt_price, kde=True)
sns.distplot(df2.sqrt_price, fit=norm)
plt.title('Distribution of sqrt_price')

plt.subplot(1,2,2)
sns.boxplot(df2.sqrt_price)
plt.title('Boxplot of sqrt_price')
plt.show()

```
 ![output_16_1](https://github.com/user-attachments/assets/60759753-bf54-4ab2-8a90-cbc652035d85)

```python
### My Observation
# - The sqrt_price distribution have a number of outliner which can be remove to better create a normal distribution of the data.
# - The majority of sqrt_price is between Q1 and Q3, thus, this columns can be a good representative of the data.
```


```python
# [3.f] Multivariate Analysis. Make two scatter plots to compare sqrt_price with a few interesting features. 

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
#using plt
plt.scatter(df2.odometer,df2.sqrt_price)
plt.title("Price VS Odermeter")
plt.xlabel("odermeter")
plt.ylabel("Price")

plt.subplot(1,2,2)
#using sns
sns.scatterplot(df2,y=df2.sqrt_price,x=df2.year)
plt.title("Price VS Year")

plt.show()
```

![output_18_0](https://github.com/user-attachments/assets/ddff6dd1-b3f5-4df3-a506-bd73d6d89df6)


```python
# #### [g] Feature Obeservation and Hypothesis
# Base on the Price Vs Odermeter plot
# - Odermeter has negative correlation with price(sqrt_price)
# - Thus, the data shows that price will tend to decreas when the vehical has higher odermeter value.
# Base on the Price vs Year plot
# - Year has positive correlation withe price(sqrt_price)
# - Thus, the data shows that price will tend to decrease as vehical gets older.

```


```python

```


```python

```

#### 4.Feature Selection


```python
# [4.a] assign the sqrt_price column of the dataframe to a new variable called target
target = df['sqrt_price']
```


```python
from Module import Module_SPe177 as md
```


```python
#[4.b] Correlation-based feature selection
k=15
ind_sx = md.selected_correlation(df2,'sqrt_price',k)

#drop sqrt_price
ind_sx.drop('sqrt_price',axis=0, inplace=True)

#find index of selected feature 
selected_feature_corr = ind_sx.index

#make dataframe
df_correlation = pd.DataFrame(df2[selected_feature_corr], columns=selected_feature_corr)
df_correlation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>odometer</th>
      <th>dr_fwd</th>
      <th>fl_gas</th>
      <th>ty_pickup</th>
      <th>ty_truck</th>
      <th>tr_other</th>
      <th>co_fair</th>
      <th>fl_other</th>
      <th>cylinders</th>
      <th>ex_white</th>
      <th>mk_gmc</th>
      <th>mk_honda</th>
      <th>co_excellent</th>
      <th>tr_manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>83797</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>76119</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>43325</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>73000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>142000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reorder the dataframe for easy use in 'select_best' function
sqrt_price = df2.pop('sqrt_price')  
df2['sqrt_price'] = sqrt_price 
```


```python
# [4.c] Select KBest method
num = 10
ind_kb = md.select_best(df2, num)

#make dataframe
df_selKBest = pd.DataFrame(df2[ind_kb], columns=ind_kb)
df_selKBest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>dr_fwd</th>
      <th>fl_gas</th>
      <th>fl_other</th>
      <th>tr_other</th>
      <th>ty_pickup</th>
      <th>ty_truck</th>
      <th>co_fair</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>8</td>
      <td>83797</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>4</td>
      <td>76119</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>4</td>
      <td>43325</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>4</td>
      <td>73000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>4</td>
      <td>142000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```

#### 5. Linear Regression Models with Feature Selection, Feature transformation and Scaling


```python
# Prepare df2 for feature
df2 = df.drop(['sqrt_price'],axis=1)
```


```python
# Variable to use for naming 
feature_selection_Name = ["All Features","Correlation Based", "SelectKBest"]
cor_k = "Correlation Based k={}".format(k)
kBest_num = "SelectKBest k={}".format(num)
```


```python
# part 5 (a) - (f)

# Initialize variable to store results
model_list = [] 
f_select_list = []
f_transform_list = []
f_scaler_list = []
r2_scores_list = []
rmse_scores_list = []

best_model = None
best_r2 = 0
best_rmse = 10000000
best_y_pred = None
best_y_train = None
best_X_train = None
best_X_test = None
best_df = None

df_list = [df2, df_correlation,df_selKBest]
poly_list = [False,True]
scal_list = [False,True]


for i,data in enumerate(df_list):
    

    # Initialize scaler and polynomial features
    for po in poly_list:       
            
        # Apply scaling if specified
        for sc in scal_list:
            

            # Select features based on the configuration
            X = data  
            y = target              

            # Split the data before any transformations to prevent data leakage
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            if po:
                X_train = md.make_poly(X_train)
                X_test = md.make_poly(X_test)
                f_transform_list.append("Poly Degree 2")
            else:
                f_transform_list.append("None")
                
            if sc:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                f_scaler_list.append("MinMax") 
            else:
                f_scaler_list.append("None") 
 

            # Create and fit the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
        
            # Make predictions on the test data
            y_pred = model.predict(X_test)
        
            # Calculate RMSE and R2
            rmse = root_mean_squared_error(y_test,y_pred)
            # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = model.score(X_train, y_train)
        
        
            # Track the best model based on R2 and rmse
            if r2 > best_r2 and rmse < best_rmse:
                best_model = model
                best_r2 = r2
                best_rmse = rmse
                best_y_pred = y_pred
                best_y_train = y_train
                best_X_train = X_train
                best_X_test = X_test
                best_df = data
        
            #Append model to list
            model_list.append(model) 
            r2_scores_list.append(r2)
            rmse_scores_list.append(rmse)
        
            if i == 0:
                f_select_list.append("All Feature")
            elif i == 1:
                f_select_list.append(cor_k)
            elif i == 2:
                f_select_list.append(kBest_num)   
        
# Output the best model information
print("Best model information:")
print(best_model)
print(best_r2)
print(best_rmse)
```

    Best model information:
    LinearRegression()
    0.7138330813463948
    28.691772599192284
    


```python

```


```python

```

#### 6. Linear Regression Model with Lasso


```python
X = df2
y = target
alphas = [100,5,10,0.1,3]

# Initialize list to keep variable for comparibility
lasso_rmse_list = []
lasso_r2_list = []
lasso_model_list = []
best_r2 = 0
best_rmse = 10000000

# modeling the linear regression
# split the training and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=1000)
    lasso.fit(X_train, y_train)
    lasso_prediction = lasso.predict(X_test)

    lasso_r2 = lasso.score(X_test, y_test)
    lasso_rmse = root_mean_squared_error(lasso_prediction, y_test)    

    #append value to list
    lasso_r2_list.append(lasso_r2)
    lasso_rmse_list.append(lasso_rmse)
    lasso_model_list.append(lasso)

    print("Alpha",a,"R2",lasso_r2,"RMSE",lasso_rmse)

```

    Alpha 100 R2 0.25726699010161536 RMSE 43.55905648596454
    Alpha 5 R2 0.531727309604836 RMSE 34.58689029267449
    Alpha 10 R2 0.46572278477226337 RMSE 36.94412865105717
    Alpha 0.1 R2 0.6594851369644628 RMSE 29.493759706919324
    Alpha 3 R2 0.5749422721795174 RMSE 32.95232476933033
    


```python
lasso_df = pd.DataFrame(zip(alphas, lasso_rmse_list, lasso_r2_list), columns=['Alpha', 'RMSE', "R2"])
```


```python
#find out the coefficient of the best result from lasso
print(lasso_df.sort_values(by=['RMSE', 'R2'], ascending=[True, False]))

```

       Alpha       RMSE        R2
    3    0.1  29.493760  0.659485
    4    3.0  32.952325  0.574942
    1    5.0  34.586890  0.531727
    2   10.0  36.944129  0.465723
    0  100.0  43.559056  0.257267
    


```python
#Thus, lasso with alpha = 0.1 gives the lowest RMSE
X = lasso_df.iloc[3,0]
lasso_best_r2 = lasso_df.iloc[3,2]
lasso_best_rmse = lasso_df.iloc[3,1]

#append to list
f_select_list.append("Lasso Alpha = {}".format(X))
f_transform_list.append("None")
f_scaler_list.append("None")
r2_scores_list.append(lasso_best_r2)
rmse_scores_list.append(lasso_best_rmse)
```

#### 7. Plot and summary analysis


```python
#create Dataframe for all models 
df_sum = pd.DataFrame({
    "Feature Selection": f_select_list,
    "Feature Transformation": f_transform_list,
    "Feature Scaling": f_scaler_list,
    "R2": r2_scores_list,
    "RMSE": rmse_scores_list
})

```


```python
df_sum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Selection</th>
      <th>Feature Transformation</th>
      <th>Feature Scaling</th>
      <th>R2</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>All Feature</td>
      <td>None</td>
      <td>None</td>
      <td>0.689467</td>
      <td>2.932623e+01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All Feature</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.689467</td>
      <td>2.932623e+01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All Feature</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.807292</td>
      <td>3.964884e+02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All Feature</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.805706</td>
      <td>1.870097e+12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Correlation Based k=15</td>
      <td>None</td>
      <td>None</td>
      <td>0.655353</td>
      <td>3.051260e+01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Correlation Based k=15</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.655353</td>
      <td>3.051260e+01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Correlation Based k=15</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.713833</td>
      <td>2.869177e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Correlation Based k=15</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.713704</td>
      <td>2.869415e+01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SelectKBest k=10</td>
      <td>None</td>
      <td>None</td>
      <td>0.651256</td>
      <td>3.053375e+01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SelectKBest k=10</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.651256</td>
      <td>3.053375e+01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SelectKBest k=10</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.696661</td>
      <td>2.910013e+01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SelectKBest k=10</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.696571</td>
      <td>2.912223e+01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Lasso Alpha = 0.1</td>
      <td>None</td>
      <td>None</td>
      <td>0.659485</td>
      <td>2.949376e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
#find the best model
df_sum.sort_values(by=["RMSE"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Selection</th>
      <th>Feature Transformation</th>
      <th>Feature Scaling</th>
      <th>R2</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Correlation Based k=15</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.713833</td>
      <td>2.869177e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Correlation Based k=15</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.713704</td>
      <td>2.869415e+01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SelectKBest k=10</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.696661</td>
      <td>2.910013e+01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SelectKBest k=10</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.696571</td>
      <td>2.912223e+01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All Feature</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.689467</td>
      <td>2.932623e+01</td>
    </tr>
    <tr>
      <th>0</th>
      <td>All Feature</td>
      <td>None</td>
      <td>None</td>
      <td>0.689467</td>
      <td>2.932623e+01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Lasso Alpha = 0.1</td>
      <td>None</td>
      <td>None</td>
      <td>0.659485</td>
      <td>2.949376e+01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Correlation Based k=15</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.655353</td>
      <td>3.051260e+01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Correlation Based k=15</td>
      <td>None</td>
      <td>None</td>
      <td>0.655353</td>
      <td>3.051260e+01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SelectKBest k=10</td>
      <td>None</td>
      <td>MinMax</td>
      <td>0.651256</td>
      <td>3.053375e+01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SelectKBest k=10</td>
      <td>None</td>
      <td>None</td>
      <td>0.651256</td>
      <td>3.053375e+01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All Feature</td>
      <td>Poly Degree 2</td>
      <td>None</td>
      <td>0.807292</td>
      <td>3.964884e+02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All Feature</td>
      <td>Poly Degree 2</td>
      <td>MinMax</td>
      <td>0.805706</td>
      <td>1.870097e+12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop  conflicting information
df_correlation = df_correlation.drop(['ty_truck','co_excellent','mk_honda','tr_other','fl_other'], axis=1)
```


```python
#select the best linear model and make the prediction: My best model is Correlation Based k=15, with Poly Degree 2, no scaler
X = df_correlation
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = md.make_poly(X_train)
X_test = md.make_poly(X_test)

 # Create and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
        
# Make predictions on the test data
y_pred = model.predict(X_test)


```


```python
# Plot the scatter plot
plt.figure()
plt.scatter(y_test, y_pred)

z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
```

![output_47_0](https://github.com/user-attachments/assets/1b0757d1-6ded-4eb9-a6cf-b9f1c80016a1)

```python
# Set up to get column name of coefficient 
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)

#Print coefficients
coef_df = pd.DataFrame(model.coef_, columns=['Coefficients'], index=feature_names)
coef_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>year</th>
      <td>-628.080550</td>
    </tr>
    <tr>
      <th>odometer</th>
      <td>0.001498</td>
    </tr>
    <tr>
      <th>dr_fwd</th>
      <td>911.962968</td>
    </tr>
    <tr>
      <th>fl_gas</th>
      <td>126.474557</td>
    </tr>
    <tr>
      <th>ty_pickup</th>
      <td>-181.393056</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>ex_white mk_gmc</th>
      <td>-10.072565</td>
    </tr>
    <tr>
      <th>ex_white tr_manual</th>
      <td>8.142501</td>
    </tr>
    <tr>
      <th>mk_gmc^2</th>
      <td>-860.805344</td>
    </tr>
    <tr>
      <th>mk_gmc tr_manual</th>
      <td>-30.687921</td>
    </tr>
    <tr>
      <th>tr_manual^2</th>
      <td>847.071149</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 1 columns</p>
</div>




```python
# My findings
# My best model is Linear Regression model with Correlation based feature selection, Polynomial degree 2 transformation, and no scaler. I choose this model because, it shows the lowest 'RMSE.' 

# Some example finding of Positive Relationships
#     Odometer: For every additional kilometer on the odometer, the price increases by approximately $0.006 (or $6 per 1000 km).
#     dr_fwd: Vehicles with forward drive (dr_fwd) are associated with a $595 increase in price.
#     mk_honda: Honda vehicles with excellent condition (co_excellent) have a $0.89 increase in price.

# Some example findings of Negative Relationships
#     Year: For every additional year, the price decreases by approximately $590.
#     fl_gas: Gasoline-powered vehicles have a $223 decrease in price.
#     ty_pickup: Pickup trucks have a $1029 decrease in price.
```


```python

```

#### 8. Out of Sample Prediction


```python
df_correlation.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>odometer</th>
      <th>dr_fwd</th>
      <th>fl_gas</th>
      <th>ty_pickup</th>
      <th>co_fair</th>
      <th>cylinders</th>
      <th>ex_white</th>
      <th>mk_gmc</th>
      <th>tr_manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5711.00000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
      <td>5711.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.05218</td>
      <td>100239.466118</td>
      <td>0.410086</td>
      <td>0.855542</td>
      <td>0.156890</td>
      <td>0.018386</td>
      <td>5.481877</td>
      <td>0.240063</td>
      <td>0.055157</td>
      <td>0.036946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.67445</td>
      <td>62651.465057</td>
      <td>0.491892</td>
      <td>0.351584</td>
      <td>0.363729</td>
      <td>0.134353</td>
      <td>1.661023</td>
      <td>0.427159</td>
      <td>0.228306</td>
      <td>0.188646</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1991.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.00000</td>
      <td>46142.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2013.00000</td>
      <td>97215.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017.00000</td>
      <td>142324.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021.00000</td>
      <td>347000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
col = df_correlation.columns
value = df_correlation.describe().loc['25%']
value2 = df_correlation.describe().loc['75%']
```


```python
synthetic_df = pd.DataFrame([value,value2], columns=col)
print(synthetic_df)
```

           year  odometer  dr_fwd  fl_gas  ty_pickup  co_fair  cylinders  \
    25%  2008.0   46142.5     0.0     1.0        0.0      0.0        4.0   
    75%  2017.0  142324.5     1.0     1.0        0.0      0.0        6.0   
    
         ex_white  mk_gmc  tr_manual  
    25%       0.0     0.0        0.0  
    75%       0.0     0.0        0.0  
    


```python
# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
synthetic_df_poly = poly.fit_transform(synthetic_df)

# Make predictions using pre-trained model
predictions = model.predict(synthetic_df_poly)

print(predictions)
```

    [108.02147937 115.85628028]
    


```python
# Before I comment on my prediction, the target variable price was square root transformed 
# so I should perform reverse calculation to get the actual price of the predicted values.
# Calculations:
#    For Vehicle 1 (25%): Actual Price ≈ (108.02147937)² ≈ $11,676
#    For Vehicle 2 (75%): Actual Price ≈ (115.85628028)² ≈ $13,413

# The difference in price between the two cars is not huge, but this is reasonable 
# because many other variables, such as fuel type and transmission, are the same between 
# the two cars.

# Thus, the predictions align with expectations, where newer vehicles in excellent condition 
# tend to be priced higher, even if they have higher mileage. 
```


```python

```

