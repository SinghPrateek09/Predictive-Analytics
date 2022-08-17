```python
# Importing necessary libraries 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from scipy import stats
```


```python
# Importing Car Prices Dataset
CP = pd.read_csv('CarPrice_Assignment.csv')

# Displaying first 5 Rows
CP.head()
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>alfa-romero giulia</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>alfa-romero stelvio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>alfa-romero Quadrifoglio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>audi 100 ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>audi 100ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Exploratory Analysis and Summary of the Dataset

CP.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   car_ID            205 non-null    int64  
     1   symboling         205 non-null    int64  
     2   CarName           205 non-null    object 
     3   fueltype          205 non-null    object 
     4   aspiration        205 non-null    object 
     5   doornumber        205 non-null    object 
     6   carbody           205 non-null    object 
     7   drivewheel        205 non-null    object 
     8   enginelocation    205 non-null    object 
     9   wheelbase         205 non-null    float64
     10  carlength         205 non-null    float64
     11  carwidth          205 non-null    float64
     12  carheight         205 non-null    float64
     13  curbweight        205 non-null    int64  
     14  enginetype        205 non-null    object 
     15  cylindernumber    205 non-null    object 
     16  enginesize        205 non-null    int64  
     17  fuelsystem        205 non-null    object 
     18  boreratio         205 non-null    float64
     19  stroke            205 non-null    float64
     20  compressionratio  205 non-null    float64
     21  horsepower        205 non-null    int64  
     22  peakrpm           205 non-null    int64  
     23  citympg           205 non-null    int64  
     24  highwaympg        205 non-null    int64  
     25  price             205 non-null    float64
    dtypes: float64(8), int64(8), object(10)
    memory usage: 41.8+ KB
    


```python
# Checking for the missing or null values

CP.apply(lambda x:np.sum(x == '?'))
```




    car_ID              0
    symboling           0
    CarName             0
    fueltype            0
    aspiration          0
    doornumber          0
    carbody             0
    drivewheel          0
    enginelocation      0
    wheelbase           0
    carlength           0
    carwidth            0
    carheight           0
    curbweight          0
    enginetype          0
    cylindernumber      0
    enginesize          0
    fuelsystem          0
    boreratio           0
    stroke              0
    compressionratio    0
    horsepower          0
    peakrpm             0
    citympg             0
    highwaympg          0
    price               0
    dtype: int64




```python
# Statistical Dataset Analysis of Numeric Series

print(CP.describe())
```

               car_ID   symboling   wheelbase   carlength    carwidth   carheight  \
    count  205.000000  205.000000  205.000000  205.000000  205.000000  205.000000   
    mean   103.000000    0.834146   98.756585  174.049268   65.907805   53.724878   
    std     59.322565    1.245307    6.021776   12.337289    2.145204    2.443522   
    min      1.000000   -2.000000   86.600000  141.100000   60.300000   47.800000   
    25%     52.000000    0.000000   94.500000  166.300000   64.100000   52.000000   
    50%    103.000000    1.000000   97.000000  173.200000   65.500000   54.100000   
    75%    154.000000    2.000000  102.400000  183.100000   66.900000   55.500000   
    max    205.000000    3.000000  120.900000  208.100000   72.300000   59.800000   
    
            curbweight  enginesize   boreratio      stroke  compressionratio  \
    count   205.000000  205.000000  205.000000  205.000000        205.000000   
    mean   2555.565854  126.907317    3.329756    3.255415         10.142537   
    std     520.680204   41.642693    0.270844    0.313597          3.972040   
    min    1488.000000   61.000000    2.540000    2.070000          7.000000   
    25%    2145.000000   97.000000    3.150000    3.110000          8.600000   
    50%    2414.000000  120.000000    3.310000    3.290000          9.000000   
    75%    2935.000000  141.000000    3.580000    3.410000          9.400000   
    max    4066.000000  326.000000    3.940000    4.170000         23.000000   
    
           horsepower      peakrpm     citympg  highwaympg         price  
    count  205.000000   205.000000  205.000000  205.000000    205.000000  
    mean   104.117073  5125.121951   25.219512   30.751220  13276.710571  
    std     39.544167   476.985643    6.542142    6.886443   7988.852332  
    min     48.000000  4150.000000   13.000000   16.000000   5118.000000  
    25%     70.000000  4800.000000   19.000000   25.000000   7788.000000  
    50%     95.000000  5200.000000   24.000000   30.000000  10295.000000  
    75%    116.000000  5500.000000   30.000000   34.000000  16503.000000  
    max    288.000000  6600.000000   49.000000   54.000000  45400.000000  
    


```python
# Statistical Dataset Analysis of Object Series

CP.describe(include=['object'])
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
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>enginetype</th>
      <th>cylindernumber</th>
      <th>fuelsystem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>147</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>top</th>
      <td>toyota corona</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>ohc</td>
      <td>four</td>
      <td>mpfi</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>6</td>
      <td>185</td>
      <td>168</td>
      <td>115</td>
      <td>96</td>
      <td>120</td>
      <td>202</td>
      <td>148</td>
      <td>159</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the unnecessary columns which doesn't influence the Price.

CP.drop('car_ID', axis = 1, inplace = True)
CP.drop('CarName', axis = 1, inplace = True)
```


```python
# Getting all the factor values of the variables and displaying them

fueltype = CP['fueltype'].unique()
print("fueltype : " + str(fueltype))

aspiration = CP['aspiration'].unique()
print("aspiration : " + str(aspiration))

doornumber = CP['doornumber'].unique()
print("doornumber : " + str(doornumber))

carbody = CP['carbody'].unique()
print("carbody : " + str(carbody))

drivewheel = CP['drivewheel'].unique()
print("drivewheel : " + str(drivewheel))

enginelocation = CP['enginelocation'].unique()
print("enginelocation : " + str(enginelocation))

enginetype = CP['enginetype'].unique()
print("enginetype : " + str(enginetype))

cylindernumber = CP['cylindernumber'].unique()
print("cylindernumber : " + str(cylindernumber))

fuelsystem = CP['fuelsystem'].unique()
print("fuelsystem : " + str(fuelsystem))
```

    fueltype : ['gas' 'diesel']
    aspiration : ['std' 'turbo']
    doornumber : ['two' 'four']
    carbody : ['convertible' 'hatchback' 'sedan' 'wagon' 'hardtop']
    drivewheel : ['rwd' 'fwd' '4wd']
    enginelocation : ['front' 'rear']
    enginetype : ['dohc' 'ohcv' 'ohc' 'l' 'rotor' 'ohcf' 'dohcv']
    cylindernumber : ['four' 'six' 'five' 'three' 'twelve' 'two' 'eight']
    fuelsystem : ['mpfi' '2bbl' 'mfi' '1bbl' 'spfi' '4bbl' 'idi' 'spdi']
    


```python
# Assigning the labels for all the factor values of the variables and displaying top 10 rows

fueltype = {'gas': 0,'diesel': 1}
aspiration = {'std': 0, 'turbo': 1}
doornumber = {'two': 0, 'four': 1}
carbody = {'convertible': 0, 'hatchback': 1, 'sedan': 2, 'wagon': 3, 'hardtop': 4}
drivewheel = {'rwd': 0, 'fwd': 1, '4wd': 2}
enginelocation = {'front': 0, 'rear': 1}
enginetype = {'dohc': 0, 'ohcv': 1, 'ohc': 2, 'l': 3, 'rotor': 4, 'ohcf': 5, 'dohcv': 6}
cylindernumber = {'four': 0, 'six': 1, 'five': 2, 'three': 3, 'twelve': 4, 'two': 5, 'eight': 6}
fuelsystem = {'mpfi': 0, '2bbl': 1, 'mfi': 2, '1bbl': 3, 'spfi': 4, '4bbl': 5, 'idi': 6, 'spdi': 7}

CP['fueltype'] = CP['fueltype'].map(fueltype)
CP['aspiration'] = CP['aspiration'].map(aspiration)
CP['doornumber'] = CP['doornumber'].map(doornumber)
CP['carbody'] = CP['carbody'].map(carbody)
CP['drivewheel'] = CP['drivewheel'].map(drivewheel)
CP['enginelocation'] = CP['enginelocation'].map(enginelocation)
CP['enginetype'] = CP['enginetype'].map(enginetype)
CP['cylindernumber'] = CP['cylindernumber'].map(cylindernumber)
CP['fuelsystem'] = CP['fuelsystem'].map(fuelsystem)

CP.head(10)
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
      <th>symboling</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>carwidth</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>...</td>
      <td>130</td>
      <td>0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>...</td>
      <td>130</td>
      <td>0</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>...</td>
      <td>152</td>
      <td>0</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>...</td>
      <td>109</td>
      <td>0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>...</td>
      <td>136</td>
      <td>0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>99.8</td>
      <td>177.3</td>
      <td>66.3</td>
      <td>...</td>
      <td>136</td>
      <td>0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>...</td>
      <td>136</td>
      <td>0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>17710.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>...</td>
      <td>136</td>
      <td>0</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>18920.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>105.8</td>
      <td>192.7</td>
      <td>71.4</td>
      <td>...</td>
      <td>131</td>
      <td>0</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>8.3</td>
      <td>140</td>
      <td>5500</td>
      <td>17</td>
      <td>20</td>
      <td>23875.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>99.5</td>
      <td>178.2</td>
      <td>67.9</td>
      <td>...</td>
      <td>131</td>
      <td>0</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>7.0</td>
      <td>160</td>
      <td>5500</td>
      <td>16</td>
      <td>22</td>
      <td>17859.167</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24 columns</p>
</div>




```python
# Splitting Dataset into Train and Test Set for Target(Dependent) and Independent Variables

Target_Var = CP[['price']]
Independent_Var = CP.iloc[:, list(range(23))]
Independent_Var_train, Independent_Var_test, Target_Var_train, Target_Var_test = train_test_split(Independent_Var, Target_Var, test_size = 0.3, random_state = 25)
```


```python
# Building a Linear Regression Model

Constant_Var = sm.add_constant(Independent_Var_train)
Est = sm.OLS(Target_Var_train, Constant_Var)
Est2 = Est.fit()
print(Est2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.906
    Model:                            OLS   Adj. R-squared:                  0.888
    Method:                 Least Squares   F-statistic:                     49.76
    Date:                Fri, 22 Apr 2022   Prob (F-statistic):           3.04e-50
    Time:                        23:34:59   Log-Likelihood:                -1315.6
    No. Observations:                 143   AIC:                             2679.
    Df Residuals:                     119   BIC:                             2750.
    Df Model:                          23                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const            -9029.8705   1.75e+04     -0.517      0.606   -4.36e+04    2.55e+04
    symboling         -106.8600    280.959     -0.380      0.704    -663.187     449.467
    fueltype           871.2615   7677.349      0.113      0.910   -1.43e+04    1.61e+04
    aspiration        2590.2587   1033.264      2.507      0.014     544.293    4636.225
    doornumber         820.2183    713.316      1.150      0.253    -592.219    2232.655
    carbody           -835.3332    380.432     -2.196      0.030   -1588.626     -82.040
    drivewheel        -806.9348    637.920     -1.265      0.208   -2070.080     456.210
    enginelocation    9714.8994   3418.561      2.842      0.005    2945.807    1.65e+04
    wheelbase           99.9678    121.858      0.820      0.414    -141.323     341.258
    carlength           74.8376     68.810      1.088      0.279     -61.413     211.088
    carwidth          -489.5226    328.836     -1.489      0.139   -1140.650     161.605
    carheight          201.6629    159.477      1.265      0.209    -114.118     517.443
    curbweight           1.7183      1.698      1.012      0.314      -1.643       5.080
    enginetype         304.3040    362.901      0.839      0.403    -414.276    1022.884
    cylindernumber    1324.3231    261.020      5.074      0.000     807.477    1841.169
    enginesize         150.0203     20.478      7.326      0.000     109.471     190.569
    fuelsystem        -254.5248    180.433     -1.411      0.161    -611.800     102.750
    boreratio        -2612.2019   1433.968     -1.822      0.071   -5451.603     227.199
    stroke           -2555.4827   1001.287     -2.552      0.012   -4538.130    -572.835
    compressionratio   114.0580    533.936      0.214      0.831    -943.189    1171.305
    horsepower           0.6623     18.906      0.035      0.972     -36.773      38.097
    peakrpm              2.3675      0.775      3.053      0.003       0.832       3.903
    citympg           -298.5601    193.559     -1.542      0.126    -681.827      84.707
    highwaympg         288.4992    170.090      1.696      0.092     -48.297     625.295
    ==============================================================================
    Omnibus:                       54.098   Durbin-Watson:                   2.149
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              251.303
    Skew:                           1.265   Prob(JB):                     2.69e-55
    Kurtosis:                       8.981   Cond. No.                     4.58e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.58e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

    C:\Users\hp\anaconda3\lib\site-packages\statsmodels\tsa\tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)
    


```python
CP.drop('symboling', axis = 1, inplace = True)
CP.drop('fueltype', axis = 1, inplace = True)
CP.drop('aspiration', axis = 1, inplace = True)
CP.drop('doornumber', axis = 1, inplace = True)
CP.drop('drivewheel', axis = 1, inplace = True)
CP.drop('wheelbase', axis = 1, inplace = True)
CP.drop('carlength', axis = 1, inplace = True)
CP.drop('carwidth', axis = 1, inplace = True)
CP.drop('carheight', axis = 1, inplace = True)
CP.drop('curbweight', axis = 1, inplace = True)
CP.drop('enginetype', axis = 1, inplace = True)
CP.drop('fuelsystem', axis = 1, inplace = True)
CP.drop('boreratio', axis = 1, inplace = True)
CP.drop('compressionratio', axis = 1, inplace = True)
CP.drop('horsepower', axis = 1, inplace = True)
CP.drop('citympg', axis = 1, inplace = True)
CP.drop('highwaympg', axis = 1, inplace = True)
```


```python
# Splitting Dataset into Train and Test Set for Target(Dependent) and Independent Variables

Target_Var = CP[['price']]
Independent_Var = CP.iloc[:, list(range(6))]
Independent_Var_train, Independent_Var_test, Target_Var_train, Target_Var_test = train_test_split(Independent_Var, Target_Var, test_size = 0.3, random_state = 25)
```


```python
# Running the model to increase its accuracy post dropping irrelevant feature variables

Constant_Var = sm.add_constant(Independent_Var_train)
Est = sm.OLS(Target_Var_train, Constant_Var)
Est2 = Est.fit()
print(Est2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.847
    Model:                            OLS   Adj. R-squared:                  0.840
    Method:                 Least Squares   F-statistic:                     125.2
    Date:                Fri, 22 Apr 2022   Prob (F-statistic):           7.14e-53
    Time:                        23:35:23   Log-Likelihood:                -1350.4
    No. Observations:                 143   AIC:                             2715.
    Df Residuals:                     136   BIC:                             2736.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    const          -1.012e+04   4332.621     -2.336      0.021   -1.87e+04   -1552.926
    carbody          221.5490    335.101      0.661      0.510    -441.133     884.231
    enginelocation  5696.8996   3374.994      1.688      0.094    -977.357    1.24e+04
    cylindernumber  1066.1603    238.417      4.472      0.000     594.676    1537.645
    enginesize       171.5792      8.641     19.856      0.000     154.491     188.668
    stroke         -2601.5401    903.208     -2.880      0.005   -4387.689    -815.391
    peakrpm            1.8234      0.634      2.875      0.005       0.569       3.078
    ==============================================================================
    Omnibus:                       28.581   Durbin-Watson:                   2.175
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               46.816
    Skew:                           0.973   Prob(JB):                     6.82e-11
    Kurtosis:                       5.018   Cond. No.                     8.62e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 8.62e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

    C:\Users\hp\anaconda3\lib\site-packages\statsmodels\tsa\tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)
    


```python

```
