### Machine Learning Process

**Data Pre-Processing**

- Import the data
- Clean the data
- Split into training & test sets
- Feature Scaling

**Modelling**

- Build the model
- Train the model
- Make predictions

**Evaluation**

- Calculate performance metrics
- Make a verdict

---

### Training and Testing Set

### Feature Scaling

- always done on column not across columns or rows
- normalization (formula)
- standardization (formula)

#### **Normalization**

$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Range:** [0 ; 1]

#### **Standardization**

$$X' = \frac{X - \mu}{\sigma}$$

**Range:** [-3 ; +3]

### Data Preprocessing

features, independent variable, dependent variable

iloc = locate indexes of rows and columns

standardization or normalization. which to choose

why feature scaling after splitting

should we feature scale all data (like one hot encoded). why not?

data:

```csv
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes
```

# Data Preprocessing Tools

## Importing the libraries

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Importing the dataset

```
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

```
print(X)
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 nan]
     ['France' 35.0 58000.0]
     ['Spain' nan 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]

```
print(y)
```

    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']

## Taking care of missing data

```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

```
print(X)
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 63777.77777777778]
     ['France' 35.0 58000.0]
     ['Spain' 38.77777777777778 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]

## Encoding categorical data

### Encoding the Independent Variable

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

```
print(X)
```

    [[1.0 0.0 0.0 44.0 72000.0]
     [0.0 0.0 1.0 27.0 48000.0]
     [0.0 1.0 0.0 30.0 54000.0]
     [0.0 0.0 1.0 38.0 61000.0]
     [0.0 1.0 0.0 40.0 63777.77777777778]
     [1.0 0.0 0.0 35.0 58000.0]
     [0.0 0.0 1.0 38.77777777777778 52000.0]
     [1.0 0.0 0.0 48.0 79000.0]
     [0.0 1.0 0.0 50.0 83000.0]
     [1.0 0.0 0.0 37.0 67000.0]]

### Encoding the Dependent Variable

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

```
print(y)
```

    [0 1 0 0 1 1 0 1 0 1]

## Splitting the dataset into the Training set and Test set

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```

```
print(X_train)
```

    [[0.0 0.0 1.0 38.77777777777778 52000.0]
     [0.0 1.0 0.0 40.0 63777.77777777778]
     [1.0 0.0 0.0 44.0 72000.0]
     [0.0 0.0 1.0 38.0 61000.0]
     [0.0 0.0 1.0 27.0 48000.0]
     [1.0 0.0 0.0 48.0 79000.0]
     [0.0 1.0 0.0 50.0 83000.0]
     [1.0 0.0 0.0 35.0 58000.0]]

```
print(X_test)
```

    [[0.0 1.0 0.0 30.0 54000.0]
     [1.0 0.0 0.0 37.0 67000.0]]

```
print(y_train)
```

    [0 1 0 0 1 1 0 1]

```
print(y_test)
```

    [0 1]

## Feature Scaling

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

```
print(X_train)
```

    [[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]
     [0.0 1.0 0.0 -0.014117293757057777 -0.07013167641635372]
     [1.0 0.0 0.0 0.566708506533324 0.633562432710455]
     [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]
     [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]
     [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]
     [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]
     [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]]

```
print(X_test)
```

    [[0.0 1.0 0.0 -1.4661817944830124 -0.9069571034860727]
     [1.0 0.0 0.0 -0.44973664397484414 0.2056403393225306]]

---

Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

- **Formula**: $\hat{y} = b_0 + b_1X_1$
- **Labels**:
- $\hat{y}$: Dependent variable
- $b_0$: y-intercept (constant)
- $b_1$: Slope coefficient
- $X_1$: Independent variable

- **General Formula**: $\hat{y} = b_0 + b_1X_1$
- **Applied Formula**: $Potatoes[t] = b_0 + b_1 \times Fertilizer[kg]$
- **Values**:
- $b_0 = 8[t]$
- $b_1 = 3[\frac{t}{kg}]$

- **Heading**: Ordinary Least Squares:
- **Residual Definition**: $residual: \epsilon_i = y_i - \hat{y}_i$
- **Regression Equation**: $\hat{y} = b_0 + b_1X_1$
- **Goal**: $b_0, b_1$ such that: $SUM(y_i - \hat{y}_i)^2$ is minimized

data:

```csv
YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
2.0,43525.00
2.2,39891.00
2.9,56642.00
3.0,60150.00
3.2,54445.00
3.2,64445.00
3.7,57189.00
3.9,63218.00
4.0,55794.00
4.0,56957.00
4.1,57081.00
4.5,61111.00
4.9,67938.00
5.1,66029.00
5.3,83088.00
5.9,81363.00
6.0,93940.00
6.8,91738.00
7.1,98273.00
7.9,101302.00
8.2,113812.00
8.7,109431.00
9.0,105582.00
9.5,116969.00
9.6,112635.00
10.3,122391.00
10.5,121872.00
```

# Simple Linear Regression

## Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Importing the dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

## Training the Simple Linear Regression model on the Training set

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

## Predicting the Test set results

```python
y_pred = regressor.predict(X_test)
```

## Visualising the Training set results

```python
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

![png](/images/2-regression/simple_linear_regression/simple_linear_regression_1.png)

## Visualising the Test set results

```python
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

![png](/images/2-regression/simple_linear_regression/simple_linear_regression_2.png)

## Making a single prediction (for example the salary of an employee with 12 years of experience)

```py
print(regressor.predict([[12]]))
```

    [138967.5015615]

Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.

**Important note:** Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

$12 \rightarrow \textrm{scalar}$

$[12] \rightarrow \textrm{1D array}$

$[[12]] \rightarrow \textrm{2D array}$

## Getting the final linear regression equation with the values of the coefficients

```py
print(regressor.coef_)
print(regressor.intercept_)
```

    [9345.94244312]
    26816.192244031183

Therefore, the equation of our simple linear regression model is:

$$\textrm{Salary} = 9345.94 \times \textrm{YearsExperience} + 26816.19$$

**Important Note:** To get these coefficients we called the "coef*" and "intercept*" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.

---

1-2

Dummy variables for categorical data and dummy variable trap

3

statistical significance and p value

4

5-10
