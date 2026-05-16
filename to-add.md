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

## Multiple Linear Regression Formula

$$\hat{y} = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n$$

- **$\hat{y}$:** Dependent variable
- **$b_0$:** y-intercept (constant)
- **$b_1$:** Slope coefficient 1
- **$X_1$:** Independent variable 1
- **$b_2$:** Slope coefficient 2
- **$X_2$:** Independent variable 2
- **$b_n$:** Slope coefficient n
- **$X_n$:** Independent variable n

---

## Assumptions of Linear Regression

### 1. Linearity

(Linear relationship between Y and each X)

### 2. Homoscedasticity

(Equal variance)

### 3. Multivariate Normality

(Normality of error distribution)

### 4. Independence

(of observations. Includes "no autocorrelation")

### 5. Lack of Multicollinearity

(Predictors are not correlated with each other)

- ✔️ $X_1 \nsim X_2$
- ❌ $X_1 \sim X_2$

### 6. The Outlier Check

(This is not an assumption, but an "extra")

---

Dummy variables for categorical data and dummy variable trap

## Dummy Variables

### Data Table

| Profit     | R&D Spend  | Admin      | Marketing  | State      |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| 192,261.83 | 165,349.20 | 136,897.80 | 471,784.10 | New York   |
| 191,792.06 | 162,597.70 | 151,377.59 | 443,898.53 | California |
| 191,050.39 | 153,441.51 | 101,145.55 | 407,934.54 | California |
| 182,901.99 | 144,372.41 | 118,671.85 | 383,199.62 | New York   |
| 166,187.94 | 142,107.34 | 91,391.77  | 366,168.42 | California |

### Dummy Variables Columns

| New York | California |
| -------- | ---------- |
| 1        | 0          |
| 0        | 1          |
| 0        | 1          |
| 1        | 0          |
| 0        | 1          |

### Equation

$$y = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2 + b_3 \cdot x_3 + b_4 \cdot D_1 + \cancel{b_5 \cdot D_2}$$

> **Always omit one dummy variable**

---

statistical significance and p value

## Statistical Significance

- $H_0$: This is a fair coin
- $H_1$: This is not a fair coin

### P-Value Scale

- 0.5
- 0.25
- 0.12
- 0.06
- **$\alpha = 0.05$** _(Significance threshold line)_
- 0.03
- 0.01

---

## **Building A Model**

**5 methods of building models:**

1. All-in
2. Backward Elimination
3. Forward Selection
4. Bidirectional Elimination
5. Score Comparison

> _Note: Methods 2, 3, and 4 are grouped together under the bracket:_ **Stepwise Regression**

---

## **"All-in" – cases:**

- Prior knowledge; OR
- You have to; OR
- Preparing for Backward Elimination

---

## **Backward Elimination**

- **STEP 1:** Select a significance level to stay in the model (e.g. SL = 0.05)
- **STEP 2:** Fit the full model with all possible predictors
- **STEP 3:** Consider the predictor with the highest P-value. If P > SL, go to STEP 4, otherwise go to FIN
- **STEP 4:** Remove the predictor
- **STEP 5:** Fit model without this variable\* and go back to step 3.

---

## **Forward Selection**

- **STEP 1:** Select a significance level to enter the model (e.g. SL = 0.05)
- **STEP 2:** Fit all simple regression models **y ~ xₙ** Select the one with the lowest P-value
- **STEP 3:** Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
- **STEP 4:** Consider the predictor with the lowest P-value. If P < SL, go to STEP 3, otherwise go to FIN

## **Bidirectional Elimination**

- **STEP 1:** Select a significance level to enter and to stay in the model
- e.g.: SLENTER = 0.05, SLSTAY = 0.05

- **STEP 2:** Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
- **STEP 3:** Perform ALL steps of Backward Elimination (old variables must have P < SLSTAY to stay). Go to step 2.
- **STEP 4:** No new variables can enter and no old variables can exit
- **FIN:** Your Model Is Ready

---

## **All Possible Models**

- **STEP 1:** Select a criterion of goodness of fit (e.g. Akaike criterion)
- **STEP 2:** Construct All Possible Regression Models: $2^N-1$ total combinations
- **STEP 3:** Select the one with the best criterion
- **FIN:** Your Model Is Ready
