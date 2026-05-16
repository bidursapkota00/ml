# Machine Learning

## The Machine Learning Process

Every ML project follows three stages.

**1. Data Preprocessing** prepares raw data for modelling. This includes importing data, handling missing values, encoding categorical variables, splitting into training and test sets, and feature scaling.

**2. Modelling** builds and trains a model on the training set, then uses it to make predictions on the test set.

**3. Evaluation** measures model performance using appropriate metrics and determines whether the model is acceptable.

---

# Part 1: Data Preprocessing

## Key Terminology

- **Features** are the input columns (independent variables) used to make predictions.
- **Dependent variable** is the output column the model predicts.
- **`iloc`** stands for "integer location" and is used to select rows and columns by their integer index positions.

## Sample Dataset

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

Here, `Country`, `Age`, and `Salary` are features (X). `Purchased` is the dependent variable (y). The dataset has missing values in the `Age` and `Salary` columns.

## Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

- **NumPy** provides numerical operations and array handling.
- **Matplotlib** is used for plotting charts.
- **Pandas** handles data import and manipulation via DataFrames.

## Step 2: Import and Separate the Dataset

```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

`iloc[:, :-1]` selects all rows and all columns except the last one (features). `iloc[:, -1]` selects all rows of the last column (dependent variable). `.values` converts the selection into a NumPy array.

```python
print(X)
```

```text
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

```python
print(y)
```

```text
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
```

## Step 3: Handle Missing Data

Missing values appear as `nan`. Deleting rows with missing data loses information. A better approach is to replace each missing value with the mean of its column.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

`fit()` computes the mean of columns 1 and 2 (Age and Salary). `transform()` replaces the `nan` values with those computed means. Only numerical columns are imputed because mean is undefined for strings.

```python
print(X)
```

```text
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
```

## Step 4: Encode Categorical Data

ML models work with numbers, not strings. Categorical columns must be converted.

### Encoding the Independent Variable (One-Hot Encoding)

For features, use **One-Hot Encoding**. It creates one binary column per category. This avoids imposing a false numerical order (e.g., France=0, Germany=1, Spain=2 would wrongly imply Germany is "between" France and Spain).

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

`ColumnTransformer` applies `OneHotEncoder` to column index 0 (Country) and passes through the remaining columns unchanged.

```python
print(X)
```

```text
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
```

The three Country values become three binary columns: `[France, Germany, Spain]`.

### Encoding the Dependent Variable (Label Encoding)

For the dependent variable, use **Label Encoding**. It converts categories into integers. This is acceptable here because the model treats y as a label, not a numeric value with magnitude.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

```python
print(y)
```

```text
[0 1 0 0 1 1 0 1 0 1]
```

`No` becomes 0 and `Yes` becomes 1.

## Step 5: Split into Training and Test Sets

The training set is used to train the model. The test set is used to evaluate it on unseen data. A typical split is 80% training and 20% test.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```

`random_state` fixes the random seed so the split is reproducible.

```python
print(X_train)
```

```text
[[0.0 0.0 1.0 38.77777777777778 52000.0]
 [0.0 1.0 0.0 40.0 63777.77777777778]
 [1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 35.0 58000.0]]
```

```python
print(X_test)
```

```text
[[0.0 1.0 0.0 30.0 54000.0]
 [1.0 0.0 0.0 37.0 67000.0]]
```

```python
print(y_train)
```

```text
[0 1 0 0 1 1 0 1]
```

```python
print(y_test)
```

```text
[0 1]
```

## Step 6: Feature Scaling

Features with vastly different ranges (e.g., Age: 27-50 vs Salary: 48000-83000) can cause some features to dominate the model. Feature scaling brings all features to a comparable range. It is always applied column-wise, not across columns or rows.

### Normalization (Min-Max Scaling)

$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Rescales values to the range **[0, 1]**. Best for algorithms that require bounded inputs (k-NN, Neural Networks) or when the data does not follow a normal distribution. Sensitive to outliers.

### Standardization (Z-Score Scaling)

$$X' = \frac{X - \mu}{\sigma}$$

Rescales values to have mean = 0 and standard deviation = 1. Most values fall in **[-3, +3]**. Works well for most algorithms (Linear Regression, SVM, PCA). Less sensitive to outliers.

### Which to Choose?

Standardization is the safer default because it works regardless of the data distribution. Use normalization when the algorithm explicitly requires bounded input values.

### Why Scale After Splitting?

Feature scaling must happen after the train-test split to avoid **data leakage**. If you scale before splitting, the mean, standard deviation, min, and max are computed using test data too. This means the model indirectly "sees" test data during training, producing overly optimistic results that will not generalize.

The correct workflow: fit the scaler on the training set only, then apply that same transformation to the test set.

### Should One-Hot Encoded Columns Be Scaled?

No. One-hot encoded columns are already 0 or 1. Scaling them would distort their meaning as binary indicators and offers no benefit.

### Applying Standardization

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

`fit_transform()` computes mean and standard deviation from the training set and applies the transformation. `transform()` applies the same training-set parameters to the test set without refitting.

```python
print(X_train)
```

```text
[[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]
 [0.0 1.0 0.0 -0.014117293757057777 -0.07013167641635372]
 [1.0 0.0 0.0 0.566708506533324 0.633562432710455]
 [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]
 [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]
 [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]
 [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]
 [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]]
```

```python
print(X_test)
```

```text
[[0.0 1.0 0.0 -1.4661817944830124 -0.9069571034860727]
 [1.0 0.0 0.0 -0.44973664397484414 0.2056403393225306]]
```

Only columns 3 onward (Age, Salary) are scaled. The one-hot encoded columns (0, 1, 2) remain untouched.

---

# Part 2: Simple Linear Regression

## Concept

Regression models predict a continuous real value. Simple Linear Regression models the relationship between one independent variable (X) and one dependent variable (y) as a straight line.

**Formula:**

$$\hat{y} = b_0 + b_1 X_1$$

- $\hat{y}$: predicted value (dependent variable)
- $b_0$: y-intercept (the value of y when X = 0)
- $b_1$: slope coefficient (how much y changes per unit change in X)
- $X_1$: independent variable

**Example:** Predicting potato yield from fertilizer: $Potatoes[t] = 8 + 3 \times Fertilizer[kg]$. Here $b_0 = 8$ tons (base yield) and $b_1 = 3$ tons/kg (yield gain per kg of fertilizer).

## Ordinary Least Squares (OLS)

OLS finds the best-fit line by choosing $b_0$ and $b_1$ that minimize the sum of squared residuals.

**Residual:** the difference between the observed value and the predicted value.

$$\epsilon_i = y_i - \hat{y}_i$$

**Objective:** minimize $\sum (y_i - \hat{y}_i)^2$

This ensures the line is as close as possible to all data points collectively.

## Sample Dataset

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

## Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Step 2: Import the Dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## Step 3: Split into Training and Test Sets

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

Feature scaling is not required for Simple Linear Regression because the `LinearRegression` class handles it internally.

## Step 4: Train the Model

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

`fit()` computes the optimal $b_0$ and $b_1$ using OLS on the training data.

## Step 5: Predict Test Set Results

```python
y_pred = regressor.predict(X_test)
```

## Step 6: Visualize Results

### Training Set

```python
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

Red dots are the actual training data points. The blue line is the regression line predicted by the model.

### Test Set

```python
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

The regression line is the same (trained on the training set). The red dots are the actual test data points. The closer the dots are to the line, the better the model.

## Step 7: Make a Single Prediction

```python
print(regressor.predict([[12]]))
```

```text
[138967.5015615]
```

The model predicts a salary of approximately $138,968 for an employee with 12 years of experience.

The input `[[12]]` uses double brackets because `predict()` expects a 2D array:
- `12` is a scalar
- `[12]` is a 1D array
- `[[12]]` is a 2D array (required format)

## Step 8: Extract the Equation

```python
print(regressor.coef_)
print(regressor.intercept_)
```

```text
[9345.94244312]
26816.192244031183
```

The final regression equation is:

$$Salary = 9345.94 \times YearsExperience + 26816.19$$

`coef_` returns the slope ($b_1$) and `intercept_` returns the y-intercept ($b_0$). These are attributes (not methods), so they are accessed without parentheses.
