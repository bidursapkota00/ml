# Machine Learning Complete Guide

![Bidur Sapkota](https://www.bidursapkota.com.np/images/gravatar.webp "Bidur Sapkota - Developer")&nbsp;[Bidur Sapkota](https://www.bidursapkota.com.np/)

![Machine Learning Complete Guide by Bidur Sapkota](ml-1200.webp "Machine Learning Complete Guide - Blog by Bidur Sapkota")

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Simple Linear Regression](#simple-linear-regression)
3. [Multiple Linear Regression](#multiple-linear-regression)

---

## The ML Process

Every ML project follows three stages.

**1. Data Preprocessing** prepares raw data for modelling. This includes importing data, handling missing values, encoding categorical variables, splitting into training and test sets, and feature scaling.

**2. Modelling** builds and trains a model on the training set, then uses it to make predictions on the test set.

**3. Evaluation** measures model performance using appropriate metrics and determines whether the model is acceptable.

---

## Data Preprocessing

### Key Terminology

- **Features** are the input columns (independent variables) used to make predictions.
- **Dependent variable** is the output column the model predicts.
- **`iloc`** stands for "integer location" and is used to select rows and columns by their integer index positions.

### Sample Dataset

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

### Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

- **NumPy** provides numerical operations and array handling.
- **Matplotlib** is used for plotting charts.
- **Pandas** handles data import and manipulation via DataFrames.

### Importing the Dataset

```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

`pd.read_csv('Data.csv')` reads the CSV file and returns a DataFrame. `iloc[:, :-1]` selects all rows (`:`) and all columns except the last one (`:-1`), which gives the features. `iloc[:, -1]` selects all rows of the last column, which is the dependent variable. `.values` converts the DataFrame selection into a NumPy array, which scikit-learn expects as input.

**Important `read_csv` parameters:**

| Parameter   | Purpose                                                |
| ----------- | ------------------------------------------------------ |
| `sep`       | Delimiter to use (default `,`). Use `sep='\t'` for TSV |
| `header`    | Row number to use as column names (default `0`)        |
| `index_col` | Column to use as the row index                         |
| `na_values` | Additional strings to recognize as NaN                 |
| `encoding`  | File encoding (e.g., `'utf-8'`, `'latin-1'`)           |

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

### Handling Missing Data

Missing values appear as `nan`. Deleting rows with missing data loses information. A better approach is to replace each missing value with the mean of its column.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

`SimpleImputer` replaces missing values based on a chosen strategy. `missing_values=np.nan` tells the imputer what to look for; the default is `np.nan`. `strategy='mean'` replaces each missing value with the mean of its column. `fit()` computes the mean of columns 1 and 2 (Age and Salary). `transform()` replaces the `nan` values with those computed means. Only numerical columns are imputed because mean is undefined for strings. `X[:, 1:3]` selects columns at index 1 and 2 (Python slicing excludes the end index).

**Important `SimpleImputer` parameters:**

| Parameter        | Purpose                                                            |
| ---------------- | ------------------------------------------------------------------ |
| `strategy`       | `'mean'` (default), `'median'`, `'most_frequent'`, `'constant'`    |
| `fill_value`     | Value to use when `strategy='constant'` (e.g., `0` or `'missing'`) |
| `missing_values` | The placeholder for missing values (default `np.nan`)              |

`'median'` is more robust to outliers than `'mean'`. `'most_frequent'` replaces with the mode, useful for categorical columns. `'constant'` fills with a fixed value specified by `fill_value`.

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

### Encoding Categorical Data

ML models work with numbers, not strings. Categorical columns must be converted.

#### One-Hot Encoding

For features/independent variables, use One-Hot Encoding. It creates one binary column per category. This avoids imposing a false numerical order (e.g., France=0, Germany=1, Spain=2 would wrongly imply Germany is "between" France and Spain).

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

`ColumnTransformer` applies different transformations to different columns in a single step. `transformers` takes a list of tuples in the format `(name, transformer, columns)`. `'encoder'` is a label you choose for this transformation. `OneHotEncoder()` is the transformer to apply. `[0]` specifies column index 0 (Country) as the target. `remainder='passthrough'` keeps all other columns unchanged; without this, non-specified columns are dropped by default. `fit_transform()` computes the encoding and applies it in one step. The result is wrapped in `np.array()` to ensure a consistent NumPy array format.

**Important `ColumnTransformer` parameters:**

| Parameter   | Purpose                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `remainder` | `'drop'` (default) drops untransformed columns, `'passthrough'` keeps them |

**Important `OneHotEncoder` parameters:**

| Parameter        | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `drop`           | `'first'` drops the first category to avoid multicollinearity        |
| `sparse_output`  | `False` returns a dense array instead of a sparse matrix             |
| `handle_unknown` | `'error'` (default) or `'ignore'` for unseen categories at transform |

`drop='first'` is important for linear models where having all dummy columns creates perfect multicollinearity (the dummy variable trap).

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

#### Label Encoding (Dependent Variable)

For the dependent variable, use Label Encoding. It converts categories into integers. This is acceptable here because the model treats y as a label, not a numeric value with magnitude.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

`LabelEncoder` assigns an integer to each unique class in alphabetical order. `fit_transform()` learns the mapping and applies it in one step. `le.classes_` stores the original class labels if you need to reverse the encoding later using `le.inverse_transform()`.

```python
print(y)
```

```text
[0 1 0 0 1 1 0 1 0 1]
```

`No` becomes 0 and `Yes` becomes 1.

### Splitting the Dataset

The training set is used to train the model. The test set is used to evaluate it on unseen data. A typical split is 80% training and 20% test.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```

`train_test_split` shuffles the data and splits it into training and test subsets. `X, y` are the feature matrix and target vector to split. `test_size=0.2` allocates 20% of the data to the test set and the remaining 80% to training. `random_state=1` fixes the random seed so the split is reproducible; any integer works, but using the same value always produces the same split.

**Important `train_test_split` parameters:**

| Parameter      | Purpose                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `test_size`    | Fraction (`0.2`), integer (count), or `None`. Default is `0.25`              |
| `train_size`   | Complement of `test_size`. Can specify instead of or along with it           |
| `random_state` | Seed for reproducibility. `None` (default) uses a different split each time  |
| `shuffle`      | `True` (default) shuffles data before splitting. Set `False` for time-series |
| `stratify`     | Pass `y` to ensure the class ratio is preserved in both sets                 |

`stratify=y` is important for imbalanced datasets. If 90% of samples are class A and 10% are class B, stratification ensures both the training and test sets maintain this 90/10 ratio.

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

### Feature Scaling

Features with vastly different ranges (e.g., Age: 27-50 vs Salary: 48000-83000) can cause some features to dominate the model. Feature scaling brings all features to a comparable range. It is always applied column-wise, not across columns or rows.

#### Normalization (Min-Max Scaling)

$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Rescales values to the range **[0, 1]**. Best for algorithms that require bounded inputs (k-NN, Neural Networks) or when the data does not follow a normal distribution. Sensitive to outliers.

#### Standardization (Z-Score Scaling)

$$X' = \frac{X - \mu}{\sigma}$$

Rescales values to have mean = 0 and standard deviation = 1. Most values fall in [-3, +3]. Works well for most algorithms (Linear Regression, SVM, PCA). Less sensitive to outliers.

#### Which to Choose?

Standardization is the safer default because it works regardless of the data distribution. Use normalization when the algorithm explicitly requires bounded input values.

#### Why Scale After Splitting?

Feature scaling must happen after the train-test split to avoid data leakage. If you scale before splitting, the mean, standard deviation, min, and max are computed using test data too. This means the model indirectly "sees" test data during training, producing overly optimistic results that will not generalize.

The correct workflow: fit the scaler on the training set only, then apply that same transformation to the test set.

#### Should One-Hot Columns Be Scaled?

No. One-hot encoded columns are already 0 or 1. Scaling them would distort their meaning as binary indicators and offers no benefit.

#### Applying Standardization

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

`StandardScaler` transforms features by removing the mean and scaling to unit variance. `fit_transform()` computes the mean ($\mu$) and standard deviation ($\sigma$) from the training set and applies the transformation $\frac{X - \mu}{\sigma}$ in one step. `transform()` applies the same training-set parameters ($\mu$ and $\sigma$) to the test set without refitting, preventing data leakage. `X_train[:, 3:]` selects columns from index 3 onward (Age, Salary), leaving the one-hot encoded columns (0, 1, 2) untouched.

**Important `StandardScaler` parameters:**

| Parameter   | Purpose                                                                            |
| ----------- | ---------------------------------------------------------------------------------- |
| `with_mean` | `True` (default) centers data by subtracting the mean. Set `False` for sparse data |
| `with_std`  | `True` (default) scales data to unit variance                                      |

**Alternative scalers in scikit-learn:**

| Scaler         | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `MinMaxScaler` | Scales to a given range, default [0, 1]. Use `feature_range` to change |
| `RobustScaler` | Uses median and IQR instead of mean and std. Robust to outliers        |
| `MaxAbsScaler` | Scales by the maximum absolute value. Good for sparse data             |
| `Normalizer`   | Scales each sample (row) to unit norm, not each feature                |

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

## Simple Linear Regression

Regression models predict a continuous real value. Simple Linear Regression models the relationship between one independent variable (X) and one dependent variable (y) as a straight line.

**Formula:**

$$\hat{y} = b_0 + b_1 X_1$$

- $\hat{y}$: predicted value (dependent variable)
- $b_0$: y-intercept (the value of y when X = 0)
- $b_1$: slope coefficient (how much y changes per unit change in X)
- $X_1$: independent variable

**Example:** Predicting potato yield from fertilizer: $Potatoes[t] = 8 + 3 \times Fertilizer[kg]$. Here $b_0 = 8$ tons (base yield) and $b_1 = 3$ tons/kg (yield gain per kg of fertilizer).

### Ordinary Least Squares (OLS)

OLS finds the best-fit line by choosing $b_0$ and $b_1$ that minimize the sum of squared residuals.

**Residual:** the difference between the observed value and the predicted value.

$$\epsilon_i = y_i - \hat{y}_i$$

**Objective:** minimize $\sum (y_i - \hat{y}_i)^2$

This ensures the line is as close as possible to all data points collectively.

### Sample Dataset

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

### Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Importing the Dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### Splitting the Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

`test_size=1/3` allocates one-third of the data (10 observations) to the test set and the remaining two-thirds (20 observations) to training. `random_state=0` fixes the seed for reproducibility.

Feature scaling is not required for Simple Linear Regression because the `LinearRegression` class handles it internally.

### Training the Model

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

`LinearRegression()` creates a linear regression model using Ordinary Least Squares. `fit(X_train, y_train)` computes the optimal $b_0$ (intercept) and $b_1$ (slope) by minimizing the sum of squared residuals on the training data. After fitting, the model stores these values in `regressor.intercept_` and `regressor.coef_`.

**Important `LinearRegression` parameters:**

| Parameter       | Purpose                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `fit_intercept` | `True` (default) calculates the intercept $b_0$. Set `False` to force the line through the origin |
| `copy_X`        | `True` (default) copies X before fitting. Set `False` to modify X in place and save memory        |
| `n_jobs`        | Number of CPU cores for computation. `None` (default) uses 1 core. `-1` uses all cores            |
| `positive`      | `False` (default). Set `True` to force all coefficients to be positive                            |

`fit_intercept=False` is used when you know the relationship passes through the origin (e.g., zero input means zero output). `n_jobs` only affects multi-target regression (multiple y columns); for single-target regression it has no effect.

### Predicting Results

```python
y_pred = regressor.predict(X_test)
```

`predict(X_test)` takes the test features and returns predicted salary values using the learned equation $\hat{y} = b_0 + b_1 X_1$. The input must be a 2D array (matrix), which is why `X_test` from `iloc` already has the correct shape.

### Visualizing Results

#### Training Set

```python
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

![Salary vs Experience (Training set) - Simple Linear Regression](/images/2-regression/simple_linear_regression/simple_linear_regression_1.png)

`plt.scatter(X_train, y_train, color='red')` plots the actual training data as red dots. `plt.plot(X_train, regressor.predict(X_train), color='blue')` draws the regression line in blue by plotting predicted values against the training features. `plt.title()`, `plt.xlabel()`, and `plt.ylabel()` set the chart title and axis labels. `plt.show()` renders and displays the plot.

**Important `plt.scatter` parameters:**

| Parameter | Purpose                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `s`       | Marker size in points² (default `20`)                                         |
| `c`       | Color or array of colors for each point                                       |
| `marker`  | Marker shape: `'o'` (default circle), `'s'` (square), `'^'` (triangle), `'x'` |
| `alpha`   | Transparency from 0 (invisible) to 1 (opaque)                                 |
| `label`   | Legend label for this dataset                                                 |

#### Test Set

```python
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

![Salary vs Experience (Test set) - Simple Linear Regression](/images/2-regression/simple_linear_regression/simple_linear_regression_2.png)

The regression line is the same (trained on the training set). The red dots are the actual test data points. The closer the dots are to the line, the better the model. Note that `plt.plot` still uses `X_train` to draw the line because the model was trained on that data; the line itself does not change.

### Single Prediction

```python
print(regressor.predict([[12]]))
```

```text
[138967.5015615]
```

The model predicts a salary of approximately $138,968 for an employee with 12 years of experience.

The input `[[12]]` uses double brackets because `predict()` expects a 2D array: the outer brackets create a list of samples, and the inner brackets define a single sample with one feature.

### Extracting the Equation

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

`coef_` returns an array of slope coefficients ($b_1$). It is an array because in multiple regression there would be one coefficient per feature. `intercept_` returns the y-intercept ($b_0$) as a single float. Together they define the complete linear equation. You can also compute the model's $R^2$ score using `regressor.score(X_test, y_test)`, which returns a value between 0 and 1 indicating how well the model explains the variance in the data.

---

## Multiple Linear Regression

Multiple Linear Regression extends Simple Linear Regression to multiple independent variables. Instead of fitting a line in 2D, it fits a hyperplane in higher-dimensional space.

**Formula:**

$$\hat{y} = b_0 + b_1 X_1 + b_2 X_2 + \dots + b_n X_n$$

- $\hat{y}$: predicted value (dependent variable)
- $b_0$: y-intercept (constant term, the value of y when all X's are 0)
- $b_1, b_2, \dots, b_n$: slope coefficients (how much y changes per unit change in each respective X, holding all other X's constant)
- $X_1, X_2, \dots, X_n$: independent variables

**Example:** Predicting profit from R&D spend, admin cost, and marketing spend: $Profit = b_0 + b_1 \times R\&D + b_2 \times Admin + b_3 \times Marketing$.

### Assumptions of Linear Regression

Before building a linear regression model, five assumptions must hold for the results to be reliable. These apply to both simple and multiple linear regression.

**1. Linearity** — There must be a linear relationship between the dependent variable and each independent variable. If the true relationship is curved, a linear model will systematically mispredict. Check with scatter plots of y vs each X.

**2. Homoscedasticity** — The variance of the residuals (errors) must be constant across all levels of the independent variables. If the spread of residuals increases or decreases with the predicted value (heteroscedasticity), the model's confidence intervals and p-values become unreliable. Check by plotting residuals vs predicted values.

**3. Multivariate Normality** — The residuals must be normally distributed. This assumption is needed for hypothesis testing (p-values, confidence intervals) to be valid. With large sample sizes, this becomes less critical due to the Central Limit Theorem. Check with a Q-Q plot or histogram of residuals.

**4. Independence** — Observations must be independent of each other. There should be no autocorrelation (where residuals are correlated with each other). This is especially relevant in time-series data where consecutive observations may be related. Check with the Durbin-Watson test.

**5. No Multicollinearity** — Independent variables must not be highly correlated with each other. If $X_1$ and $X_2$ are strongly correlated, the model cannot reliably determine the individual effect of each variable. This inflates the variance of coefficient estimates and makes them unstable. Check with the Variance Inflation Factor (VIF); a VIF above 5-10 indicates problematic multicollinearity.

Additionally, always check for outliers. Outliers are not an assumption but can heavily influence the regression line, especially in small datasets. Use leverage plots and Cook's distance to identify influential points.

![Linear Regression Assumptions](/images/2-regression/multiple_linear_regression/assumptions-of-linear-regression.png)

### Dummy Variables

Categorical variables (e.g., State: New York, California) cannot be used directly in the regression equation. They must be converted into numerical form using dummy variables.

**Example Dataset:**

| Profit     | R&D Spend  | Admin      | Marketing  | State      |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| 192,261.83 | 165,349.20 | 136,897.80 | 471,784.10 | New York   |
| 191,792.06 | 162,597.70 | 151,377.59 | 443,898.53 | California |
| 191,050.39 | 153,441.51 | 101,145.55 | 407,934.54 | California |
| 182,901.99 | 144,372.41 | 118,671.85 | 383,199.62 | New York   |
| 166,187.94 | 142,107.34 | 91,391.77  | 366,168.42 | California |

One-Hot Encoding creates a binary column for each category:

| New York | California |
| -------- | ---------- |
| 1        | 0          |
| 0        | 1          |
| 0        | 1          |
| 1        | 0          |
| 0        | 1          |

The regression equation becomes:

$$y = b_0 + b_1 \cdot X_1 + b_2 \cdot X_2 + b_3 \cdot X_3 + b_4 \cdot D_1$$

Where $X_1, X_2, X_3$ are the numerical features (R&D, Admin, Marketing) and $D_1$ is the dummy variable for one state.

#### The Dummy Variable Trap

If a categorical variable has $n$ categories, you must include only $n - 1$ dummy columns. Including all $n$ creates perfect multicollinearity because one column can always be predicted from the others (e.g., if New York = 0, then California must = 1). This is called the dummy variable trap.

**Always omit one dummy variable.** The omitted category becomes the baseline, and the coefficients of the remaining dummies represent the difference relative to that baseline. For example, if California is omitted, the coefficient of New York tells you how much more (or less) profit New York generates compared to California, all else equal.

Note: scikit-learn's `LinearRegression` does not require you to manually drop a dummy column because it can handle the multicollinearity internally. However, for statistical models (e.g., `statsmodels`), you must drop one column or use `OneHotEncoder(drop='first')`.

### Statistical Significance & P-Values

Statistical significance determines whether a predictor actually has a real effect on the dependent variable or whether the observed relationship is due to random chance.

**Null Hypothesis ($H_0$):** The predictor has no effect on the dependent variable (its coefficient is zero).

**Alternative Hypothesis ($H_1$):** The predictor does have an effect (its coefficient is not zero).

The **p-value** is the probability of observing the data (or something more extreme) assuming $H_0$ is true. A small p-value means it is unlikely the result occurred by chance.

**Significance Level ($\alpha$):** The threshold for rejecting $H_0$, typically set to 0.05 (5%).

- If p-value < $\alpha$ (e.g., p < 0.05): Reject $H_0$. The predictor is statistically significant.
- If p-value ≥ $\alpha$ (e.g., p ≥ 0.05): Fail to reject $H_0$. The predictor is not statistically significant and may not belong in the model.

### Building a Model

With multiple potential predictors, you need to decide which variables to include. There are five methods for building a model.

#### All-In

Include all predictors in the model. Use when:

- You have prior knowledge that all variables are relevant.
- The domain requires it (e.g., regulatory compliance).
- You are preparing for Backward Elimination.

#### Backward Elimination

1. Select a significance level to stay in the model (e.g., SL = 0.05).
2. Fit the full model with all predictors.
3. Find the predictor with the highest p-value. If p > SL, go to step 4. Otherwise, the model is ready (FIN).
4. Remove that predictor.
5. Refit the model without the removed variable. Go back to step 3.

This starts with everything and removes one variable at a time until all remaining predictors are significant.

#### Forward Selection

1. Select a significance level to enter the model (e.g., SL = 0.05).
2. Fit all simple regression models ($y \sim X_n$). Select the predictor with the lowest p-value.
3. Keep this variable and fit all possible models with one extra predictor added to the current set.
4. Find the predictor with the lowest p-value among the new additions. If p < SL, go to step 3. Otherwise, the model is ready (FIN — keep the previous model, not the last one tested).

This starts with nothing and adds one variable at a time until no more significant predictors can be added.

#### Bidirectional Elimination

1. Select two significance levels: SL_ENTER (to enter, e.g., 0.05) and SL_STAY (to stay, e.g., 0.05).
2. Perform the next step of Forward Selection (new variables must have p < SL_ENTER to enter).
3. Perform all steps of Backward Elimination (existing variables must have p < SL_STAY to stay).
4. Repeat steps 2-3 until no new variables can enter and no existing variables can exit. The model is ready (FIN).

This combines both methods to prevent variables that were added early from remaining in the model when they become insignificant after other variables are added.

#### All Possible Models (Score Comparison)

1. Select a criterion of goodness of fit (e.g., Akaike Information Criterion, $R^2_{adj}$, BIC).
2. Construct all possible regression models: $2^N - 1$ total combinations for N predictors.
3. Select the model with the best criterion score.

This is the most thorough approach but computationally expensive. With 10 predictors, there are 1,023 possible models. With 20 predictors, over 1 million.

> **Note:** Backward Elimination, Forward Selection, and Bidirectional Elimination are collectively known as Stepwise Regression. In practice, Backward Elimination is the most commonly used because it is fast and straightforward.

### Sample Dataset

```csv
R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.17,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
78013.11,121597.55,264346.06,California,126992.93
94657.16,145077.58,282574.31,New York,125370.37
91749.16,114175.79,294919.57,Florida,124266.9
86419.7,153514.11,0,New York,122776.86
76253.86,113867.3,298664.47,California,118474.03
78389.47,153773.43,299737.29,New York,111313.02
73994.56,122782.75,303319.26,Florida,110352.25
67532.53,105751.03,304768.73,Florida,108733.99
77044.01,99281.34,140574.81,New York,108552.04
64664.71,139553.16,137962.62,California,107404.34
75328.87,144135.98,134050.07,Florida,105733.54
72107.6,127864.55,353183.81,New York,105008.31
66051.52,182645.56,118148.2,Florida,103282.38
65605.48,153032.06,107138.38,New York,101004.64
61994.48,115641.28,91131.24,Florida,99937.59
61136.38,152701.92,88218.23,New York,97483.56
63408.86,129219.61,46085.25,California,97427.84
55493.95,103057.49,214634.81,Florida,96778.92
46426.07,157693.92,210797.67,California,96712.8
46014.02,85047.44,205517.64,New York,96479.51
28663.76,127056.21,201126.82,Florida,90708.19
44069.95,51283.14,197029.42,California,89949.14
20229.59,65947.93,185265.1,New York,81229.06
38558.51,82982.09,174999.3,California,81005.76
28754.33,118546.05,172795.67,California,78239.91
27892.92,84710.77,164470.71,Florida,77798.83
23640.93,96189.63,148001.11,California,71498.49
15505.73,127382.3,35534.17,New York,69758.98
22177.74,154806.14,28334.72,California,65200.33
1000.23,124153.04,1903.93,New York,64926.08
1315.46,115816.21,297114.46,Florida,49490.75
0,135426.92,0,California,42559.73
542.05,51743.15,0,New York,35673.41
0,116983.8,45173.06,California,14681.4
```

### Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Importing the Dataset

```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

`iloc[:, :-1]` selects the first four columns (R&D Spend, Administration, Marketing Spend, State) as features. `iloc[:, -1]` selects the last column (Profit) as the dependent variable.

### Encoding Categorical Data

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

`[3]` targets column index 3 (State). `OneHotEncoder()` converts the three states (California, Florida, New York) into three binary columns. `remainder='passthrough'` keeps R&D Spend, Administration, and Marketing Spend unchanged.

You do not need to manually avoid the dummy variable trap here. The `LinearRegression` class in scikit-learn handles multicollinearity internally, so including all dummy columns does not cause issues.

### Splitting the Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

`test_size=0.2` gives 40 training samples and 10 test samples from the 50 startups.

### Training the Model

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

`LinearRegression` automatically handles multiple features. `fit()` computes $b_0, b_1, \dots, b_n$ by minimizing the sum of squared residuals across all features simultaneously. There is no need for feature scaling or manual feature selection; the class handles everything internally.

### Predicting Results

```python
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

```text
[[103015.2  103282.38]
 [132582.28 144259.4 ]
 [132447.74 146121.95]
 [  71976.1   77798.83]
 [ 178537.48 191050.39]
 [ 116161.24 105008.31]
 [  67851.69  81229.06]
 [  98791.73  97483.56]
 [ 113969.44 110352.25]
 [ 167921.07 166187.94]]
```

`np.set_printoptions(precision=2)` limits NumPy output to 2 decimal places for readability. `reshape(len(y_pred), 1)` converts each 1D array into a column vector. `np.concatenate(..., 1)` joins the predicted and actual values side by side (axis=1) for easy comparison. The left column is the predicted profit and the right column is the actual profit.

### Single Prediction

```python
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
```

```text
[181566.92]
```

The model predicts a profit of approximately `$181,567` for a startup in California (encoded as `[1, 0, 0]`) with R&D spend of `$160,000`, admin spend of `$130,000`, and marketing spend of `$300,000`. The dummy variables come first because `ColumnTransformer` placed the encoded columns before the passthrough columns.

### Extracting the Equation

```python
print(regressor.coef_)
print(regressor.intercept_)
```

```text
[ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
42467.53
```

`coef_` returns one coefficient per feature: three for the dummy variables and three for the numerical features. `intercept_` returns $b_0$. The coefficients show that R&D Spend (`7.73e-01`) has the largest effect on profit, while Administration (`3.29e-02`) and Marketing Spend (`3.66e-02`) have much smaller effects. This means for every dollar increase in R&D spend, profit increases by approximately $0.77, holding all other variables constant.
