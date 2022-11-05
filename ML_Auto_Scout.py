#   Tasks

### 1. Import Modules, Load Data and Data Review
### 2. Data Pre-Processing
### 3. Implement Linear Regression 
### 4. Implement Ridge Regression
### 5. Implement Lasso Regression 
### 6. Implement Elastic-Net
### 7. Visually Compare Models Performance In a Graph


## 1. Import Modules, Load Data and Data Review
import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from scipy.stats import skew

from sklearn.model_selection import cross_validate, cross_val_score
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500)


# Read Data
df = pd.read_csv("Auto_scout_dataset.csv")
df.head()


# Info about dataset
df.info()
df.describe().T
df.columns


# We copy and store the original dataset as df2.
df2 = df.copy()


## Feature Engineering
df.select_dtypes(include ="object").head()


## We check dummies control.
for col in df.select_dtypes('object'):
    print(f"{col:<20}:", df[col].nunique())
    
df.make_model.value_counts()

ax = df.make_model.value_counts().plot(kind ="bar")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.axis("off")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.03, p.get_height() * 1.03))


df[df.make_model=="Audi A2"]
df.drop(index=[2614], inplace =True)
df.shape


sns.histplot(df.price, bins=50, kde=True)


## We check the skewness of dataset.
skew(df.price)

df_numeric = df.select_dtypes(include ="number")
df_numeric

# We perform heatmap for correlations between features.
sns.heatmap(df_numeric.corr(), annot =True)


## Multicollinearity Control
df_numeric.corr()[(df_numeric.corr()>= 0.9) & (df_numeric.corr() < 1)].any().any()
df_numeric.corr()[(df_numeric.corr()<= -0.9) & (df_numeric.corr() > -1)].any().any()

# For outliers' check.
sns.boxplot(df.price);

plt.figure(figsize=(16,6))
sns.boxplot(x="make_model", y="price", data=df, whis=3)
plt.show()


## Get dummies 
df = df.join(df["Comfort_Convenience"].str.get_dummies(sep = ",").add_prefix("cc_"))
df = df.join(df["Entertainment_Media"].str.get_dummies(sep = ",").add_prefix("em_"))
df = df.join(df["Extras"].str.get_dummies(sep = ",").add_prefix("ex_"))
df = df.join(df["Safety_Security"].str.get_dummies(sep = ",").add_prefix("ss_"))

df.drop(["Comfort_Convenience","Entertainment_Media","Extras","Safety_Security"], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first =True)

df.head()

df.shape

df.isnull().any().any()


# Correlation parameters with respect to price.
corr_by_price = df.corr()["price"].sort_values()[:-1]
corr_by_price

# We plot the correlations.
plt.figure(figsize = (20,10))
sns.barplot(x = corr_by_price.index, y = corr_by_price)
plt.xticks(rotation=90)
plt.tight_layout()


## 2. Data Pre-Processing
# Train | Test Split
# random_state = 101
# test size = 0.2

# Features (X) and target variable (Y).
X = df.drop(columns="price")
y = df.price

# We train the train dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train.shape
X_test.shape


## 3. Implement Linear Regression
# Import the modul
# Fit the model 
# Predict the test set
# Determine feature coefficiant
# Evaluate model performance (use performance metrics for regression and cross_val_score)
# Compare different evaluation metrics


# We define a function for calculating scores.
def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# We use Linear Regression to train.
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

pd.options.display.float_format = '{:.3f}'.format

train_val(lm, X_train, y_train, X_test, y_test)



## Adjusted R2 Score

# We define a function for adjusted R2 score.
def adj_r2(y_test, y_pred, df):
    r2 = r2_score(y_test, y_pred)
    n = df.shape[0]   # number of observations
    p = df.shape[1]-1 # number of independent variables 
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2

y_pred = lm.predict(X_test)

adj_r2(y_test, y_pred, df)



## Cross Validation

model = LinearRegression()
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 10)

# We show the scores as dataframe.
pd.DataFrame(scores)

# We calculate mean values over cv = 10.
pd.DataFrame(scores).iloc[:, 2:].mean()

train_val(lm, X_train, y_train, X_test, y_test)



## Prediction Error

from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz

visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()


# Residual Plot
plt.figure(figsize=(12,8))
residuals = y_test-y_pred

sns.scatterplot(x = y_test, y = -residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()

from yellowbrick.regressor import ResidualsPlot

visualizer = RadViz(size=(1000, 720))
model = LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()



## Dropping observations from the dataset that worsen my predictions

# test size = 0.2

df3 = df[~(df.price>35000)]
df3

len(df[df.price>35000])

df2[df2.price>35000].groupby("make_model").count().iloc[:,0]

df2.make_model.value_counts()

X = df3.drop(columns = "price")
y = df3.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 101)

lm2 = LinearRegression()
lm2.fit(X_train,y_train)

visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();

train_val(lm2, X_train, y_train, X_test, y_test)


y_pred = lm2.predict(X_test)

lm_R2 = r2_score(y_test, y_pred)
lm_mae = mean_absolute_error(y_test, y_pred)
lm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

my_dict = { 'Actual': y_test, 'Pred': y_pred, 'Residual': y_test-y_pred }
compare = pd.DataFrame(my_dict)

comp_sample = compare.sample(20)
comp_sample

comp_sample.plot(kind='bar',figsize=(15,9))
plt.show()

pd.DataFrame(lm2.coef_, index = X.columns, columns=["Coef"]).sort_values("Coef")



## 4. Implement Ridge Regression

# Import the modul 
# Do not forget to scale the data or use Normalize parameter as True 
# Fit the model 
# Predict the test set 
# Evaluate model performance (use performance metrics for regression) 
# Tune alpha hiperparameter by using [cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) and determine the optimal alpha value.
# Fit the model and predict again with the new alpha value. 


# Scaling.

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



## Ridge

# Random state = 42.

from sklearn.linear_model import Ridge, RidgeCV
ridge_model = Ridge(random_state = 42)
ridge_model.fit(X_train_scaled, y_train)

train_val(ridge_model, X_train_scaled, y_train, X_test_scaled, y_test)



## Finding best alpha for Ridge

from sklearn.model_selection import GridSearchCV

alpha_space = np.linspace(0.01, 100, 100)
alpha_space

ridge_model = Ridge(random_state=42)

param_grid = {'alpha':alpha_space}

ridge_grid_model = GridSearchCV(estimator=ridge_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)


# We train our model by implementing ridge grid model.
ridge_grid_model.fit(X_train_scaled,y_train)

# We find best parameters.
ridge_grid_model.best_params_

# We write the cv-based results as dataframe.
pd.DataFrame(ridge_grid_model.cv_results_)

# We find best index value.
ridge_grid_model.best_index_

# We find best score.
ridge_grid_model.best_score_

train_val(ridge_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

y_pred = ridge_grid_model.predict(X_test_scaled)
rm_R2 = r2_score(y_test, y_pred)
rm_mae = mean_absolute_error(y_test, y_pred)
rm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

ridge = Ridge(alpha=1.02, random_state=42).fit(X_train_scaled, y_train)

pd.DataFrame(ridge.coef_, index = X.columns, columns=["Coef"]).sort_values("Coef")



## 5. Implement Lasso Regression

# Import the modul 
# Do not forget to scale the data or use Normalize parameter as True(If needed)
# Fit the model 
# Predict the test set 
# Evaluate model performance (use performance metrics for regression) 
# Tune alpha hyperparameter by using [cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) and determine the optimal alpha value.
# Fit the model and predict again with the new alpha value.
# Compare different evaluation metrics

# Note: To understand the importance of the alpha hyperparameter, you can observe the effects of different alpha values on feature coefficants.*

# random state = 42.

from sklearn.linear_model import Lasso, LassoCV

lasso_model = Lasso(random_state = 42)
lasso_model.fit(X_train_scaled, y_train)

train_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)



## Finding best alpha for Lasso.

lasso_model = Lasso(random_state=42)

param_grid = {'alpha':alpha_space}

lasso_grid_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

# We train our model by implementing Lasso grid model.
lasso_grid_model.fit(X_train_scaled,y_train)

# Best parameters.
lasso_grid_model.best_params_

# Best score.
lasso_grid_model.best_score_

train_val(lasso_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

y_pred = lasso_grid_model.predict(X_test_scaled)
lasm_R2 = r2_score(y_test, y_pred)
lasm_mae = mean_absolute_error(y_test, y_pred)
lasm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

lasso = Lasso(alpha=1.02, random_state=42).fit(X_train_scaled, y_train)
pd.DataFrame(lasso.coef_, index = X.columns, columns=["Coef"]).sort_values("Coef")



## 6. Implement Elastic-Net

# Import the modul 
# Do not forget to scale the data or use Normalize parameter as True(If needed)
# Fit the model 
# Predict the test set 
# Evaluate model performance (use performance metrics for regression) 
# Tune alpha hyperparameter by using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and determine the optimal alpha value.
# Fit the model and predict again with the new alpha value.
# Compare different evaluation metrics

# random state = 42.

from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(random_state=42)
elastic_model.fit(X_train_scaled,y_train)

train_val(elastic_model, X_train_scaled, y_train, X_test_scaled, y_test)



## Finding best alpha and l1_ratio for ElasticNet

# random state = 42.

elastic_model = ElasticNet(random_state=42)

param_grid = {'alpha':[1.02, 2,  3, 4, 5, 7, 10, 11],
              'l1_ratio':[.5, .7, .9, .95, .99, 1]}

elastic_grid_model = GridSearchCV(estimator=elastic_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

# We train our model by implementing elastic grid model.
elastic_grid_model.fit(X_train_scaled,y_train)

# Best parameter values.
elastic_grid_model.best_params_

# Best score.
elastic_grid_model.best_score_

train_val(elastic_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

y_pred = elastic_grid_model.predict(X_test_scaled)
em_R2 = r2_score(y_test, y_pred)
em_mae = mean_absolute_error(y_test, y_pred)
em_rmse = np.sqrt(mean_squared_error(y_test, y_pred))



## Feature Importance

from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz


viz = FeatureImportances(Lasso(alpha=1.02), labels=X_train.columns)
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train_scaled, y_train)
viz.show()

df_new = df2[["make_model", "hp_kW", "km","age", "Gearing_Type", "price"]]
df_new.head()

df_new[df_new["make_model"] == "Audi A2"]

# We drop Audi A2 from dataset.
df_new.drop(index=[2614], inplace =True)

df_new = df_new[~(df_new.price>35000)]

df_new = pd.get_dummies(df_new)
df_new

len(df_new)

X = df_new.drop(columns = ["price"])
y = df_new.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_model = Lasso(random_state=42)

param_grid = {'alpha':alpha_space}

lasso_final_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

lasso_final_model.fit(X_train_scaled, y_train)

lasso_final_model.best_params_

lasso_final_model.best_score_

train_val(lasso_final_model, X_train_scaled, y_train, X_test_scaled, y_test)

y_pred = lasso_final_model.predict(X_test_scaled)
fm_R2 = r2_score(y_test, y_pred)
fm_mae = mean_absolute_error(y_test, y_pred)
fm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))



## 7. Visually Compare Models Performance In a Graph

scores = {"linear_m": {"r2_score": lm_R2, 
 "mae": lm_mae, 
 "rmse": lm_rmse},

 "ridge_m": {"r2_score": rm_R2, 
 "mae": rm_mae,
 "rmse": rm_rmse},
    
 "lasso_m": {"r2_score": lasm_R2, 
 "mae": lasm_mae, 
 "rmse": lasm_rmse},

 "elastic_m": {"r2_score": em_R2, 
 "mae": em_mae, 
 "rmse": em_rmse},
         
 "final_m": {"r2_score": fm_R2, 
 "mae": fm_mae , 
 "rmse": fm_rmse}}
scores = pd.DataFrame(scores).T
scores


#metrics = scores.columns
for i, j in enumerate(scores):
    plt.figure(i)
    if j == "r2_score":
        ascending = False
    else:
        ascending = True
    compare = scores.sort_values(by=j, ascending=ascending)
    ax = sns.barplot(x = compare[j] , y= compare.index)
    for p in ax.patches:
            width = p.get_width()                        # get bar length
            ax.text(width,                               # set the text at 1 unit right of the bar
                    p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                    '{:.4f}'.format(width),             # set variable to display, 2 decimals
                    ha = 'left',                         # horizontal alignment
                    va = 'center') 
            


## Prediction new observation

# random state = 42.

final_scaler = MinMaxScaler()
final_scaler.fit(X)
X_scaled = final_scaler.transform(X)

lasso_model = Lasso(random_state=42)

param_grid = {'alpha':alpha_space}

final_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

final_model.fit(X_scaled,y)

final_model.best_estimator_

my_dict = {
    "hp_kW": 66,
    "age": 2,
    "km": 17000,
    "make_model": 'Audi A3',
    "Gearing_Type": "Automatic"
}

my_dict = pd.DataFrame([my_dict])
my_dict

my_dict = pd.get_dummies(my_dict)
my_dict

X.head(1)

my_dict = my_dict.reindex(columns = X.columns, fill_value=0)
my_dict

my_dict = final_scaler.transform(my_dict)
my_dict

final_model.predict(my_dict)