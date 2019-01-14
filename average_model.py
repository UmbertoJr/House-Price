import pandas as pd
import numpy as np
from time import time
from scipy.stats import skew
from scipy.special import boxcox1p
from multiprocessing import Pool

start = time()

train = pd.read_csv('./train.csv', index_col=0)
test = pd.read_csv('./test.csv', index_col=0 )
test_id = test.index

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train["SalePrice"] = np.log1p(train["SalePrice"])


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


### some outliers
all_data.FullBath.replace({4:3}, inplace=True)

all_data.MiscFeature.replace({"TenC":"Gar2"}, inplace=True)

## rounding features
all_data.BsmtFinSF1 = all_data.BsmtFinSF1.apply(lambda x: np.round(x, -2))

all_data.LowQualFinSF = all_data.LowQualFinSF.apply(lambda x: np.round(x, -2))

all_data.WoodDeckSF = all_data.WoodDeckSF.apply(lambda x: np.round(x, -1))


## replace NA
all_data["PoolQC"].fillna("None", inplace= True)
all_data["MiscFeature"].fillna("None", inplace= True)
all_data["Alley"].fillna("None", inplace= True)
all_data.drop('Fence', axis=1, inplace=True)

all_data.GarageQual.replace({"Ex":5, "Gd":4,"TA":3,"Fa":2,"Po":1}, inplace=True)
all_data.GarageQual.fillna(0, inplace=True)
all_data.GarageQual = all_data.GarageQual.astype(np.int16)


all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


## Neighborhood replacement
all_data.Neighborhood.replace({"Blueste":"Sawyer",
                            "Blmngtn": "Gilbert"                            
                           }, inplace = True)


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col].fillna('None', inplace= True)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col].fillna(0, inplace= True)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col].fillna(0, inplace= True)
for col in ('BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col].fillna('None', inplace= True)
    
all_data["MasVnrType"].fillna("None", inplace=True)
all_data["MasVnrArea"].fillna(0, inplace=True)
all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0], inplace=True)

all_data.drop(['Utilities'], axis=1, inplace=True)

all_data["Functional"].fillna("Typ", inplace=True)
all_data['Electrical'].fillna(all_data['Electrical'].mode()[0], inplace=True)
all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0], inplace=True)
all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0], inplace=True)
all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0], inplace=True)
all_data['SaleType'].fillna(all_data['SaleType'].mode()[0], inplace=True)
all_data['MSSubClass'].fillna("None", inplace=True)


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


all_data_for_lasso = all_data.copy()

################################## BOOST based models data
from sklearn.preprocessing import LabelEncoder
cols =( 'BsmtFinType1', 'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',   
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


######################## LASSO based models data
from sklearn.preprocessing import LabelEncoder
cols =['BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'LandSlope','LotShape',
       'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','YrSold','MoSold']
       
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data_for_lasso[c].values)) 
    all_data_for_lasso[c] = lbl.transform(list(all_data_for_lasso[c].values))

    
def build_train_test(all_data):
  numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
  # Check the skew of all numerical features
  skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

  skewness = pd.DataFrame({'Skew' :skewed_feats})
  skewness.head(10)
  skewness = skewness[abs(skewness) > 0.75]
  skewed_features = skewness.index
  lam = 0.19       #0.15
  for feat in skewed_features:
      all_data[feat] = boxcox1p(all_data[feat], lam)
      
  all_data = pd.get_dummies(all_data)
  for c in all_data.columns:
    if all_data[c].dtype != "float" and all_data[c].sum()< 10:
      all_data.drop(c, axis=1, inplace=True)
      
  global ntrain
  train = all_data[:ntrain]
  test = all_data[ntrain:]
  return train, test

train, test = build_train_test(all_data)
n_col = train.shape[1]

train_lasso, test_lasso = build_train_test(all_data_for_lasso)
n_col_lasso = train_lasso.shape[1]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLars, HuberRegressor, LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error





#Validation function
n_folds = 8


def rmsle_cv(model):
  X = [train.values, train_lasso.values]
  kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X[model.data])
  rmse= np.sqrt(-cross_val_score(model,X[model.data] , y_train, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))
  return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.data = 1

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.data = 1

BR = make_pipeline(RobustScaler(), BayesianRidge())
BR.data = 1

LL = make_pipeline(RobustScaler(), LassoLars(alpha=0.0001))
LL.data = 0

HR = HuberRegressor(epsilon=1., max_iter=300)
HR.data = 0

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.data = 0



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        self.data_mod = [m.data for m in self.models]
        self.X = [train.values, train_lasso.values]
        self.y = y_train
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
      ####dare in imput X = [train.values, train_lasso.values]
      self.models_ = [clone(x) for x in self.models]

      # Train cloned base models
      for model, i  in zip(self.models_, self.data_mod):
          model.fit(X[i], y)

      return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X[i]) for model, i in zip(self.models_, self.data_mod)
        ])
        return np.mean(predictions, axis=1)   
    
    def CV(self):
      global n_folds
      kf = KFold(n_folds, shuffle=True, random_state=42)
      scores = []
      for train_index, test_index in kf.split(train.values):
            X_train_0, X_test_0 = self.X[0][train_index], self.X[0][test_index]
            X_train_1, X_test_1 = self.X[1][train_index], self.X[1][test_index]
            self.fit([X_train_0,X_train_1], self.y[train_index])
            scores.append(np.sqrt(mean_squared_error(self.y[test_index], self.predict([X_test_0,X_test_1]) )))
      return scores
    
averaged_model_1 = AveragingModels(models = (ENet , GBoost, HR))
averaged_model_1.data = ":"


model = averaged_model_1
model.fit([train.values, train_lasso.values], y_train)
pred = np.expm1(model.predict([test.values, test_lasso.values]))

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = pred

name = "average_model.csv"
sub.to_csv(name, index=False)




print("Execution time : "+ str(time() - start))