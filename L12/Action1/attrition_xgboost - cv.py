import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# 数据加载
train_data_ini = pd.read_csv('./train.csv')

test_data = pd.read_csv('./test.csv')


train_data_ini['Attrition']=train_data_ini['Attrition'].map(lambda x:1 if x=='Yes' else 0)

train_data_positive=train_data_ini[train_data_ini['Attrition'] == 1]
train_data = pd.concat([train_data_ini, train_data_positive], ignore_index=True, sort=False)
train_data = pd.concat([train_data, train_data_positive], ignore_index=True, sort=False)


features=['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

    
train_features = train_data[features]

test_features=test_data[features]


#########################################################################################################################################################################################

attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train_features[feature]=lbe.fit_transform(train_features[feature])
    test_features[feature]=lbe.transform(test_features[feature])
    lbe_list.append(lbe)

    

train_features['combine1']= train_features['OverTime']*train_features['MonthlyIncome']
test_features['combine1']= test_features['OverTime']*test_features['MonthlyIncome']

train_features['combine2']= train_features['OverTime']+train_features['EnvironmentSatisfaction']
test_features['combine2']= test_features['OverTime']+test_features['EnvironmentSatisfaction']

##train_features['combine3']= train_features['EnvironmentSatisfaction']*train_features['MonthlyIncome']
##test_features['combine3']= test_features['EnvironmentSatisfaction']*test_features['MonthlyIncome']

##train_features['combine4']= train_features['MaritalStatus']+train_features['EnvironmentSatisfaction']
##test_features['combine4']= test_features['MaritalStatus']+test_features['EnvironmentSatisfaction']

##train_features['combine5']= train_features['MaritalStatus']+train_features['OverTime']
##test_features['combine5']= test_features['MaritalStatus']+test_features['OverTime']

result=test_data

X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_data['Attrition'], test_size=0.5, random_state=42)

train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(test_features)

##########################################################################################################################################################################################
xgb1 = XGBClassifier(max_depth=2,
                     learning_rate=0.01,
                     n_estimators=800,
                     silent=False,
                     objective='binary:logistic',
                     booster='gbtree',
                     n_jobs=4,
                     gamma=0.3,
                     reg_lambda=0.2,
                     min_child_weight=2,
                     subsample=0.6,
                     colsample_bytree=0.8,
                     seed=7)

##cv_result = xgb.cv(xgb1.get_xgb_params(),
##                   train_data,
##                   num_boost_round=xgb1.get_xgb_params()['n_estimators'],
##                   nfold=5,
##                   metrics='auc',
##                   early_stopping_rounds=50,
##                   callbacks=[xgb.callback.early_stop(50),
##                              xgb.callback.print_evaluation(period=1,show_stdv=True)])


param_grid = {'max_depth':[1,2,3,4,5],
             'min_child_weight':[1,2,3,4,5]}


##param_grid = {'gamma':[1,2,3,4,5,6,7,8,9]}
##param_grid = {'gamma':[i/10.0 for i in range(10,30)]}
##
##param_grid = {'subsample':[i/10.0 for i in range(5,11)],
##             'colsample_bytree':[i/10.0 for i in range(5,11)]}
##
##param_grid = {'reg_lambda':[i/10.0 for i in range(1,11)]}


grid_search = GridSearchCV(xgb1,param_grid,scoring='roc_auc',iid=False,cv=5)
grid_search.fit(X_train,y_train)

print('best_params:',grid_search.best_params_)
print('best_score:',grid_search.best_score_)

######################################################################################################################################################################################
xgb_bst1 = xgb1.fit(X_train, y_train,eval_set=[ (X_train, y_train)], eval_metric='auc',early_stopping_rounds=200)
predict = xgb_bst1.predict_proba(X_valid)[:,1]
result1=metrics.roc_auc_score(y_valid,predict)
print(result1)


##model = xgb.train(xgb1.get_xgb_params(),train_data)
##model.get_score(importance_type='gain')
##xgb.plot_importance(model)
##plt.show()
######################################################################################################################################################################################
predict = xgb_bst1.predict_proba(test_features)

result['Attrition'] = pd.Series(predict[:,1])

result.to_csv("test_result.csv")














