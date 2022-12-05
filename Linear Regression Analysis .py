import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as smf
from statsmodels.stats.stattools import durbin_watson
from scipy.interpolate import *
from scipy.stats import ttest_rel
import seaborn as sns
data = pd.read_csv('/Users/macbookpro/Desktop/collector.csv', index_col=0)
data.head()
data.describe()
eta_data = data['eta_data']
T_red = data['T_red']
plt.rcParams['figure.figsize'] = (15,10)
plt.scatter(data['T_red'],data['eta_data'], c=T_red, cmap='hsv')
cbar = plt.colorbar()
cbar.set_label('Efficiency/Temperature Drop Ratio', fontsize=12)
plt.xlabel('T_red', fontsize=15)
plt.ylabel('eta_data', fontsize=15)
plt.title('Solar Collector Efficiency vs Temperature Drop', fontsize=17)
plt.show()
data.dtypes
data.isnull().sum()
data.duplicated('T_red')
data.duplicated('eta_data')
correlations = data.corr()
sns.heatmap(correlations)
plt.show()
intercept = smf.add_constant(T_red)
model = smf.OLS(data['eta_data'], intercept)
model1 = model.fit()
model_predict = model1.predict()
model1.summary()
dir(model1)
print('F-value of First Model:', model1.fvalue)
polynomial_features = PolynomialFeatures(degree=2)
pf = polynomial_fea.fit_transform(intercept)
model = smf.OLS(data['eta_data'], pf)
model2 = model.fit()
model2_predict = model2.predict()
model2.summary()
dir(model2)
print('F-value of Second Model:', model2.fvalue)
print('P-value of First Model:', model1.f_pvalue)
print('Zero Loss Efficiency: '+str(model1.params[0]))
print('Linear Coefficient: '+str(model1.params[1]))
print('P-value of Second Model:', model2.f_pvalue)
print('Zero Loss Efficiency: '+str(model2.params[0]))
print('Linear Coefficient: '+str(model2.params[2]))
print('Quadratic Coefficient: '+str(model2.params[5]))
n= 40
y = data['eta_data']
x = data['T_red']
p1 = np.polyfit(x,y,1)
yhat1 = p1[1] + p1[0]*x
COD1 = model1.rsquared
CCOD1 = 1-((1-COD1)*((n-1)/(n-2)))
RMSE1 = model1.mse_model
print('The Corrected Coefficient of Determination: '+str(CCOD1))
print('The Root Mean Squared: '+str(RMSE1))
n=40
y = data['eta_data']
x = data['T_red']
p2 = np.polyfit(x,y,2)
yhat2 = p2[2] + p2[1]*x + p2[0]*x**2
COD2 = model2.rsquared
CCOD2 = 1-((1-COD2)*((n-1)/(n-3)))
RMSE2 = model2.mse_model
print('The Corrected Coefficient of Determination: '+str(CCOD2))
print('The Root Mean Squared: '+str(RMSE2))
raw_data = ('data')                                                                                                           
first_model = ('First Model')                                                                                             
second_model = ('Second Model')                                                                                          
plt.xlabel('T_red', fontsize=15)
plt.ylabel('eta_data', fontsize=15)
plt.scatter(x, y, label=raw_data)
plt.title('Polynomial Regression of the First and Second Order', fontsize=19)
plt.plot(x, yhat2, color='g', label=second_model)
plt.plot(x,yhat1, color='r', label=first_model)
plt.legend(prop ={'size': 12.5})
plt.show()
plt.scatter(x, model1.resid, color='b')
plt.xlabel('T_red', fontsize=15)
plt.ylabel('y-yhat1', fontsize=15)
plt.title('The First Model Residuals', fontsize = 17)
plt.show()
plt.scatter(x, model2.resid, color ='r')
plt.xlabel('T_red', fontsize=15)
plt.ylabel('y-yhat2', fontsize=15)
plt.title('The Second Model Residuals', fontsize = 17)
plt.show()
# Please insert your code here
plt.hist(model1.resid, color='b')
plt.title('The First Model Residuals Histogram Plot', fontsize = 17)
plt.xlabel('T_red', fontsize=15)
plt.ylabel('y-yhat1', fontsize=15)
plt.show()
plt.hist(model2.resid, color='r')
plt.xlabel('T_red', fontsize=15)
plt.ylabel('y-yhat1', fontsize=15)
plt.title('The Second Model Residuals Histogram Plot', fontsize = 17)
plt.show()
x = smf.add_constant(x)
model = smf.OLS(y, x).fit()
goldfeld_quandt = smf.stats.diagnostic.het_goldfeldquandt(y, x, drop=0.2, split=0.4, alternative='increasing', store=False)
print('The Test Statistic is: '+str(goldfeld_quandt[0])+
      ' & The Corresponding p-value is: '+str(goldfeld_quandt[1]))
DW1 = (durbin_watson(y-yhat1))
print('The Durbin Watson Statiscal Value of the First Residual: '+str(DW1))
DW2 = (durbin_watson(y-yhat2))
print('The Durbin Watson Statiscal Value of the Second Residual: '+str(DW2))
