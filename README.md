# DL_Pipeline


In [1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import talos as ta

Using TensorFlow backend.

In [2]:

import warnings
warnings.filterwarnings('ignore')

In [3]:

# Always good to set a seed for reproducibility
SEED = 7
np.random.seed(SEED)

In [4]:

# Loading Data
df = pd.read_csv('diabetes.csv')
# Getting dataframe columns names
df_name=df.columns
df.head()

Out[4]:
	Pregnancies 	Glucose 	BloodPressure 	SkinThickness 	Insulin 	BMI 	DiabetesPedigreeFunction 	Age 	Outcome
0 	6 	148 	72 	35 	0 	33.6 	0.627 	50 	1
1 	1 	85 	66 	29 	0 	26.6 	0.351 	31 	0
2 	8 	183 	64 	0 	0 	23.3 	0.672 	32 	1
3 	1 	89 	66 	23 	94 	28.1 	0.167 	21 	0
4 	0 	137 	40 	35 	168 	43.1 	2.288 	33 	1
In [5]:

df.isna().sum()

Out[5]:

Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64

In [6]:

df.describe()

Out[6]:
	Pregnancies 	Glucose 	BloodPressure 	SkinThickness 	Insulin 	BMI 	DiabetesPedigreeFunction 	Age 	Outcome
count 	768.000000 	768.000000 	768.000000 	768.000000 	768.000000 	768.000000 	768.000000 	768.000000 	768.000000
mean 	3.845052 	120.894531 	69.105469 	20.536458 	79.799479 	31.992578 	0.471876 	33.240885 	0.348958
std 	3.369578 	31.972618 	19.355807 	15.952218 	115.244002 	7.884160 	0.331329 	11.760232 	0.476951
min 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.078000 	21.000000 	0.000000
25% 	1.000000 	99.000000 	62.000000 	0.000000 	0.000000 	27.300000 	0.243750 	24.000000 	0.000000
50% 	3.000000 	117.000000 	72.000000 	23.000000 	30.500000 	32.000000 	0.372500 	29.000000 	0.000000
75% 	6.000000 	140.250000 	80.000000 	32.000000 	127.250000 	36.600000 	0.626250 	41.000000 	1.000000
max 	17.000000 	199.000000 	122.000000 	99.000000 	846.000000 	67.100000 	2.420000 	81.000000 	1.000000
In [7]:

sns.pairplot(df, hue='Outcome', palette = 'husl')
plt.show()

In [8]:

def histogram_plot(col_name):
    plt.figure(figsize=(10,7), dpi = 180)
    sns.distplot(df[col_name], color="dodgerblue",\
                  kde_kws={"color": "b", "lw": 1, "label": "KDE"},\
                  hist_kws={"linewidth": 3})  
    plt.legend()
    plt.show()

In [9]:

histogram_plot('Pregnancies')
histogram_plot('Glucose')
histogram_plot('SkinThickness')
histogram_plot('BMI')
histogram_plot('BloodPressure')
histogram_plot('DiabetesPedigreeFunction')

In [10]:

def box_plot(col_name):
    plt.figure(figsize=(1,3), dpi = 180)
    sns.boxplot(df[col_name], orient = 'v', color = 'black', fliersize = 0.5, linewidth = 0.5)
    plt.show()

In [11]:

box_plot('Pregnancies')
box_plot('Glucose')
box_plot('SkinThickness')
box_plot('BMI')
box_plot('BloodPressure')
box_plot('DiabetesPedigreeFunction')

In [12]:

# Dropping outliers
z_scores = stats.zscore(df)
print(z_scores)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis = 1)
clean_df = df[filtered_entries]
print(clean_df)

[[ 0.63994726  0.84832379  0.14964075 ...  0.46849198  1.4259954
   1.36589591]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.36506078 -0.19067191
  -0.73212021]
 [ 1.23388019  1.94372388 -0.26394125 ...  0.60439732 -0.10558415
   1.36589591]
 ...
 [ 0.3429808   0.00330087  0.14964075 ... -0.68519336 -0.27575966
  -0.73212021]
 [-0.84488505  0.1597866  -0.47073225 ... -0.37110101  1.17073215
   1.36589591]
 [-0.84488505 -0.8730192   0.04624525 ... -0.47378505 -0.87137393
  -0.73212021]]
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0              6      148             72             35        0  33.6   
1              1       85             66             29        0  26.6   
2              8      183             64              0        0  23.3   
3              1       89             66             23       94  28.1   
5              5      116             74              0        0  25.6   
..           ...      ...            ...            ...      ...   ...   
763           10      101             76             48      180  32.9   
764            2      122             70             27        0  36.8   
765            5      121             72             23      112  26.2   
766            1      126             60              0        0  30.1   
767            1       93             70             31        0  30.4   

     DiabetesPedigreeFunction  Age  Outcome  
0                       0.627   50        1  
1                       0.351   31        0  
2                       0.672   32        1  
3                       0.167   21        0  
5                       0.201   30        0  
..                        ...  ...      ...  
763                     0.171   63        0  
764                     0.340   27        0  
765                     0.245   30        0  
766                     0.349   47        1  
767                     0.315   23        0  

[688 rows x 9 columns]

In [13]:

df.head()

Out[13]:
	Pregnancies 	Glucose 	BloodPressure 	SkinThickness 	Insulin 	BMI 	DiabetesPedigreeFunction 	Age 	Outcome
0 	6 	148 	72 	35 	0 	33.6 	0.627 	50 	1
1 	1 	85 	66 	29 	0 	26.6 	0.351 	31 	0
2 	8 	183 	64 	0 	0 	23.3 	0.672 	32 	1
3 	1 	89 	66 	23 	94 	28.1 	0.167 	21 	0
4 	0 	137 	40 	35 	168 	43.1 	2.288 	33 	1
In [14]:

from sklearn.preprocessing import MinMaxScaler
clean_df_name = clean_df.columns
clean_df.to_csv('clean.csv', index = False)
clean_df = pd.read_csv('clean.csv')

for i in range(0, len(clean_df_name) - 1):
    clean_df[clean_df_name[i]] = MinMaxScaler().fit_transform(clean_df[clean_df_name[i]]\
                                                              .values.reshape(-1,1))


X_c =  clean_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y_c = clean_df[['Outcome']]
X_train_c, X_test_c, y_train_c, y_test_c =train_test_split(X_c,Y_c,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=clean_df['Outcome'])
print(X_train_c)
print(y_train_c)

     Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  \
228     0.230769  0.432258       0.326531       0.650000  0.000000  0.323370   
101     0.384615  0.516129       0.510204       0.000000  0.000000  0.429348   
651     0.230769  0.554839       0.551020       0.383333  0.190361  0.277174   
423     0.538462  0.451613       0.530612       0.283333  0.265060  0.152174   
153     0.076923  0.225806       0.367347       0.700000  0.115663  0.687500   
..           ...       ...            ...            ...       ...       ...   
118     0.153846  0.335484       0.448980       0.216667  0.118072  0.078804   
324     0.384615  0.354839       0.306122       0.466667  0.200000  0.429348   
561     0.538462  0.451613       0.408163       0.000000  0.000000  0.250000   
345     0.384615  0.645161       0.591837       0.433333  0.686747  0.375000   
489     0.076923  0.464516       0.469388       0.466667  0.000000  0.250000   

     DiabetesPedigreeFunction       Age  
228                  0.346349  0.191489  
101                  0.102675  0.361702  
651                  0.177151  0.276596  
423                  0.280550  0.212766  
153                  0.433839  0.042553  
..                        ...       ...  
118                  0.411424  0.106383  
324                  0.304411  0.191489  
561                  0.472885  0.276596  
345                  0.270427  0.787234  
489                  0.091106  0.000000  

[516 rows x 8 columns]
     Outcome
228        0
101        1
651        1
423        0
153        0
..       ...
118        0
324        0
561        1
345        1
489        0

[516 rows x 1 columns]

In [15]:

print(np.shape(X_train_c ))
print(np.shape(y_train_c ))

(516, 8)
(516, 1)

In [16]:

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend as K

def create_model(optimizer="adam", dropout=0.1, 
                 init='uniform', dense_nparams=8):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(8,), kernel_initializer=init,)) 
    model.add(Dropout(dropout), )
#     for layer_size in dense_layer_sizes:
    model.add(Dense(dense_nparams, activation='relu'))
    model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model

In [17]:

batch_size = [8, 16, 32, 64]
epochs = [10, 50, 100]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
              'hard_sigmoid', 'linear']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
neurons = [1, 5, 10, 15, 20, 25, 30]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

In [18]:

param_grid = {
#     'dense_layer_sizes': [(8), (16), (8,16), (16,32)],
#     'tfidf__use_idf': [True, False],
    'epochs': [10], # , 100, 
#     'dense_nparams': [32], # , 256, 512
    'init': [ 'uniform' ], # , 'zeros', 'normal' 
    'batch_size':[2, 16, 32],
    'optimizer':['Adam'], # 'Adamax', 'sgd', 'RMSprop'
    'dropout': [0.5] # , 0.4, 0.3, 0.2, 0.1, 0
}

In [19]:

p_grid = {
#     'dense_layer_sizes': [(32),(64), (32, 32), (64, 64)],
    'optimizer': optimizer,
    'learn_rate': learn_rate,
    'activation': activation,
    'dropout_rate': dropout_rate,
}

In [20]:

kfold_splits = 2
keras_estimator = KerasClassifier(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=keras_estimator,  
                    n_jobs=-1, 
                    verbose=1,
                    return_train_score=True,
                    cv=kfold_splits,  #StratifiedKFold(n_splits=kfold_splits, shuffle=True)
                    param_grid=param_grid)

grid_result = grid.fit(X_train_c, y_train_c, ) #callbacks=[tbCallBack]

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

Fitting 2 folds for each of 3 candidates, totalling 6 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.

[Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:   10.6s finished

Epoch 1/10
516/516 [==============================] - 1s 1ms/step - loss: 0.6791 - acc: 0.6492
Epoch 2/10
516/516 [==============================] - 0s 524us/step - loss: 0.6623 - acc: 0.6705
Epoch 3/10
516/516 [==============================] - 0s 588us/step - loss: 0.6533 - acc: 0.6705
Epoch 4/10
516/516 [==============================] - 0s 665us/step - loss: 0.6425 - acc: 0.6705
Epoch 5/10
516/516 [==============================] - 0s 613us/step - loss: 0.6366 - acc: 0.6705
Epoch 6/10
516/516 [==============================] - 0s 605us/step - loss: 0.6309 - acc: 0.6705
Epoch 7/10
516/516 [==============================] - 0s 520us/step - loss: 0.6341 - acc: 0.6705
Epoch 8/10
516/516 [==============================] - 0s 497us/step - loss: 0.6313 - acc: 0.6705
Epoch 9/10
516/516 [==============================] - 0s 574us/step - loss: 0.6281 - acc: 0.6705
Epoch 10/10
516/516 [==============================] - 0s 582us/step - loss: 0.6273 - acc: 0.6705
Best: 0.670543 using {'batch_size': 2, 'dropout': 0.5, 'epochs': 10, 'init': 'uniform', 'optimizer': 'Adam'}
0.670543 (0.034884) with: {'batch_size': 2, 'dropout': 0.5, 'epochs': 10, 'init': 'uniform', 'optimizer': 'Adam'}
0.670543 (0.034884) with: {'batch_size': 16, 'dropout': 0.5, 'epochs': 10, 'init': 'uniform', 'optimizer': 'Adam'}
0.670543 (0.034884) with: {'batch_size': 32, 'dropout': 0.5, 'epochs': 10, 'init': 'uniform', 'optimizer': 'Adam'}


