from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)


data=[]
## read csv file
dataframeObject = pd.DataFrame(pd.read_csv(str(sys.argv[1])))

## assigning the features names and putting them in a list
features=list(dataframeObject.columns.values)

for index in range(1,len(features)):
    index_feature=features[index]
    dataframeObject[[index_feature]].replace(np.nan,0)
    data.append(dataframeObject[[index_feature]].to_numpy())

data=np.squeeze(np.array(data))
shape=np.shape(data)

#print(data,shape,'data')

## convert all strings in integer values
for count in range(0,shape[0]):
  possibilities=[]
  data_temp=[]
  if isinstance(data[count,0],str):
    for in_count in range(0,shape[1]):
       if not(data[count,in_count] in possibilities):
          possibilities.append(data[count,in_count])
       index_val = int(possibilities.index(data[count,in_count]))
       data_temp.append(index_val)
    data[count,:]=np.array(data_temp)

shape=np.shape(data)
#print(data,shape,'data_after')

## data definition
DATA=np.transpose(data[0:22,:])
labels=data[22,:]

shape=np.shape(DATA)
#print(DATA,shape,'data_after_after')
#print(np.shape(labels),'labels')
##definition of crossvalidation
kf = KFold(n_splits=int(sys.argv[2]))
kf.get_n_splits(DATA)
acc=np.zeros([int(sys.argv[2])])

KFold(n_splits=int(sys.argv[2]), random_state=None, shuffle=False)

if int(sys.argv[3])==1:
    plt.ion()
    plt.show()

for i, (train_index, test_index) in enumerate(kf.split(DATA)):
        print(f":Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        #D=DATA[train_index,:].astype(float)
        # define the model
        DATA_train=DATA[train_index,:]
        DATA_test=DATA[test_index,:]
        ## grid search don't implemented if thhis is not necessary
        #C_range = np.logspace(-2, 10, 13)
        #gamma_range = np.logspace(-9, 3, 13)
        #param_grid = dict(gamma=gamma_range, C=C_range)
        #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        #grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        #grid.fit(DATA_train,labels[train_index].astype('int'))
        model = make_pipeline(MinMaxScaler(),LinearSVC(random_state=0, tol=1e-5))
        # fit the model on the training set
        model.fit(DATA_train, labels[train_index].astype('int'))
        # make predictions on the test set
        predictions = model.predict(DATA_test)
        # calculate classification accuracy
        acc[i] = accuracy_score(labels[test_index].astype('int'), predictions)
        print(acc[i])
        if int(sys.argv[3])==1:
           RocCurveDisplay.from_estimator(model, DATA_test, labels[test_index].astype('int'))
           plt.draw()
           plt.pause(0.001)
           input("Press [enter] to continue.")
acc_mean=np.mean(acc)
acc_std=np.std(acc)
print(f":accuracy:{acc_mean} +/- {acc_std}")
