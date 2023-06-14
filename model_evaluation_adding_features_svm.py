from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

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
        transforms = list()
        transforms.append(('mms', MinMaxScaler()))
        transforms.append(('ss', StandardScaler()))
        transforms.append(('rs', RobustScaler()))
        transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
        transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
        transforms.append(('pca', PCA(n_components=7)))
        transforms.append(('svd', TruncatedSVD(n_components=7)))

        # create the feature union
        fu = FeatureUnion(transforms)
        # define the feature selection
        #rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
        # define the model
        model = SVC(random_state=0,tol=1e-5,C=1e-2,gamma='auto')
        steps = list()
        steps.append(('fu', fu))
        #steps.append(('rfe', rfe))
        steps.append(('m', model))
        pipeline = Pipeline(steps=steps)
        #DATA[test_index,:] = t.transform(DATA[test_index,:].astype(float))
        # define the model
        DATA_train=DATA[train_index,:]
        DATA_test=DATA[test_index,:]
        DATA_train = pipeline.fit(DATA_train,labels[train_index].astype('int'))
        predictions = pipeline.predict(DATA_test)
        # calculate classification accuracy
        acc[i] = accuracy_score(labels[test_index].astype('int'), predictions)
        print(acc[i])
        if int(sys.argv[3])==1:
            RocCurveDisplay.from_estimator(pipeline, DATA_test, labels[test_index].astype('int'))
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue.")
acc_mean=np.mean(acc)
acc_std=np.std(acc)
print(f":accuracy:{acc_mean} +/- {acc_std}")
