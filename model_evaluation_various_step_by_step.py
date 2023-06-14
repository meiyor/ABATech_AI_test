from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFE
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot
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

KFold(n_splits=int(sys.argv[2]), random_state=None, shuffle=False)

acc=np.zeros([int(sys.argv[2])])

for i, (train_index, test_index) in enumerate(kf.split(DATA)):
        print(f":Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        
        ## first define define the data 
        DATA_train=DATA[train_index,:]
        DATA_test=DATA[test_index,:]
        DATA_TRAIN=[]
        DATA_TEST=[]
        ## scalers transformers concatenate all the requested output for generating more features
        mint=MinMaxScaler()
        temp=mint.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=temp
        stdt=StandardScaler()
        temp=stdt.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        robt=RobustScaler()
        temp=robt.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        ## distribution changes transforms
        qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
        temp=qt.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        temp=kbd.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        ## feature selectors transforms
        pca=PCA(n_components=7)
        temp=pca.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        svd = TruncatedSVD(n_components=7)
        temp=svd.fit_transform(DATA_train.astype(float),labels[train_index].astype('int'))
        DATA_TRAIN=np.concatenate((DATA_TRAIN,temp), axis=1)
        
        ## apply this to test data
        temp=mint.transform(DATA_test.astype(float))
        DATA_TEST=temp
        temp=stdt.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        temp=robt.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        temp=qt.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        temp=kbd.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        temp=pca.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        temp=svd.transform(DATA_test.astype(float))
        DATA_TEST=np.concatenate((DATA_TEST,temp), axis=1)
        # define the feature selection
        #rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
        # define the model
        model = LogisticRegression(solver='liblinear')
        # fit the model on the training set
        model.fit(DATA_TRAIN, labels[train_index].astype('int'))
        # make predictions on the test set
        #tsne_results_test =  tsne_pipeline.transform(DATA_test)
        predictions = model.predict(DATA_TEST)
        # calculate classification accuracy
        acc[i] = accuracy_score(labels[test_index].astype('int'), predictions)
        print(acc[i])
acc_mean=np.mean(acc)
acc_std=np.std(acc)
print(f":accuracy:{acc_mean} +/- {acc_std}")
