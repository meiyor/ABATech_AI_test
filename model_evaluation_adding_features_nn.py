from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA, SparsePCA
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
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import pandas as pd
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

font = {
    'weight' : 'bold',
    'size'   : 16}

plt.rc('font', **font)


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
pr=np.zeros([int(sys.argv[2])])
re=np.zeros([int(sys.argv[2])])
f1=np.zeros([int(sys.argv[2])])

labels_real=[]
labels_pred=[]

KFold(n_splits=int(sys.argv[2]), random_state=None, shuffle=False)

#if int(sys.argv[3])==1:
plt.ion()
plt.show()

for i, (train_index, test_index) in enumerate(kf.split(DATA)):
        np.random.seed(1234)
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
        #transforms.append(('sparsepca', SparsePCA(n_components=7)))
        transforms.append(('svd', TruncatedSVD(n_components=7)))

        # create the feature union
        fu = FeatureUnion(transforms)
        # define the feature selection
        #rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
        # define the model
        model = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(100,10),learning_rate='invscaling',random_state=1234,power_t=0.1,max_iter=350,tol=1e-7,n_iter_no_change=50)
        model.t=20 ## define iterations for invscaling
        model.out_activation_='softmax' ## define the output
        steps = list()
        steps.append(('fu', fu))
        #steps.append(('rfe', rfe))
        scaler=MinMaxScaler()
        steps.append(('sc',scaler))
        steps.append(('m', model))
        pipeline = Pipeline(steps=steps)
        # train the model
        DATA_train=DATA[train_index,:]
        DATA_test=DATA[test_index,:]
        #tsne_results =  tsne_pipeline.fit_transform(DATA_train)
        pipeline.fit(DATA_train,labels[train_index].astype('int'))
        predictions = pipeline.predict(DATA_test)
        # calculate classification accuracy
        acc[i] = accuracy_score(labels[test_index].astype('int'), predictions)
        print(acc[i])
        labels_real.append(labels[test_index].astype('int'))
        labels_pred.append(predictions)
        if int(sys.argv[3])==1:
            pr[i] = precision_score(labels[test_index].astype('int'), predictions)
            re[i] = recall_score(labels[test_index].astype('int'), predictions)
            f1[i] = f1_score(labels[test_index].astype('int'), predictions)
            print(pr[i],re[i],f1[i],'metrics')
            RocCurveDisplay.from_estimator(pipeline, DATA_test, labels[test_index].astype('int'))
            plt.draw()
            plt.pause(0.001)
            print()
            input("Press [enter] to continue.")
acc_mean=np.mean(acc)
acc_std=np.std(acc)
print(f":accuracy:{acc_mean} +/- {acc_std}")
if int(sys.argv[3])==1:
    pr_mean=np.mean(pr)
    pr_std=np.std(pr)
    print(f":precision:{pr_mean} +/- {pr_std}")
    re_mean=np.mean(re)
    re_std=np.std(re)
    print(f":recall:{re_mean} +/- {re_std}")
    f1_mean=np.mean(f1)
    f1_std=np.std(f1)
    print(f":F1:{f1_mean} +/- {f1_std}")
## confusion matrices
#print(np.hstack(labels_real), np.hstack(labels_pred), np.shape(np.hstack(labels_real)), np.shape(np.hstack(labels_pred)) ,'shapes')
CMatrix=confusion_matrix(np.squeeze(np.hstack(labels_real)), np.squeeze(np.hstack(labels_pred)),labels=model.classes_)
disp_cmatrix = ConfusionMatrixDisplay(confusion_matrix=CMatrix,display_labels=['satisfied','dissatisfied'])
CMatrix_norm=confusion_matrix(np.squeeze(np.hstack(labels_real)), np.squeeze(np.hstack(labels_pred)),normalize='all',labels=model.classes_)
disp_cmatrix_norm = ConfusionMatrixDisplay(confusion_matrix=CMatrix_norm,display_labels=['satisfied','dissatisfied'])
if int(sys.argv[3])==1:
   ## plot matrix with the numbers per class
   disp_cmatrix.plot()
   for labels in disp_cmatrix.text_.ravel():
      labels.set_fontsize(16)
   plt.draw()
   plt.pause(0.001)
   print()
   input("Press [enter] to continue.")
   ##plot matrix normalized
   disp_cmatrix_norm.plot()
   for labels in disp_cmatrix_norm.text_.ravel():
       labels.set_fontsize(20)
   plt.draw()
   plt.pause(0.001)
   print()
   input("Press [enter] to continue.")
