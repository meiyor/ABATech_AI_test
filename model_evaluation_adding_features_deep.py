from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
#from sklearn.metrics import RocCurveDisplay
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
import gc

n_inputs=124
weight_decay = 0.0001

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

#if int(sys.argv[3])==1:
#    plt.ion()
#    plt.show()

for i, (train_index, test_index) in enumerate(kf.split(DATA)):
        tf.random.set_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
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
        ## decoder definition
        visible = Input(shape=(n_inputs,))
        ## decoder level 1
        e = Dense(100,kernel_initializer=tf.keras.initializers.GlorotNormal())(visible)
        #e = BatchNormalization()(e)
        e = ReLU()(e)
        #e = Dropout(0.7)(e)
        e = Dense(10,kernel_initializer=tf.keras.initializers.GlorotNormal())(e)
        #e = BatchNormalization()(e)
        e = ReLU()(e)
        #e = Dropout(0.2)(e)
        # output layer
        output = Dense(2, activation='softmax')(e)
        # create the feature union
        fu = FeatureUnion(transforms)
        # define the feature selection
        #rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
        # define the model
        deep = Model(inputs=visible, outputs=output)
        # compile decoder model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=20,decay_rate=0.1)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        deep.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        DATA_train=DATA[train_index,:]
        DATA_test=DATA[test_index,:]
        ## add the normalizer in the steps
        steps = list()
        steps.append(('fu', fu))
        scaler=MinMaxScaler()
        steps.append(('sc',scaler))
        pipeline = Pipeline(steps=steps)
        DATA_train_encoder = pipeline.fit_transform(DATA_train,labels[train_index].astype('int'))
        DATA_test_encoder = pipeline.transform(DATA_test)
        ## train encoder
        history = deep.fit(DATA_train_encoder.astype(float),tf.keras.utils.to_categorical(labels[train_index].astype('int'),num_classes=2), epochs=250, batch_size=200, verbose=2,  validation_data=(DATA_test_encoder.astype(float),tf.keras.utils.to_categorical(labels[test_index].astype('int'),num_classes=2)))
        ## plot if the user wants
        if int(sys.argv[3])==1:
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()
        labels_predict_train = deep.predict(DATA_train_encoder.astype(np.float32))
        # decode the test data
        labels_predict_test = deep.predict(DATA_test_encoder.astype(np.float32))
        labels_train=np.argmax(labels_predict_train, axis=1)
        labels_test=np.argmax(labels_predict_test, axis=1)
        labels_train=np.squeeze(labels_train)
        labels_test=np.squeeze(labels_test)
        # calculate classification accuracy
        acc[i] = accuracy_score(labels[test_index].astype('int'), labels_test)
        print(acc[i])
        #if int(sys.argv[3])==1:
        #    RocCurveDisplay.from_estimator(model, DATA_Test, labels[test_index].astype('int'))
        #    plt.draw()
        #    plt.pause(0.001)
        #    input("Press [enter] to continue.")
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del deep
        gc.collect()
acc_mean=np.mean(acc)
acc_std=np.std(acc)
print(f":accuracy:{acc_mean} +/- {acc_std}")
