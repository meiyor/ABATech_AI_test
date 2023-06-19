## run this code with python>=3.8 to avoid syntax problem
import pandas as pd

import matplotlib.pyplot as plt

import sys

import stemgraphic

#import seaborn

import numpy as np

font = {
    'weight' : 'bold',
    'size'   : 16}

plt.rc('font', **font)


colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'lightcoral', 'seagreen', 'rebeccapurple', 'salmon', 'orange', 'silver', 'siena']
## function to define colors
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='red')
    plt.setp(bp['caps'][0], color='red')
    plt.setp(bp['caps'][1], color='red')
    plt.setp(bp['whiskers'][0], color='red')
    plt.setp(bp['whiskers'][1], color='red')
    plt.setp(bp['fliers'][0], color='red')
    plt.setp(bp['fliers'][1], color='red')
    plt.setp(bp['medians'][0], color='red')

    plt.setp(bp['boxes'][1], color='blue')
    plt.setp(bp['caps'][2], color='blue')
    plt.setp(bp['caps'][3], color='blue')
    plt.setp(bp['whiskers'][2], color='blue')
    plt.setp(bp['whiskers'][3], color='blue')
    plt.setp(bp['medians'][1], color='blue')

def setBoxColors_grouped(bp,size_features):
   ccount=0
   for i in range(0,size_features):
     plt.setp(bp['boxes'][i], color=colors[i])
     plt.setp(bp['caps'][i+ccount], color=colors[i])
     plt.setp(bp['caps'][i+ccount+1], color=colors[i])
     plt.setp(bp['whiskers'][i+ccount], color=colors[i])
     plt.setp(bp['whiskers'][i+ccount+1], color=colors[i])
     if i<size_features-2:
       plt.setp(bp['fliers'][i+ccount], color=colors[i])
       #plt.setp(bp['fliers'][i+ccount+1], color=colors[i])
     plt.setp(bp['medians'][i], color=colors[i])
     ccount=ccount+1

def plot_univariable(data,features_name,index,possibilities):
   ## plot the stem and leaf plots first
    print(f"stem and leaf plots - {features_name[index]}")
    ## use pd.Series to avoid problems with the attribute sample
    fig, ax1 = stemgraphic.stem_graphic(pd.Series(data[index,index_satisfied]))
    plt.title(f"steam and leaf plot - satisfied - for the feature {features_name[index]}")
    #plt.show()
    fig, ax2 = stemgraphic.stem_graphic(pd.Series(data[index,index_dissatisfied]))
    plt.title(f"steam and leaf plot - dissatisfied - for the feature {features_name[index]}")
    ax1.plot()
    ax2.plot()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    #plt.show()
    ## Histograms
    print(len(data[index,index_satisfied]),len(data[index,index_satisfied]),len(data[index,:]),'lengths')
    print(f'Histograms - {features_name[index]}')
    if features_name[index] == 'Waiting time (min)' or  features_name[index] == 'Delay in care (min)' :
        counts_satisfied, bins_satisfied = np.histogram(data[index,index_satisfied],bins=np.linspace(10,500,100))
        counts_dissatisfied , bins_dissatisfied  = np.histogram(data[index,index_dissatisfied],bins=np.linspace(10,500,100))
    else:
        counts_satisfied, bins_satisfied = np.histogram(data[index,index_satisfied])
        counts_dissatisfied , bins_dissatisfied  = np.histogram(data[index,index_dissatisfied])   
    ## do the univariate EDA analysis
    fig, ax = plt.subplots()
    ax.hist( bins_satisfied[:-1], weights=counts_satisfied, color='salmon', alpha=0.5, label='satisfied')
    ax.hist( bins_dissatisfied[:-1], weights=counts_dissatisfied, color='lightblue', alpha=0.5, label='dissatisfied')
    plt.title(f"Histograms {features_name[index]}")
    plt.legend(loc='upper right')
    plt.ylabel('Frequencies')
    plt.xlabel(f'{features_name[index]}')
    ## add the xticks
    if len(possibilities):
       tticks=np.linspace(0,max(data[index,:]),len(possibilities))
       if len(possibilities)==3:
           tticks[len(tticks)-1]=tticks[len(tticks)-1]-0.2
       else:
           tticks[len(tticks)-1]=tticks[len(tticks)-1]-0.1
       print(tticks,'tticks')
       ax.set_xticks(tticks)
       ax.set_xticklabels(possibilities)
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    ## Box plots
    print(f'Box-Plots - {features_name[index]}')
    fig = plt.figure()
    ax = plt.axes()
    bp = plt.boxplot([data[index,index_satisfied],data[index,index_dissatisfied]], positions = [0, 1], widths = 0.6)
    setBoxColors(bp)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['satisfied','dissatisfied'])
    plt.title(f"Box-plot {features_name[index]}")
    plt.ylabel(f"{features_name[index]} values")
    ## add the yticks
    if len(possibilities):
       tticks_y=np.linspace(0,max(data[index,:]),len(possibilities))
       print(tticks_y,'tticks_y')
       ax.set_yticks(tticks_y)
       ax.set_yticklabels(possibilities)
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")

## MAIN CODE
data = []
features_name = []
##dataframe from the CSV file. The first parameter will be always the csv file name and this reading is faster than the .xlsx
dataframeObject = pd.DataFrame(pd.read_csv(str(sys.argv[1])))

print(dataframeObject)

## show the columns of data to know the features name
for col_features in dataframeObject.columns:
    print(col_features)

## assigning the features names and putting them in a list
features=list(dataframeObject.columns.values)


## check if the inputs are string or integer for identify positions you will be able to do it in both ways
for index in range(2,len(sys.argv)):
   if sys.argv[index].isdigit():
      index_feature=features[int(sys.argv[index])]
      features_name.append(index_feature)
      dataframeObject[[index_feature]].replace(np.nan,0)
      data.append(dataframeObject[[index_feature]].to_numpy())
   else:
      #print(dataframeObject[[str(sys.argv[index])]].to_numpy)
      features_name.append(str(sys.argv[index]))
      dataframeObject[[str(sys.argv[index])]].replace(np.nan,0)
      data.append(dataframeObject[[str(sys.argv[index])]].to_numpy())

## transform the list to a numpy
data=np.squeeze(np.array(data))
shape=np.shape(data)

## take the satisfied and dissatisfied classes for vector plotting
last_index_features=features[len(features)-1]

labels_string = np.squeeze(dataframeObject[[last_index_features]].to_numpy())
#print(labels_string,'labels_string')

index_satisfied = np.squeeze(np.where(labels_string=='satisfied'))
index_dissatisfied = np.squeeze(np.where(labels_string=='dissatisfied'))

#print(index_satisfied,'satisfied')
#print(index_dissatisfied,'dissatisfied')

if len(shape)==1:
   ## adding an extra axis to manage the data easily
   data=data[np.newaxis,:]

#print(data,'data_original')
shape=np.shape(data)
## check if the data is string and transform it to numeric do it in general
POS=[]
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
  POS.append(possibilities)  
  #else:
    ## replace nans for zeros in this approach
    #print(count,data,type(data),'data_n')

#print(data,'data')

## now evaluate if the input is only one or more than 2
if shape[0]==1:
    plt.ion()
    plt.show()
    plot_univariable(data,features_name,0,POS[0])
else:
   DATA_satisfied=[]
   DATA_dissatisfied=[]
   labels=['satisfied','dissatisfied']
   plt.ion()
   plt.show() 
   for index_t in range(0,shape[0]):
       plot_univariable(data,features_name,index_t,POS[index_t])
       DATA_satisfied.append(data[index_t,index_satisfied])
       DATA_dissatisfied.append(data[index_t,index_dissatisfied])
   
   print('Grouped Box Plot')   
   fig = plt.figure()
   ax = plt.axes()
   hB=[]
   pos1=np.linspace(0,len(features_name)-1,len(features_name))
   pos2=np.linspace(len(features_name)+1,len(features_name)+len(features_name),len(features_name))
   ##plot the multivariate boxplots grouped
   bp = plt.boxplot(DATA_satisfied, positions = pos1, widths = 0.6)
   setBoxColors_grouped(bp,len(features_name))

   bp = plt.boxplot(DATA_dissatisfied, positions = pos2, widths = 0.6)
   setBoxColors_grouped(bp,len(features_name))
   
   for igrouped in range(0,len(features_name)):
       temp, = plt.plot([1,1],color=colors[igrouped])
       hB.append(temp)
   plt.legend(hB,features_name)
   for igrouped in range(0,len(features_name)):
       hB[igrouped].set_visible(False)
   plt.ylabel("feature values")
   ## set xtickslabels
   val1 = pos1[round(len(features_name)/2)-1]
   val2 = pos2[round(len(features_name)/2)-1]
   ax.set_xticks([val1,val2])
   ax.set_xticklabels(labels)
   plt.draw()
   plt.pause(0.001)
   input("Press [enter] to continue.")

   print('Scatter Plot')
   for index_s in range(0,shape[0]):
      for index_ss in range(index_s+1,shape[0]):
       fig = plt.figure()
       ax = plt.axes()
       ax.scatter(DATA_satisfied[index_s],DATA_satisfied[index_ss], s=20, c='r', marker="o", label='satisfied')
       ax.scatter(DATA_dissatisfied[index_s],DATA_dissatisfied[index_ss], s=20, c='b', marker="x", label='dissatisfied')
       plt.xlabel(f"{features_name[index_s]}")
       plt.ylabel(f"{features_name[index_ss]}")
       plt.legend(loc='upper left')
       plt.draw()
       plt.pause(0.001)
       input("Press [enter] to continue.")
