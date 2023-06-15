# ABATech_AI_test
In this explanatory README file we will explain the techniques to deploy an Exploratory Data Analysis (EDA) given the demopgraphic and company-based data given by the company **ABATech** located in Medellin, Colombia. This company provides outsourcing and consultancy services to clinical and custom corporation in the United States. Here we will explain first the **EDA** with the instruction to deploy it and also the instructions to evaluate three different classifiers, such as, logistic, SVM, and Neural Network for satisfaction decoding from two-class problem with data trials labelled with **satisfied** and **dissatisfied**. In this implementation we extend statistical features from the original features of the data, thus obtaining an accuracy higher than **0.95** with the best classifier and for a **5-fold** crossvalidation. To clarify, we do not utilize an autoencoder to reduce the dimensionality of the features given the evident unbalance between the number of sample and the number of features **n_samples >> n_features**. This fact encourage us to extend or generate more features from the 22 original features. The original features included in this implementation were **Gender**,	**Patient Type**,	**Age**,	**Ensurance**,	**Class**,	**Clinic Distance**,	**Wifi**,	**Time convenience**,	**Online booking**,	**Clinic location**,	**Food and drink**,	**Waiting room**,	**Comfort of the facilities**,	**Inflight entertainment**,	**Pre-entry service**,	**Post-entry service**,	**Visitor service**,	**Check-in service**,	**Medic service**,	**Cleanliness**,	**Waiting time (min)**, and	**Delay in care (min)**. Some of them were categorical, other integer, and some few float type, such as the time features. Now we proceed to explain the EDA and the model evaluations.

# Exploratory Data Analysis (EDA)
In the EDA we implement two types of analysis. 1) an **univariate analysis** plotting a) **stem and leaf** plots using the stemgraphic package, b) a **histogram** plot separating the **satisfied** and **dissatisfied** distributions, and c) a **boxplot** comparing the **satisfied** and **dissatisfied** distributions one to the side of the other. This analysis is doing univariate, this means that the user can parse the index or the name of the individual feature he/she want to analyze. If more than one feature index or name is given to the code a 2) **multivariate analysis** is deployed including a)**grouped-feature boxplot** plotting the variation of all the selected features put as parameters comparing **satisfied** and **dissatisfied** distributions one to the side of the other, and b) a **scatter-plot** for each combination of features given as input parameters. 

Remember to run this before start running any Python code or command. Remember to have installed a version of **Python>=3.8** as well as pip and python-dev. Now the first step to start running the code is to install the requirements specified in the **requirements.txt** file, please download it. You can proceed with the following command in  pip. 

```bash
pip install -r requirements.txt
```

After you clone this folder and you are located in the root folder you can proceed to convert the **.xlsx** file including the data, to a **.csv** to make your data managing simpler and faster. You can see this process following the EDA notebook in the **notebooks** folder, but you can do it easier running the following Python command.

```python
python read_data_excel.py technical_test.xlsx 
```
In this command you will read the name of the data file **technical_test.xlsx** and you will generate a **.csv** file with the same name **technical_test.csv**. The next step is to fill up the spots of the file that do not have a value or are empty. For doing this you need to use the command **sed** of bash as follows.

```bash
sed -i 's/,,/,0,/g' technical_test.csv 
```
After running this the user will be ready to run the EDA in property. As we described above the code receives the amount of feature indexes or names as you consider. Of course you can only plot and analyze the 22 features you have at once. The two types of command you can utilize, one for indexes and one for features name, are the following. Remember that you can also include number/indexes and features-names (strings) at the same time. The code can proces that input.

```python
python read_csv_EDA.py technical_test.csv 1 2 3 4...
python read_csv_EDA.py technical_test.csv "feature-name1" "feature-name2" "feature-name3"...  
```
In the following images we will show examples of **stem and leaf**, **histogram**, **boxplot**, and **scatter-plot** examples extracted for particular examples of certain features contained in the **technical_test.csv** file. Some of the boxplots show a statistical difference between the **satisfied** and **dissatisfied** distributions. These differences show that the participants that score higher values in the features (around 4-5) are more **satisfied** in comparison with the **dissatisfied** participants. First we start showing some **stem and leaf** plots for the feature **Age**.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/steam_leaf_1.png" width="800" height="450">

This plot shows the  **stem and leaf** plot for the feature **Age** for the distribution **satisfied**. This plot shows an approximate normal distribution for this particular features

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/steam_leaf_2.png" width="800" height="450">

This plot shows the  **stem and leaf** plot for the feature **Age** for the distribution **dissatisfied**. This plot shows an approximate normal distribution for this particular features. A histogram plot showing these distributions in the same axis comfirm that the difference between the means of this feature is not very different. For this continuous variable the tendency is not showing an evident difference between the **satisfied** and **dissatisfied** distributions.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/histogram_age.png" width="800" height="450">

On the other hand, the 1-5 feature answer denoted as **medic service** show a more plausible difference in the histograms with higher frequencies in higher values in the **satisfied** distribution in comparison with the **dissatisfied** one. Here we show this particular histogram.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/histogram_medic_service.png" width="800" height="450">

This effect happens in other categorical variables as we can see that in the boxplots of other features such as **post entry** and **check-in service**. Again here the **satisfied** distribution shows higher values in comparison with the **dissatisfied** one , thus implying a significant statistical difference in this marginal data. In the following plots we show the **post entry** and **check-in service** boxplots.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/boxplot_post_entry.png" width="800" height="450">

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/boxplot_check_in_service.png" width="800" height="450">

The same effect can be observable when the code is executed in a **multivariate analysis** where multiple features show a similar behavior with higher ratings in the **satisfied** in comparison with the **dissatisfied**. This effect must be taken into account by the classifiers and the feature generator algorithms that we use in this repository - to obtain better identification (classification) performance between 
