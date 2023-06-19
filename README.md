# ABATech_AI_test
In this explanatory README file we will explain the techniques to deploy an Exploratory Data Analysis (EDA) given the demopgraphic and company-based data given by the company **ABATech** located in Medellin, Colombia. This company provides outsourcing and consultancy services to clinical and custom corporation in the United States. Here we will explain first the **EDA** with the instruction to deploy it and also the instructions to evaluate three different classifiers, such as, logistic, SVM, and Neural Network for satisfaction decoding from two-class problem with data trials labelled with **satisfied** and **dissatisfied**. In this implementation we extend statistical features from the original features of the data, thus obtaining an accuracy equal to **0.96** with the best classifier and for a **5-fold** crossvalidation. To clarify, we do not utilize an autoencoder to reduce the dimensionality of the features due to the evident unbalance between the number of sample and the number of features **n_samples >> n_features**. This fact encourage us to extend or generate more features from the 22 original features. The original features included in this implementation were **Gender**,	**Patient Type**,	**Age**,	**Ensurance**,	**Class**,	**Clinic Distance**,	**Wifi**,	**Time convenience**,	**Online booking**,	**Clinic location**,	**Food and drink**,	**Waiting room**,	**Comfort of the facilities**,	**Inflight entertainment**,	**Pre-entry service**,	**Post-entry service**,	**Visitor service**,	**Check-in service**,	**Medic service**,	**Cleanliness**,	**Waiting time (min)**, and	**Delay in care (min)**. Some of them were categorical, other integer, and some few float type, such as, the time-based features. Now we proceed to explain the EDA and the model evaluations.

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
In the following images we will show examples of **stem and leaf**, **histogram**, **boxplot**, and **scatter-plot** examples extracted for particular examples of certain features contained in the **technical_test.csv** file. Some of the boxplots show a statistical difference between the **satisfied** and **dissatisfied** distributions. These differences show that the participants that score higher values in the features (around 4-5) are more **satisfied** in comparison with the **dissatisfied** participants. First we start showing some **stem and leaf** plots for the **Age** feature.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/steam_leaf_1.png" width="800" height="450">

This plot/figure shows the  **stem and leaf** plot for the **Age** feature for the distribution **satisfied**. This plot shows an approximate normal distribution for this particular features

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/steam_leaf_2.png" width="800" height="450">

This plot/figure shows the  **stem and leaf** plot for the **Age** feature for the distribution **dissatisfied**. This plot shows an approximate normal distribution for this particular feature. A histogram plot showing these distributions in the same axis comfirm that the difference between the means of this feature is not very different. For this type of continuous variable/feature, the tendency is not showing an evident difference between the **satisfied** and **dissatisfied** distributions.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/histogram_age.png" width="800" height="450">

On the other hand, the 0-5 feature answer denoted as **medic service** show a more plausible difference in the histograms with higher frequencies asssociated with higher values in the **satisfied** distribution in comparison with the **dissatisfied** one. Here we show this particular histogram and the effect is more evident in comparison with the **Age** feature.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/histogram_medic_service.png" width="800" height="450">

This effect happens in other categorical variables, as we can see that in the boxplots of other features such as **post entry** and **check-in service**. Again here the **satisfied** distribution shows higher values in comparison with the **dissatisfied** one , thus implying a significant statistical difference in this marginal data. In the following plots we show the **post entry** and **check-in service** boxplots.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/boxplot_post_entry.png" width="800" height="450">

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/boxplot_check_in_service.png" width="800" height="450">

The same effect can be observable when the code is executed in a **multivariate analysis** where multiple features show a similar behavior with higher ratings in the **satisfied** in comparison with the **dissatisfied**. This effect must be taken into account by the classifiers and the feature generator algorithms that we use in this repository - to obtain better **satisfied** and **dissatisfied** identification (classification) trials. To check the higher values associated with **satisfied** we can plot a grouped bar-plot between the features **post entry**, **Visitor service**, **check-in service**, and **Medic service**. We present these barplots as follows.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/grouped_barplot.png" width="800" height="450">

The data is showing a general tendency consisting in the participants who feel more approval with certain company services are more correlated with the **satisfied** trials, and the participants that dissapproves those services are more related to the **dissatisfied** trials. This seems a logical behavior in this particular type of data, and the effect is confirmed in multiple features.  The main issue with this dataset appears more when scatter-plots do not show a tendency or a pattern of differentiation between **satisfied** and **dissatisfied** distributions. This occurs because there are multiple participants that gives overlapped answers in the categorical features. The continuous features are more sparse, and as well as the categorical, they are very overlapped. We can see this effect in the following scatter-plots, the first between **post entry** and **Visitor service**, and the second between **Age** and **Delay in care (min)**.

 <img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/scatter_plot_post_entry_visitor_service.png" width="800" height="450">

 <img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/scatter_plot_age_delay_time.png" width="800" height="450">

The effect observed after pairing the features between them, force us to look for more features or features extracted statistically to obtain a potential good    performance in the **satisfied** and **dissatisfied** distributions decoding. We will discuss in the following section that consists in the evaluation of three classifiers/models such as **logistic** linear, linear **SVM**, and a two hidden-layer **Neural Network**, each of this model was evaluated following the original features and features generated from the original on statistical maps provided by [scikit-learn](https://scikit-learn.org) package in Python.

# Models Evaluation

In this repository we evaluate three classifiers, as we mentioned above, 1) [**logistic linear**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), 2) [**linear SVM**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), and 3) two-hidden layer [**Neural Network**](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). Each of this classifiers was evaluated in two feature-set scenarios, 1) the first adding the 22 features in this dataset scaled with [**MinMaxScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), and 2) the second generating 124 more features. A) 22 from the [**MinMaxScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), B) 22 from the [**StandardScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), C) 22 from the [**RobustScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), D) 22 from the [**QuantileTransformer**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) that tranform each feature distribution to more normal and avoid the effect of the outliers, E) 22 from [**KBinsDiscretizer**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) that binarize the data in a series of uniform intervals, F) 7 from [**PCA**](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) decomposition that extract the 7 more variable components from a lower-dimension SVD feature space, and G) 7 from [**TruncatedSVD**](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
that deploys another SVD decomposition but without centering the data, thus preserving the sparsity of each feature in a new feature-space. The following figure shows the pipeline that we evaluate in this repository and the one that represents the better results.

 <img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/pipelineabatech.jpg" width="900" height="450">

If the user does not want to go in detail into the notebooks for running the models evaluation, can run the following Python commands to execute the baselines incluing only the normalized 22 features. The commands include the value of Kfold evaluation (for our particular case **k=5**) and the selector to plot a ROC curve for each fold. You can run the following Python commands to run the model baselines for the **logistic** linear, linear **SVM**, and the two hidden-layer **Neural Network** classifiers.

```python
python baseline_logistic.py technical_test.csv <folding_parameter> <plotting_selector> 
python baseline_svm.py technical_test.csv <folding_parameter> <plotting_selector>
python baseline_nn.py technical_test.csv <folding_parameter> <plotting_selector>
```
Similarly, for the 124 features we have three different .py files associated with the execution/evaluation of the three classifiers we included in this project **logistic** linear, linear **SVM**, and the two hidden-layer **Neural Network** classifiers. The Python commands have the same input parameters, the value of the Kfold evaluation and plotting selector for allowing the ROC curves per fold. You can run the following Python commands to run the adding-features model for the **logistic** linear, linear **SVM**, and the two hidden-layer **Neural Network** classifiers.

```python
python model_evaluation_adding_features.py technical_test.csv <folding_parameter> <plotting_selector> 
python model_evaluation_adding_features_svm.py technical_test.csv <folding_parameter> <plotting_selector>
python model_evaluation_adding_features_nn.py technical_test.csv <folding_parameter> <plotting_selector>
```
You can follow the previous Python commands if you don't want to follow the details of the notebooks in the **notebooks** folder, but you can take any alternative for evaluating the models. In this evaluation we test a 5-fold crossvalidation (as we mentioned above) for the baselines and the added-features models (124 features). In the following table we report the average and the standard deviation of the accuracy for each modality and each classifier.

|   feat/class    |Logistic |   SVM   |   2-layer NN     |     
|----------------	|--------	|--------	|-----------------	|
| 22 features     | 0.87271 ± 0.00267  	| 0.87193 ± 0.00216  	| 0.94372 ± 0.00288 |
| 124 features    | 0.92167 ± 0.00187  	| 0.92521  ± 0.00097  	| **0.96072 ± 0.00112**	|

The best accuracy we found when we extend the features to 124 and train the two-layer **Neural Network**. The accuracy values across the folds are higher than **ACC>=0.95**. Adding new statistical features and analyzing the new features with an autoencoder can be possibilities of a future work - and maybe improve more the decoding performance. However, the performance achieved with the proposed system and the aggregated features is significant having 129880 samples, and trying to balance the difference between the samples and features is a good option for this model evaluation. In the following Figure, we show the average ROC curve (across the 5 folds) for the two-layer **Neural Network** feeding the classifier with extended 124 features. The **AUC=0.99** for this combination of feature set and classifier. Other metrics for the best classifier performance, in this case the two-layer **Neural Network**, are for instance:

- **Precision =** 0.98288 ± 0.00175
- **Recall =** 0.94945 ± 0.00315
- **F1=** 0.96587 ± 0.001067

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/ROC_curve_best_model.png" width="900" height="450">

To illustrate better the performance of the two-layer **Neural Network** feeding the classifier with extended 124 features, we measure a confusion matrix showing (a) the number of trials on each cell, and (b) the rate relative to the total trials located on each cell. We can see that the amount of trials classified correctly in significant in comparison with the error of this classifier, thus representing the best performance for **satisfied** and **dissatisfied**  trials decoding. The following plot/figure show the macro confusion matrix for the two modalities described above.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/confusion_matrices.jpg" width="1000" height="400">

We added an extra option to emulate our based classifier and metrics using [**Tensorflow**](https://www.tensorflow.org/) with [**keras**](https://keras.io/). They are installed following the requirements.txt list. To evaluate the implementation of the **Neural Network** classifier we modeled in [scikit-learn](https://scikit-learn.org) in [**Tensorflow**](https://www.tensorflow.org/) with [**keras**](https://keras.io/) you can run the following command in Python. In this case, the plotting selector equal to 1, will make appear the learning curve showing the training and test lost for each fold.

```python
python model_evaluation_adding_features_deep.py technical_test.csv <folding_parameter> <plotting_selector> 
```

Now to automatize the app a little bit more, we implemented a simple web-app in **Flask**, **javascript**, and **html**. From this web-app it is possible to execute the models specifying any feature-set, any folding parameter, and any classifier. You just need to run it locally in the address **http://0.0.0.0:8080** or check the automatic address **Flask** is giving to you when you are running the app up. For this app always access the port **8080** you can mount this app in the web using [**ngrok**](https://ngrok.com/).
Please open a new account and install the last **ngrok** version. You can run a **ngrok** server for this app using the following command.

```bash
ngrok http 8080
```
Having or not having **ngrok** installed you can run the app locally from **Flask** server. Simply run a Python command to activate the flask web-app in **http://0.0.0.0:8080** or **http://10.50.120.157:8080/**, and be sure you are in the root folder accessing the folders **static** and **templates**.

```python
python app.py
```
After you run this command from you root folder you can open **http://0.0.0.0:8080** or **http://10.50.120.157:8080/** in your more suitable web-browser. We suggest Firefox or Google Chrome to run this web-app. Please recheck in what address the **app.py** file will be running and change the address in **app.js** for doing the corresponding fetches to **Flask**. The following screen will appear in your browser when you load this local address. There you will be able to see a section for login marked with a blue rectangle, a section for setting the parameters marked with a red rectangle 1) the classifiers with a mutually exclusive checkbox, 2) the folding parameter form input as a text slide, and 3) a checkbox to select the original 22 features or extend them to 124. A status section is marked with a green rectangle, in this section we will report the result for the corresponding model evaluation.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/screen_app_first.jpg" width="900" height="450">

The application will let you evaluate your model when filled all the fields, such as, the username, folding parameter, classifier, and feature-set. If any of these fields is empty the application will rebound and won't let you/user evaluate your selected model letting you know you/user need to fill all the fields. To continue adding your data to the **Flask** application you need to click the button **Run Evaluation**. After you clicked this button the application will confirm it received your data and if you will continue with your selected model evaluation. The following screen show the status update in a red rectangle. This status section shows the data reception confirmation. After this status is updated, you/user need to click the **Run Evaluation** button to run the model deploying in the background - clicking the button a second time.

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/screen_app_second.jpg" width="900" height="450">

After you clicked the **Run Evaluation** button for the second time and the model is deployed, the **Flask**-based web-app will wait until the Kfold validation finishes. Depending on the model and the feature-set you/user have selected, the Kfold evaluation will take less or more time to finish. The status section will be updated when the model process finishes in the background. The status section will receive the results from Python stoud/stdin. Accuracies for each fold will be reported in the status section as well as the average and the standard deviation of the accuracies across the folds. The following screen show the moment when the application received the models evaluation. Accuracy results appear in the red rectangle. 

<img src="https://github.com/meiyor/ABATech_AI_test/blob/main/images/screen_app_third.jpg" width="900" height="450">

I feel very motivated doing this project for ABATech, I hope it will be a starting point to implement more complete and ambitious projects in the future, and more important I hope to work/collaborate with you in the near future.
