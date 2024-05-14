# Machine-Learning-House-Price-Prediction

Usage and implementation of a machine learning algorithm to allow for house price predictions in california.

This repo demonstrates usage of a custom pre-trained model that can be used over and over again without having to retrain the data.

---


### **<p align="center"> Diagram of the general flow of implentation can be seen below:</p>**
<p align="center"><img src="https://github.com/thatwonguy/Machine-Learning-House-Price-Prediction/assets/78534460/134e2693-b975-4165-97c7-b6e0ad81b792"></p>


1. **GATHERING DATA:**
   - The most important step is understanding what a successful result looks like. In this case we want to predict house values and have a location where the end-user can provide specific input parameters and receive a house value generated by a machine learning model.
   - Once we understand a final outcome we want we need to gather the relevant data that would allow us to do that.
   - ***We landed on US Census data on house prices, with a target column that provides house values that we can train our data on.***
2. **DATA PREPARATION:**
   - Here we bring in the data into a notebook and perform EDA, getting a good sense of the data, removing null values, and cleaning the data to see how we can work with it and what would be the best machine learning model to use and determine if we need more data or different data and continuing to prepare the data to ensure this data is clean and adequate to address the question we want answered with our model.
3. **DATA WRANGLING:**
   - This goes hand in hand with data preparation above. Obtaining all the data and prepping, preprocessing, cleaning and transforming the data. Please note: the preprocessing step needs to be clearly documented and defined, as any new data being fed into the machine learning model will always have to be preprocessed in the same consistent manner orderwise the Machine Learning model will not provide the results.
4. **ANALYSE DATA:**:
   - Again, this step goes hand in hand with the 2 previous steps inside your jupyter notebook environment typically and may start to involve f1-scores, variance/bias analysis, featuring engineering, correlation metrics, accuracy scores, and other statistical metrics to try and pin-point the best algorithm and solution for model training results and outputs. *A machine learning model that gives poor results is of no use ultimately.*
5. **TRAIN MODEL:**
   - In this case we utilized the sci-kit learn python library to carry out multiple models for testing to see which would provide us the best outcome:
      - linear regression
      - random forest
      - xgboost regressor (performed the best when also coupled with a gridsearchcv layer for additional optimization)
6. **TEST MODEL:**
   - This is all still being carried out in the reseach.ipynb notebook with metrics that show which model performed the best. Ideally, the data science/machine learning engineering team would have sprints set-aside during development as well as after deployment to continue to massage the machine learning model and algorithms to get better results and optimize results. This can be done with algorithm fine-tuning, better data, more data, more feature engineering. Time needs to be allocated for this if we want to see better results and this should be incorporated as continuous work during an AGILE SPRINT and treated as such.
   - This would also be the step where we would need to deploy the model and host the end-user experience and test outputs to ensure it will deploy as expected in the next step. This would ideally been done on a test/dev/staging environment designed to mimic the actual production environment.
   - Data Engineers and ML Engineers would take over at this step. The Data scientists and ML Engineers solely should be focused on the machine learning models and a seperate team would ideally handle deployment of the models. Additional coding and scripting and pipeline work is needed to deploy the models.
   - The data science team, with the help of data engineers, would implement the models in the form of pickle files or some other format.**Yes**, a machine learning model, along with the results and data and algorithm can all be saved in a neat package to be implemented for any new data so that no additional training is necessary. If the data science or machine learning team obtains a better model in the future, they should simply be able to provide the better model saved in the form of something like a pickle file that can be exchanged with the previous pickle file containing the less accurate model.
7. **DEPLOYMENT:**
   - The data engineering team would ideally abstract this pickle file in the code so that any new models would simply replace this model with a new one and no other portions of the established code-base would need to change. Good micro-services architecture and potential usage of OOPS programming on the part of data engineers would need to be implemented here.
   - These pickle files would essentially be pre-trained models (the best one we want to use) that would be incorporated into the code to be used for end-users. If the data science team finds a better model, they simply provide the new pickle file, and the old file gets replaced, wheraupon the end-user will immediately see better and more accurate results. This can change depending on what kind of infrastructure your team is using or has available to use.
   - The end result of this particular exercise demonstrating this process can be seen here:
## [https://ml-house-price-prediction-california.streamlit.app/](https://ml-house-price-prediction-california.streamlit.app/) 
