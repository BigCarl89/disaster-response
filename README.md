# Disaster Response Pipeline Project by Alberto Carlone
#### Udacity Data Scientist Nanodegree

### Project Description:
This project aims to create a tool, deployed on a webapp, to categorize and recognize if a message is related to a request for help or a natural disaster.
To do this we use NLP (Natural Language Processing) algorithms, other than ETL and ML Pipelines.
In particular, the project is divided into three distinct parts:
1. An ETL (Extract-Transform-Load) pipeline that cleans and save the initial datasets of messages;
2. A ML pipeline where we train a model that categorize our messages. To do this we used, other than NLP tokenization algorithm, three distinct ML classifiers:
   Random Forest Classifier, Naive Bayes Classifier and Adaptive Boosting;
3. A web app based on Flask and Plotly (for visualization)

### Libraries and Python Version:
Python 3.7.5  
Pandas, Numpy, Scikit-learn for data wrangling, cleaning and ML  
NLTK for NLP  
Flask and Plotly for the web app development  


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

### Acknowledgment:
Figure Eight for the two datasets used for this project.
