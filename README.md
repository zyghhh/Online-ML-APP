# Online Machine Learning App
Based on my graduation thesis research, I have developed an online machine learning application platform using the Streamlit framework. This platform utilizes Python 3.9 and extends the open-source online machine learning library River. It integrates the data-oriented development framework Streamlit and addresses the challenges of data mining in concept drift data streams. The platform provides evaluation criteria based on accuracy, time, and memory consumption, as well as various evaluation metrics for regression or classification problems. It aims to configure suitable and efficient online learning algorithms for complex data streams generated in different real-world production environments.

## Introduction to Files
### app.py
This is the main file to run the online machine learning application platform.

### home.py
Builds the home page.

### machine_learning.py
The classification tool page for classic offline learning algorithms based on batch processing.

### online_machine.py
The tool page for classification or regression using online machine learning algorithms.

### ensemble_learning.py
The machine learning tool page based on an online ensemble adaptive framework.

### contrast_experiment.py
The experimental tool page for parallel visual analysis of specified datasets.

### style.css
The front-end page style settings file.

### multipages.py
Defines multiple types of pages.

### dataset.csv/dataset2.csv
User-uploaded datasets.

## Usage Instructions
Required Environment
river: 0.15.0
python: 3.9.16
streamlit: 1.20.0

## Running Steps
Open the folder in the command line with the required virtual environment installed, then type: streamlit run app.py. The system will run in the default browser.
Note: Due to the use of some web-based UI components, the initial loading of the interface may be slow.

