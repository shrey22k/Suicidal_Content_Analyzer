# Suicidal Post Analyzer using NLP and Machine Learning
This is a final year project that uses Natural Language Processing and a Machine Learning model to analyze and classify user-generated text for potential suicide risk.

## Project Architecture
The project consists of a Python Flask backend that serves a trained Logistic Regression model, and a simple HTML/JavaScript frontend for user interaction.

## Setup and Installation
### Prerequisites
Python 3.7+
Git and Git LFS

## 1. Clone the Repository
Clone this repository to your local machine. Make sure you have Git LFS installed.

git clone [Your-GitHub-Repo-URL]
cd [your-repo-name]

## 2. Download the Dataset
This project uses the "Suicide and Depression Detection" dataset from Kaggle. The dataset is not included in this repository due to its size.

Download the dataset from here: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

Place the Suicide_Detection.csv file in the root of this project folder.

## 3. Set Up the Python Environment
Create and activate a virtual environment:
python -m venv venv

### On Windows
.\venv\Scripts\activate
### On Mac/Linux
source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

(Note: You'll need to create a requirements.txt file by running pip freeze > requirements.txt in your activated environment).

## How to Run the Project
### 1. Train the Model
Run the training script to generate the model and vectorizer files (model.joblib and vectorizer.joblib).
python train.py

### 2. Start the Backend Server
Run the Flask API server. This will serve the model at http://127.0.0.1:5000.
python app.py

### 3. Launch the Frontend
Open the index_advanced.html file with a live server (e.g., the "Live Server" extension in VS Code).
The web application will now be running and can communicate with the backend.