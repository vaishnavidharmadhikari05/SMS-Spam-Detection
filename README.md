# SMS Spam Detection using Machine Learning

## Project Overview

This project focuses on building a machine learning model to classify SMS messages as **Spam** or **Ham (Not Spam)**.

The project uses Natural Language Processing (NLP) techniques to convert text messages into numerical features and Logistic Regression for classification.

 ### Dataset Source

The SMS Spam Collection dataset was obtained from the 
[UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

The dataset contains 5,574 SMS messages labelled as ham (legitimate) or spam.
- **Total messages after removing duplicates:** 5,169
- **Target classes:** Ham and Spam
- **Preprocessing:** Removed duplicate messages and performed exploratory data analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Google Colab

## Project Workflow

1. Data collection and loading
2. Data cleaning and duplicate removal
3. Exploratory Data Analysis (EDA)
4. Splitting data into training, validation, and testing sets
5. Text feature extraction using TF-IDF
6. Training a Logistic Regression model
7. Evaluating the baseline model
8. Classification threshold tuning using validation data
9. Final evaluation on the test dataset
10. Testing custom SMS messages

## Model Evaluation

The final model was evaluated on a separate test dataset using a classification threshold of 0.3.

| Metric | Result |
|---|---:|
| Accuracy | 97.49% |
| Spam Precision | 94% |
| Spam Recall | 85% |
| Spam F1-score | 90% |

## Key Learning Outcomes

- Understanding text preprocessing and NLP fundamentals.
- Applying TF-IDF for text feature extraction.
- Training and evaluating a Logistic Regression classifier.
- Understanding precision, recall, F1-score, and confusion matrices.
- Learning why classification threshold tuning matters.

## Limitations

The model may misclassify certain messages, particularly those containing unfamiliar words or patterns. It is a learning prototype and should not be considered a fully reliable spam filtering system.

## How to Run

1. Clone this repository.
2. Open `SMS_spam_detection.ipynb` in Google Colab or Jupyter Notebook.
3. Download the SMS Spam Collection dataset.
4. Update the dataset file path if necessary.
5. Run the notebook cells sequentially.

## Author

Created as a machine learning project to explore text classification and spam detection.
