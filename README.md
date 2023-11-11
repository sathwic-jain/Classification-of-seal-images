# Classification-of-seal-images

This project involves the classification of seal pup images based on extracted features. The goal is to predict the class of an image using a machine learning model trained on the provided dataset.

## Task Description
The dataset comprises two directories: `binary` for binary classification and `multi` for multi-class classification. Each directory contains the following files:
- **X_train.csv**: Features extracted from images
- **Y_train.csv**: Corresponding class IDs
- **X_test.csv**: Test dataset without corresponding outputs

## Steps Involved

1. **Loading and Preparing Data**: Load the provided datasets, clean the data, and generate new features if required.
2. **Data Analysis and Visualization**: Analyze and visualize the dataset to understand its characteristics.
3. **Feature Selection**: Choose a suitable subset of features for model training.
4. **Model Training**: Select and train different classification models.
5. **Model Evaluation**: Evaluate and compare the performance of the trained models using best practices.
6. **Results and Discussion**: Present the critical discussion of the results, approach, methods, and dataset insights.

## How to Run

 **Run the Program**:
    ```
    python seal_pup_classification.py
    ```

 **Check Results**: Find the output predictions in the generated `Y_test.csv` file.

## Report Overview

Included in the repository is a report detailing the steps taken, models used, their performance, and critical insights. It covers the detailed dedscription of approach, results, and discussion based on the experimental findings.

- **Report File**: `CS5014-P2_Report.pdf.pdf`
