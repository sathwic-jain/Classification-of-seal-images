import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier


#Exploratory data analysis
def check_data(df, df_name):
    #checking the data
    print(f"{df_name} :")
    print(df.head(10))
    print()

def check_data_shape(df,df_name):
    #checking the data info
    print(f"Shape of'{df_name}'")
    print(df.shape)
    print()

def check_null(df,df_name):
    #checks null values across all the columns
    null_vals = df.isnull().any(axis=0).sum()
    print(f"Total null values of'{df_name}' along the columns '{null_vals}'.")
    print()

#duplicate check
def duplicates(df,df_name):
    #checks duplicate values across all the columns
    duplicated_rows = X_train_binary.duplicated()
    total_duplicates_rows = duplicated_rows.sum()
    total_duplicates_cols = X_train_binary.T.duplicated().sum()
    print(f"Number of duplicate rows in'{df_name}' is '{total_duplicates_rows}'.")
    print(f"Number of duplicate cols in'{df_name}' is '{total_duplicates_cols}'.")
    print()


#Visualisation
def plot_data_Y(df,df_name):
    # select the first column of the dataframe
    column_name = df.columns[0]
    # count the frequency of each value in the selected column
    value_counts = df[column_name].value_counts()
    # create a pie chart of the value counts
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:pink', 'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
    labels = value_counts.index.tolist()
    # create a pie chart of the value counts
    #plt.pie(value_counts.values, labels=None)
    plt.pie(value_counts.values, colors=colors, labels=None)
    # set the title
    plt.title('Value Counts')
    # add a legend on the side
    plt.legend(title='Categories', labels=labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plt.legend(value_counts.index, loc='center left', bbox_to_anchor=(1.0, 0.5))
    # show the plot
    #plt.show()
    #saving the plot for future reference
    plt.savefig(f"visualPlot/{df_name}_data_visualisation.png")



# function to evaluate the models
def display_evaluation_for_models(y_val, y_val_pred, category, model):
    #Calculating evaluation metrics 
    val_score_acc = accuracy_score(y_val, y_val_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred, adjusted=True)
    precision = precision_score(y_val, y_val_pred, average='weighted')
    recall = recall_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    #Printing evaluation metrics
    print(f" Evaluation Metrics for {category} class")
    print(f"Accuracy for {model} : {val_score_acc}")
    print(f"Balanced Accuracy for {model} : {balanced_acc}")
    print(f"Precision for {model} : {precision}")
    print(f"Recall for {model} : {recall}")
    print(f"F1_score for {model} : {f1}")
    
    # create and save confusion matrix plot
    confusion_mat = confusion_matrix(y_val, y_val_pred)
    print('Confusion matrix:')
    print(confusion_mat)
    confusion_mat_plot = ConfusionMatrixDisplay(confusion_mat).plot()
    fig, ax = plt.subplots(figsize=(10, 10))
    confusion_mat_plot.plot(ax=ax, cmap=plt.cm.Reds)
    ax.set_title(f"{model}_{category}")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45)
    fig.savefig(f"images/{category}/{model}_confusion_matrix.png")


#importing the datasets
# binary datasets
#X_train_binary = pd.read_csv('X_train.csv', header=None)
X_train_binary = pd.read_csv(r"CS5014/binary/X_train.csv", header=None)
Y_train_binary = pd.read_csv(r"CS5014/binary/Y_train.csv", header=None)
X_test_B = pd.read_csv(r"CS5014/binary/X_test.csv", header=None)

# multi datasets
X_train_multi = pd.read_csv(r"CS5014/multi/X_train.csv", header=None)
Y_train_multi = pd.read_csv(r"CS5014/multi/Y_train.csv", header=None)
X_test_M = pd.read_csv(r"CS5014/multi/X_test.csv", header=None)

#Data analysis

#checking the data---------------------------------------------------
print('Checking the data')
print()
check_data(X_train_binary, 'X_train_binary')
check_data(Y_train_binary, 'Y_train_binary')
check_data(X_train_multi, 'X_train_multi')
check_data(Y_train_multi, 'Y_train_multi')

#shape of the data---------------------------------------------------
print('Checking the shape info :')
print()
check_data_shape(X_train_binary,'X_train_binary')
check_data_shape(Y_train_binary,'Y_train_binary')
check_data_shape(X_train_multi,'X_train_multi')
check_data_shape(Y_train_multi,'Y_train_multi')

#checking for null values---------------------------------------------------
print('Checking null values :')
print()
check_null(X_train_binary,'X_train_binary')
check_null(Y_train_binary,'Y_train_binary')
check_null(X_train_multi,'X_train_multi')
check_null(Y_train_multi,'Y_train_multi')

#checking for duplicates---------------------------------------------------
print('Checking duplicate values :')
print()
duplicates(X_train_binary, 'X_train_binary')
duplicates(X_train_multi, 'X_train_multi')

#Plotting the Y data to check the distribution-----------------------------------
print('Plotting and saving the data distribution :')
print()
plot_data_Y(Y_train_binary,'Y_train_binary')
plot_data_Y(Y_train_multi,'Y_train_multi')

#Splitting the data into training and validation sets-----------------------------------
print('Splitting the data into training and validation sets :')
print()
X_train_B, X_val_B, y_train_B, y_val_B = \
train_test_split(X_train_binary, Y_train_binary, test_size=0.20, stratify=Y_train_binary, random_state=50)

X_train_M, X_val_M, y_train_M, y_val_M = \
train_test_split(X_train_multi, Y_train_multi, test_size=0.20, stratify=Y_train_multi, random_state=50)

#Data preprocessing
print('Data preprocessing : Encoding and Scaling the data ')
print()
#Encoding the data---------------------------------------------------
# Create a LabelEncoder object
encoder = preprocessing.LabelEncoder()

#binary data
# Fit the LabelEncoder to the data and transform the data
labels = encoder.fit_transform(y_train_B.values.ravel())
y_train_B = pd.DataFrame(labels) # numpy arrays to pandas dataframe
#transforming validation data
labels = encoder.fit_transform(y_val_B.values.ravel())
y_val_B = pd.DataFrame(labels) # numpy arrays to pandas dataframe

#For multi class:
# Fit the LabelEncoder to the data and transform the data
labels = encoder.fit_transform(y_train_M.values.ravel())
y_train_M = pd.DataFrame(labels) # numpy arrays to pandas dataframe
#transforming validation data
labels = encoder.fit_transform(y_val_M.values.ravel())
y_val_M = pd.DataFrame(labels) # numpy arrays to pandas dataframe

#Scaling the data---------------------------------------------------
# Create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
#binary data
scaler.fit(X_train_B)
# Transform the training and testing data using the scaler
X_train_std = scaler.transform(X_train_B)
X_test_std_B = scaler.transform(X_test_B)
X_val_B = scaler.transform(X_val_B)
X_Binary_Train = pd.DataFrame(X_train_std)
X_Binary_Test = pd.DataFrame(X_test_std_B)
X_Binary_Val = pd.DataFrame(X_val_B)

#multi data
scaler.fit(X_train_M)
# Transform the training and testing data using the scaler
X_train_std = scaler.transform(X_train_M)
X_test_std_M = scaler.transform(X_test_M)
X_val_M = scaler.transform(X_val_M)
X_Multi_Train = pd.DataFrame(X_train_std)
X_Multi_Test = pd.DataFrame(X_test_std_M)
X_Multi_Val = pd.DataFrame(X_val_M)

#Dimensionality reduction---------------------------------------------------
print('Dimensionality reduction through Correlation matrix and PCA')
print()
#Correlation matrix----------------
#binary data
correlations = X_Binary_Train.corr()

#correlation threshold
threshold = 0.8

# Find pairs of columns with correlations above the thresold 
high_correlations = np.where(np.abs(correlations) > threshold)
high_correlations = [(X_Binary_Train.columns[x], X_Binary_Train.columns[y]) for x, y in zip(*high_correlations) if x != y and x < y]

# Drop columns with high correlation
for col1, col2 in high_correlations:
    if col2 in X_Binary_Train.columns:
        X_Binary_Train.drop(col2, axis=1, inplace=True)
        X_Binary_Val.drop(col2, axis=1, inplace=True)
        X_Binary_Test.drop(col2, axis=1, inplace=True)
        #print(f"Dropped column '{col2}' due to high correlation with '{col1}'.")

#multi data
correlations = X_Multi_Train.corr()

#correlation threshold
threshold = 0.8

# Find pairs of columns with correlations above the thresold 
high_correlations = np.where(np.abs(correlations) > threshold)
high_correlations = [(X_Multi_Train.columns[x], X_Multi_Train.columns[y]) for x, y in zip(*high_correlations) if x != y and x < y]

# Drop columns with high correlation
for col1, col2 in high_correlations:
    if col2 in X_Multi_Train.columns:
        X_Multi_Train.drop(col2, axis=1, inplace=True)
        X_Multi_Test.drop(col2, axis=1, inplace=True)
        X_Multi_Val.drop(col2, axis=1, inplace=True)
        #print(f"Dropped column '{col2}' due to high correlation with '{col1}'.")


#PCA----------------
print('Plotting and saving PCA plots')
print()
#binary data
pca = PCA()
pca.fit(X_Binary_Train)
fig = plt.figure(figsize=(20,6)) # Set the figure size to be 20`` inches wide by 6 inches tall
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.title('Scree Plot')
plt.axvline(x=20, color='red', linestyle='--')#markdown for x = 20 to show the elbow
#plt.show()
plt.savefig(f"PCA_plot/X_BinaryClass_PCA_plot.png")

# perform PCA on the selected features.
pca = PCA(n_components=20)
X_binary_PCA_Train = pca.fit_transform(X_Binary_Train)
X_binary_PCA_Test= pca.transform(X_Binary_Test)

#Validation set
X_val_PCA = pca.fit_transform(X_Binary_Val)

#multi data
pca = PCA()
pca.fit(X_Multi_Train)
fig = plt.figure(figsize=(20,6)) # Set the figure size to be 20`` inches wide by 6 inches tall
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.title('Scree Plot')
plt.axvline(x=20, color='red', linestyle='--')#markdown for x = 20
#plt.show()
#saving the plot for future reference
plt.savefig(f"PCA_plot/X_multiClass_PCA_plot.png")

# perform PCA on the selected features.
pca = PCA(n_components=20)
X_Multi_PCA_Train = pca.fit_transform(X_Multi_Train)
X_Multi_PCA_Test= pca.transform(X_Multi_Test)


#Validation set
X_val_PCA_M = pca.fit_transform(X_Multi_Val)
print('Entering the train and validation phase')
print()
#Training the models---------------------------------------------------

#Binary classification---------------------
# Define the hyperparameter space for logistic regression
logistic_param_dist = {
    'penalty': [None, 'l2'],
    'class_weight': ['balanced'], #None is taken out due to imbalance
    'max_iter': [100, 250, 500, 1000, 10000]
}


# Define the hyperparameter space for k-nearest neighbors
knn_param_dist = {
    'n_neighbors': randint(10, 20),
    'weights': ['distance'], #uniform is taken out due to imbalance
    'metric': ['euclidean', 'manhattan']
}

#Defining DT hyperparameters
dt_param_dist = {'max_depth': [2, 4, 6, 8, 10, None],
              'min_samples_split': randint(2, 10),
              'min_samples_leaf': randint(1, 10),
              'max_features': ['sqrt', 'log2', None],
              'class_weight': ['balanced'], #None is taken out due to imbalance
              'criterion': ['gini', 'entropy']}

# Create the logistic regression and k-nearest neighbors, decision tree objects
logistic = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

# Create a dictionary of models and their corresponding hyperparameter spaces
models_binary = {
    'logistic': (logistic, logistic_param_dist),
    'knn': (knn, knn_param_dist),
    'dt': (dt,dt_param_dist)
}

# Create a dictionary to store the best hyperparameters found for each model
best_params_Binary = {}

#Converting y to a column array
y_train_B=y_train_B.to_numpy().ravel()

# Loop over each model and perform a randomized search
for model_name, (model, param_dist) in models_binary.items():
    if model_name == 'logistic': #Logistic regression is a special case and is handled with grid search due to the small number of hyperparameters
        grid_search = GridSearchCV(logistic, logistic_param_dist, cv=5)
        grid_search.fit(X_binary_PCA_Train, y_train_B)
        best_params_Binary[model_name] = grid_search.best_params_
        # Make predictions on the testing set
        y_pred = grid_search.predict(X_val_PCA)
    else:   
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=100,
            cv=5,
            random_state=42
        )
        random_search.fit(X_binary_PCA_Train, y_train_B)
        best_params_Binary[model_name] = random_search.best_params_
        # Make predictions on the testing set
        y_pred = random_search.predict(X_val_PCA)

    # Calculate and displaythe metrics of the model
    display_evaluation_for_models(y_val_B, y_pred, 'BINARY', model_name)
    accuracy = accuracy_score(y_val_B, y_pred)

#Multi classification---------------------
# Define the hyperparameter space for SVC
#SVC_param_dist = {
#    'C': uniform(loc=0, scale=10),
#    'kernel': ['rbf'],
#    'degree': [2, 3, 4],
#    'gamma': ['scale', 'auto'] + list(np.logspace(-5, 2, 8))
#}
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Create the logistic regression and k-nearest neighbors, decision tree objects
#svc = SVC()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

# Create a dictionary of models and their corresponding hyperparameter spaces
models_Multi = {
    'knn': (knn, knn_param_dist),
    'dt': (dt,dt_param_dist),
#    'SVC': (svc, SVC_param_dist),
     'RF' : (rf, rf_param_dist)
}

# Create a dictionary to store the best hyperparameters found for each model
best_params_Multi = {}

#Converting y to a column array
y_train_M=y_train_M.to_numpy().ravel()

# Loop over each model and perform a randomized search
for model_name, (model, param_dist) in models_Multi.items():
    random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100, 
    cv=5, 
    n_jobs=-1, 
    scoring='accuracy', 
    random_state=42
    )
        
    random_search.fit(X_Multi_PCA_Train, y_train_M)
    best_params_Multi[model_name] = random_search.best_params_
    # Make predictions on the testing set
    y_pred = random_search.predict(X_val_PCA_M)

    # Calculate,display and store the metrics of each model
    display_evaluation_for_models(y_val_M, y_pred, 'MULTI', model_name)
    print()
#According to the results, the best model for binary classification is KNN 
#We use the same hyperparamters that were used during thre training phase for the test set
print('Prediction on the test set by manually selected features')
print()
def predictor(X_test,X_train, y_train, model_param, model, category):
    if(model == "Logistic Regression"):
        model_obj = LogisticRegression(**model_param)
    elif(model == "knn"):
        model_obj = KNeighborsClassifier(**model_param)
    elif(model == "Decision Tree"):
        model_obj = DecisionTreeClassifier(**model_param)
    elif(model == "RF"):
        model_obj = RandomForestClassifier(**model_param)
    model_obj.fit(X_train, y_train)
    #predicting the test data
    y_test_pred = model_obj.predict(X_test)
    #saving the predictions
    np.savetxt(f"predictions/{category}/{model}_predictions.csv", y_test_pred, delimiter=",", fmt='%d')
    print(f"Predictions for {model} saved to predictions folder")

predictor(X_binary_PCA_Test,X_binary_PCA_Train, y_train_B,best_params_Binary['knn'],'knn','BINARY')
predictor(X_Multi_PCA_Test,X_Multi_PCA_Train, y_train_M,best_params_Multi['RF'],'RF','MULTI')
print('Prediction have been saved on the source folder')