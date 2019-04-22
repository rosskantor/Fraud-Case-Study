import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from fancyimpute import KNN
from keras import models, layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def load_csv_2():
    """
    Loads modified, imputed csv data file.
    returns data frame used by main function
    """
    return pd.read_csv('../data/RossData_imputed.csv')


def fill_blanks(df , filler=0):
    """
    df: name of dataframe containing blanks to be filled with zeros
    filler: values to be filled in NaN cells. Default is 0.
    returns dataframe with no blanks
    """
    return df.fillna(filler, inplace=True)


def split_X_y_cols(df) -> object:
    """
    df: dataframe name
    X_col: names of x columns to be split from autofinance DataFrame
    y_col: name of y column to be split from autofinance DataFrame
    returns X and y dataframe
    """
    X = df[['Variable01', 'Variable02', 'Variable03', 'Variable04', 'Variable05', 'Variable07',
                               'Variable08', 'Variable09', 'Variable10', 'Variable11',
                               'Region_MT', 'Region_MW', 'Region_NE', 'Region_P', 'Region_S', 'Region_WE']]
    y = df[['Result02']]

    return X, y


def get_dummies(df):
    """
    df: name of dataframe containing a Region column to receive dummy variables
    returns dataframe with regional dummy variables
    """
    return pd.get_dummies(df, columns=['Region'], drop_first=True)


def split_apply(x, y):
    """
    X: DataFrame containing X columns
    y: DataFrame containing y column
    This function creates training and test tables
    returns X and y training and test dataframes
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


def cost_benefit():
    """
    Builds cost benefit matrix
    returns 2X2 cost benefit matrix
    """
    cb = np.array([[660, -120], [0, 0]])


def standardize( x_train, x_test, x_col_input):
    """
    X_train: training DataFrame
    X_test: testing DataFrame
    X_col_input: name of X columns to be standardized
    returns normalized X train and test dataframes
    """

    scaler = StandardScaler().fit(x_train)

    x_train_1 = pd.DataFrame(data=scaler.transform(x_train), columns=x_col_input)

    x_test_1 = pd.DataFrame(data=scaler.transform(x_test), columns=x_col_input)


def splitdf( df):
    """
    df: name of DataFrame to be imputed per KNN(5)
    """
    dfreg = pd.DataFrame(df.Region)
    df1 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[:10000].select_dtypes(exclude='object')),
                            columns=df.iloc[:10000].select_dtypes(exclude='object').columns,
                            index=df.iloc[:10000].select_dtypes(exclude='object').index)
    df2 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[10000:20000].select_dtypes(exclude='object')),
                            columns=df.iloc[10000:20000].select_dtypes(exclude='object').columns,
                            index=df.iloc[10000:20000].select_dtypes(exclude='object').index)
    df3 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[20000:].select_dtypes(exclude='object')),
                            columns=df.iloc[20000:].select_dtypes(exclude='object').columns,
                            index=df.iloc[20000:].select_dtypes(exclude='object').index)
    y = df1.append(df2)
    y2 = y.append(df3)
    df = y2.merge(dfreg, left_index=True, right_index=True)
    df.to_csv('RossData_imputed.csv')


def roc_curve(probabilities, labels):
    """
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates, Thresholds for the
    ROC curve and the profit matrix
    """
    cb = costbenefit()
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []
    profit = []

    num_positive_cases = labels.sum()
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels.iloc[:, 0])
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)
        # Calculate predicted negative cases
        profit.append((true_positives * cb[0][0]) + (false_positives * cb[0][1]))
        # Populate TP

        # Populate FP

        # Populate FN

        # Populate TN

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist(), profit


def plotter(y_test, X_cols, y_Cols, figname, probreturn, x_short_identifier):
    """
    y_test: y test dataframe
    X_cols: name of x columns
    y_Cols: name of y column
    figname: name of plot (probably used)
    probreturn: list of probabilities to be passed to grapher
    x_short_identifier: keys identifying X column short names
    returns profit matrix and graph
    """
    tpr, fpr, thresholds, profit = roc_curve(probreturn, y_test)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.rcParams.update({'font.size': 18})
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(fpr, tpr, 'b')
    ax2.plot(thresholds, profit, 'r')
    x = np.linspace(0, 1, len(thresholds))
    ax3.plot(x, x, linestyle='--')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity, Recall)", color='b')
    ax2.set_ylabel("Profit", color='r')
    ax3.set_yticklabels([])
    plt.title(
        "ROC and Profitability plot of features \n " + str(x_short_identifier) + " Max Profit " + str(max(profit)))
    plt.savefig(figname + '.png')
    plt.show()
    return profit


def model(x_train, x_test, y_train, y_test):
    """
    x_train: x training dataframe
    y_train: y training DataFrame
    x_test: x test dataframe
    y_test: y test dataframe
    returns list of probabilities for each test value
    """

    # reshape y array
    y = np.array(np.ravel(y_train)).astype(int)

    # define scaler on x train
    scaler.fit(x_train)

    # fit model on x train and y array
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(scaler.transform(x_train), y)

    # return predictions
    preds = clf.predict(scaler.transform(x_test))

    # return probabilities
    probs = clf.predict_proba(scaler.transform(x_test))

    # logistic model accuracy
    log_accuracy = accuracy_score(preds, y_test)

    # logistic model recall
    log_recall = recall_score(preds, y_test)

    # logistic model precision
    log_precision = precision_score(preds, y_test)

    return log_accuracy, log_recall ,log_precision


def tree(x_train, x_test, y_train, y_test):
    """
    x_train: x training dataframe
    y_train: y training DataFrame
    x_test: x test dataframe
    y_test: y test dataframe
    returns list of probabilities for each test value
    """

    # define random forest classifier
    t = RandomForestClassifier(n_estimators=400, oob_score=True)

    # fit random forest model
    t.fit(x_train, y_train.values.ravel())

    # predict random forest model
    t_pred = t.predict(x_test)

    # rf model accuracy
    tree_accuracy = accuracy_score(t_pred, y_test.values.ravel())

    # rf model precision
    tree_precision = precision_score(t_pred, y_test.values.ravel())

    # rf model recall
    tree_recall = recall_score(t_pred, y_test.values.ravel())

    return tree_accuracy, tree_precision, tree_recall


def sample_again(df):
    majority = df[autofinance.Result02 == 0]
    minority = df[autofinance.Result02 == 1]

    df_minority_upsampled = resample(df[df.Result02 == 1],
                                     replace=True,  # sample with replacement
                                     n_samples=len(majority),  # to match majority class
                                     random_state=123)

    return pd.concat([df_minority_upsampled, majority])


def neural_net(train, test, train1, test1):
    # Start a Neural Network
    network = models.Sequential()

    # Add fully connected layer with a Relu activation function

    network.add(layers.Dense(units=16, activation='relu', \
                             input_shape=(16,)))

    # Add fully connected layer with a Relu activation function
    network.add(layers.Dense(units=16, activation='relu'))

    # Add fully connected layer with a Relu activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile Neural Network
    network.compile(loss='binary_crossentropy',  # Cross-entropy \
                    optimizer='rmsprop',  # Root Mean Square Propogation \
                    metrics=['accuracy']  # Accuracy performance metric
                    )

    history = network.fit(x_train,  # Features
                          y_train,  # Target
                          epochs=25,  # Number of iterations
                          verbose=1,  # Print Success after each epoch
                          batch_size=100,  # Number of observations per batch
                          validation_data=(x_test, y_test))  # Test data


def main():

    # Execute functions
    autofinance_df = load_csv_2()
    df_no_blanks = fill_blanks(autofinance_df)
    df_dummies = get_dummies(df_no_blanks)
    df_resmpled = sample_again(df_dummies)
    X, y = split_X_y_cols(df_resmpled)
    x_train, x_test, y_train, y_test = split_apply(X, y)
    model(x_train, x_test, y_train, y_test)
    tree(x_train, x_test, y_train, y_test)
    neural_net(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
