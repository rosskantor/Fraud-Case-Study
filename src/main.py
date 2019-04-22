import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
import random


def identify_fraud(item):
    """
    :param item: array element
    :return: 1 for fraud 0 if not
    """
    if item.find('fraud') == 0:
        return 1
    else:
        return 0


def plot_roc(x, y, clf_class, **kwargs):
    """
    :param x: dataframe or two dimensional array
    :param y: one-dimensional array
    :param clf_class: sklearn classifier
    :param kwargs: key-word arguments used to be supplied to classifiers
    :return: None.  The function prints plots.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    clf = clf_class(**kwargs)
    clf.fit(x_train, y_train)

    clf_class = pickle.dumps(clf)
    # Predict probabilities, not classes
    y_prob = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=clf.__class__.__name__ + f' {round(roc_auc, 2)}')


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')


def plot_roc_curve(x,y,title):
    """
    :param x: dataframe or two dimensional array
    :param y: one-dimensional array
    :param title: string, chart title to be printed
    :return: None.
    """
    plot_roc(x, y, GradientBoostingClassifier)
    plot_roc(x, y, AdaBoostClassifier)
    plot_roc(x, y, RandomForestClassifier)
    plot_roc(x, y, LogisticRegression)
    plot_roc(x, y, DecisionTreeClassifier)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.legend(loc="lower right")
    plt.savefig('image/'+title+'.png')


def report_accuracy(y_test,y_predict):
    """
    :param y_test: the numpy array of the true class of the test data
    :param y_predict: the numpy array of the predicted class of the test data
    :return: None
    """

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
    classification_report= metrics.classification_report(y_true=y_test, y_pred=y_predict)

    print('Accuracy: {:.3f}'.format(accuracy))
    print('=========================================================')
    print('Classification Report: \n{}'.format(classification_report))


if __name__ == '__main__':
    df = pd.read_json('../data/data.json')
    df['Isfraud'] = df['acct_type'].apply(identify_fraud)

    # before_dataclean_plot = msno.matrix(df)
    # fig = before_dataclean_plot.get_figure()
    # fig.savefig("image/before_dataclean_plot.png",bbox_inches='tight')
    #
    # before_dataclean_heatmap = msno.heatmap(df)
    # fig = before_dataclean_heatmap.get_figure()
    # fig.savefig("image/before_dataclean_heatmap.png",bbox_inches='tight')

    df_re = df.drop(['has_header','venue_country','venue_latitude', 'venue_longitude', 'venue_name', 'venue_state'],axis=1)
    df_final = df_re.dropna()

    # after_dataclean_plot = msno.matrix(df_final)
    # fig = after_dataclean_plot.get_figure()
    # fig.savefig("image/after_dataclean_plot.png",bbox_inches='tight')

    lb_make = LabelEncoder()
    df_final["country_code"] = lb_make.fit_transform(df_final["country"])
    df_final["payout_type_code"] = lb_make.fit_transform(df_final["payout_type"])

    df_re = df_final.drop(['approx_payout_date','country','payout_type','acct_type','currency','description','email_domain','listed','name','org_desc','org_name','payee_name','previous_payouts','ticket_types','venue_address'],axis=1)

    # X = df_re
    # y = X.pop('Isfraud')
    # X_scaled = preprocessing.scale(X)
    # plot_roc_curve(X_scaled,y,'unbalanced_roc')

    df_fraud_0 = df_re[df_re.Isfraud == 0]
    df_fraud_1 = df_re[df_re.Isfraud == 1]
    df_fraud_0_new = df_fraud_0.sample(df_fraud_1.shape[0])
    df_bal = pd.concat([df_fraud_0_new, df_fraud_1], axis=0)

    X_bal = df_bal
    y_bal = X_bal.pop('Isfraud')
    X_bal_scaled = preprocessing.scale(X_bal)
    #plot_roc_curve(X_bal_scaled, y_bal,'balanced_roc')
    random.seed( 3 )
    X_train, X_test, y_train, y_test = train_test_split(X_bal_scaled, y_bal, random_state=42)
    clf = RandomForestClassifier()
    #clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_predict = clf.predict(X_test)
    report_accuracy(y_test,y_predict)

    # with open ('gradient_boosting_model.pkl','wb') as f:
    #     pickle.dump(clf, f)
