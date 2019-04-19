from flask import Flask, render_template
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import db

app = Flask(__name__)

#with open('gb_model.pkl', 'rb') as f:
   #model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    "render a splash page"
    return render_template('forms/index.html')

@app.route('/results', methods=['GET'])
def render():
    """ """
    stopper = 'Good to Go'
    df = pd.read_json('./data/data.json')
    y = df.pop('acct_type')
    f = db.fraud(df, y)
    f.requester()
    f.jasmine_we_miss_you()
    f.pred()
    f.pred_proba()
    f.update_output_1()
    f.update_output_2()
    f.write_to_psql()
    predicted_prob = f.pred_prob
    data = predict_proba(predicted_prob)
    return render_template('forms/results.html', data=data)

def load_data_from_api():
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    r = requests.get(url)
    x = r.json()
    df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in x.items() ]))
    return df

def process_data(df):
    df_re = df.drop(['has_header','venue_country','venue_latitude', 'venue_longitude',
                    'venue_name', 'venue_state'],axis=1)
    df_final = df_re.fillna(0)
    lb_make = LabelEncoder()
    df_final["country_code"] = lb_make.fit_transform(df_final["country"])
    df_final["payout_type_code"] = lb_make.fit_transform(df_final["payout_type"])

    df_re = df_final.drop(['approx_payout_date','country','payout_type','acct_type',
                            'currency','description','email_domain','listed','name',
                            'org_desc','org_name','payee_name','previous_payouts',
                            'ticket_types','venue_address'],axis=1)
    return df_re


def predict():
    """ """
    predicted = model.predict(df)
    return predicted

def predict_proba(predicted):
    if predicted[0][1] > 0.7:
        return "high risk"
    elif predicted[0][1] > 0.5:
        return "medium risk"
    else:
        return "low risk"

def add_to_db(df):
    pass


if __name__ == '__main__':
    #open database - only once

    #run the application
    app.run(host='0.0.0.0', port=8080, debug=True)
