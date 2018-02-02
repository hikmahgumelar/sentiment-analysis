import os, sys, shutil, time

from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
    clf = joblib.load('models/nb')
    tfidf = joblib.load('models/tfidf')

    print('model loaded')

    if request.method == 'POST':

        tweet = request.form['tweet']

        data = [tweet]

        data = tfidf.transform(data)

        prediction = clf.predict(data)

    return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)
