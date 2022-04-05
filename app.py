from flask import Flask, render_template,request
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
app = Flask(__name__)

nltk.download('vader_lexicon')
# two decorators, same function
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-sentiment')
def check():
    statement=request.form.get('statement')
    print(statement)
    sid=SentimentIntensityAnalyzer()
    sa=sid.polarity_scores(str(statement))
    for k in sa:
            print(k,sa[k])
    return sa

if __name__ == '__main__':
    app.run(debug=True)