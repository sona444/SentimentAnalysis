import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
file="D:\SentimentAnalysis\Covid_data\covid-data.xlsx"

dfs=pd.read_excel(file)
print(dfs)
sid=SentimentIntensityAnalyzer()
for index,data in dfs.iterrows()
    data=data[0].split('.  ')
    for sa in data:
        sa=sid.polarity_scores(str(data))
        for k in sa:
            print(k,sa[k])