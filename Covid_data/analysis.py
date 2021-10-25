import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
y = x*2

# first plot with X and Y data
plt.plot(x, y)

x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]

# second plot with x1 and y1 data
plt.plot(x1, y1, '-.')

plt.xlabel("X-axis data")
plt.ylabel("Y-axis data")
plt.title('multiple plots')
plt.show()

nltk.downloader.download('vader_lexicon')
file="D:\SentimentAnalysis\Covid_data\covid-data.xlsx"

dfs=pd.read_excel(file)
print(dfs)
sid=SentimentIntensityAnalyzer()
for index,data in dfs.iterrows():
    data=data[0].split('.  ')
    for sa in data:
        sa=sid.polarity_scores(str(data))
        for k in sa:
            print(k,sa[k])