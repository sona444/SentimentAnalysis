import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
statement='I am happy'
sid=SentimentIntensityAnalyzer()
sa=sid.polarity_scores(str(statement))
print(sa)