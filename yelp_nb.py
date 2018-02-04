# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:44:14 2018

@author: swaraj
"""
''' In this script we will work on yelp review data set and try to classify 
reviews from 1-5 stars based on the reviews using NLP '''
import numpy as np
import pandas as pd

data = pd.read_csv('yelp.csv')

data.head()


'''
  business_id               date               review_id           stars  
0  9yKzy9PApeiPPOUJEtnvkg  2011-01-26  fWKvX83p0-ka4JS3dc6E5A      5   
1  ZRJwVLyzEJq1VAihDhYiow  2011-07-27  IjZ33sJrzXqU-0X6U8NwyA      5   
2  6oRAC4uyJCsJl1X0WZpVSA  2012-06-14  IESLBzqUCLdSzSqm0eCSxQ      4   
3  _1QQZuf4zZOyFCvXc0o6Vg  2010-05-27  G-WvGaISbqqaMHlNnByodA      5   
4  6ozycU1RpktNG2-1BroVtw  2012-01-05  1uJFq2r5QfJG_6ExMRCaGw      5   

                        text                            type
0  My wife took me here on my birthday for breakf...  review   
1  I have no idea why some people give bad review...  review   
2  love the gyro plate. Rice is so good and I als...  review   
3  Rosie, Dakota, and I LOVE Chaparral Dog Park!!...  review   
4  General Manager Scott Petello is a good egg!!!...  review   

                  user_id  cool  useful  funny  
0  rLtl8ZkDX5vH5nAx9C3q5Q     2       5      0  
1  0a2KyEL0d3Yb1V6aivbIuQ     0       0      0  
2  0hT2KtfLiobPvh6cDC8JQg     0       1      0  
3  uZetl9T0NcROGOyFfughhg     1       2      0  
4  vYmM4KTsC8ZfQBg-j5MWkw     0       0      0  

'''

data.info()

'''RangeIndex: 10000 entries, 0 to 9999
Data columns (total 10 columns):
business_id    10000 non-null object
date           10000 non-null object
review_id      10000 non-null object
stars          10000 non-null int64
text           10000 non-null object
type           10000 non-null object
user_id        10000 non-null object
cool           10000 non-null int64
useful         10000 non-null int64
funny          10000 non-null int64
dtypes: int64(4), object(6)
'''

'''No missing data here!'''

data['review length'] = data['text'].apply(len)
stars = data.groupby('stars').mean()

'''We will only classify a review as 5 stars or 1 star'''
data_mod = data[(data.stars==1) | (data.stars==5)]

'''Splitting features and label'''
X = data_mod['text']
y = data_mod['stars']


'''
Now we need to process the reviews which are in text to some form which 
the model is able to understand. We will use tokenizer which will convert 
the text into a sparse matrix indicating the words presence in Bag of words'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

'''Splitting into train and test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

'''Implementing Naive Bayes'''
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

'''Using cross validation for getting a better idea about the model'''
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(nb, X, y, cv=10)
cv_scores.mean()

'''0.92584967948480101 Obtained an accuracy of 92.5% using Multinomial Naive Bayes Classifier'''

