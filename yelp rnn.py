# -*- coding: utf-8 -*-
"""
In this script we will work on yelp review data set and try to predict 
reviews as 1 star or 5 stars using Recurrent Neural Networks
"""
import numpy as np
import pandas as pd
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding,LSTM

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


'''We will only classify a review as 5 stars or 1 star'''
data_mod = data[(data.stars==1) | (data.stars==5)]

'''Splitting features and label'''
X = data_mod['text']
y = data_mod['stars']

'''
Converting 1 star to 0 and 5 star to 1 for the model
'''
y = y.apply(lambda x: (x-1) if x == 1 else 1)

'''
Now we need to process the reviews which are in text to some form which 
the model is able to understand.
We will use tokenizer which will convert the text into sequence 
The num_words indicates it will use the top 20,000 words only '''
from keras.preprocessing.text import Tokenizer
tz = Tokenizer(num_words=20000,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
tz.fit_on_texts(X)
X = tz.texts_to_sequences(X)

'''Limiting reviews to first 300 words to be able to train faster'''
X = sequence.pad_sequences(X, maxlen=300)

'''Splitting into train and test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)


'''Implementing the Recurrent Neural Network'''


'''
Embedding layer converts the input data into dense vectors of fixed size which 
neural network can process better. 20,000 is our vocabulary size which we chose 
in our tokenizer and 128 is the output dimension of 128 units.

Next is LSTM layer here, which stands for Long short term memory which will 
retain words in the review. Dropout prevents overfitting too much on training data.

At the end we have an output layer.'''
classifier = Sequential()
classifier.add(Embedding(20000, 128))
classifier.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
classifier.add(Dense(1, activation='sigmoid'))
classifier.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=25  , epochs=10, validation_data=(X_test, y_test))


'''
This is the last epoch
Epoch 10/10
2737/2737 [==============================] - 58s 21ms/step 
- loss: 0.0186 - acc: 0.9949 - val_loss: 0.4810 - val_acc: 0.9125

Obtained an accuracy of 91.25% with the test set. '''

score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
Test loss: 0.480989840669
Test accuracy: 0.912527798413
'''

