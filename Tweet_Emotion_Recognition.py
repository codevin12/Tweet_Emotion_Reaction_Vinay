#!/usr/bin/env python
# coding: utf-8

# ## Tweet Emotion Recognition
# 
# ---
# 
# Dataset: [Tweet Emotion Dataset](https://github.com/dair-ai/emotion_dataset)
# 
# ## Task 1: Introduction
# Natural Language Processing Applications with TensorFlow using LSTM

# 

# ## Task 2: Setup and Imports
# 
# 1. Installing Hugging Face's nlp package
# 2. Importing libraries

# In[ ]:


get_ipython().system('pip install nlp')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random


def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

    
print('Using TensorFlow version', tf.__version__)


# ## Task 3: Importing Data
# 
# 1. Importing the Tweet Emotion dataset
# 2. Creating train, validation and test sets
# 3. Extracting tweets and labels from the examples

# In[ ]:


dataset = nlp.load_dataset('emotion')


# In[ ]:


dataset


# In[ ]:


train = dataset['train']
val = dataset['validation']
test = dataset['test']


# In[ ]:


def get_tweet(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels


# In[ ]:


tweets, labels = get_tweet(train)


# In[ ]:


tweets[5], labels[5]


# ## Task 4: Tokenizer
# 
# 1. Tokenizing the tweets

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[ ]:


tokenizer = Tokenizer(num_words = 10000, oov_token = '<UNK>')
tokenizer.fit_on_texts(tweets)


# In[ ]:


tokenizer.texts_to_sequences([tweets[5]])


# In[ ]:


tweets[5]


# ## Task 5: Padding and Truncating Sequences
# 
# 1. Checking length of the tweets
# 2. Creating padded sequences

# In[ ]:


lengths = [len(t.split(' ')) for t in tweets]
plt.hist(lengths,bins = len(set(lengths)))
plt.show()


# In[ ]:


maxlen = 50

from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating = 'post', padding = 'post', maxlen = maxlen)
    return padded


# In[ ]:


padded_train_seq = get_sequences(tokenizer,tweets)


# In[ ]:


padded_train_seq[5]


# ## Task 6: Preparing the Labels
# 
# 1. Creating classes to index and index to classes dictionaries
# 2. Converting text labels to numeric labels

# In[ ]:


classes = set(labels)
print(classes)


# In[ ]:


plt.hist(labels, bins= 11)
plt.show()


# In[ ]:


class_to_index = dict((c,i) for i,c in enumerate(classes))
index_to_class = dict((v,k) for k,v in class_to_index.items())


# In[ ]:


class_to_index


# In[ ]:


index_to_class


# In[ ]:


names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])


# In[ ]:


train_labels = names_to_ids(labels)
print(train_labels[5])


# ## Task 7: Creating the Model
# 
# 1. Creating the model
# 2. Compiling the model

# In[ ]:


model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000,16,input_length = maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences = True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation = 'softmax')

])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)


# In[ ]:


model.summary()


# ## Task 8: Training the Model
# 
# 1. Preparing a validation set
# 2. Training the model

# In[ ]:


val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)


# In[ ]:


val_tweets[0], val_labels[0]


# In[ ]:


h = model.fit(
    padded_train_seq,train_labels,
    validation_data = (val_seq, val_labels),
    epochs = 20,
    callbacks = [
                 tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 2)
    ]
)


# ## Task 9: Evaluating the Model
# 
# 1. Visualizing training history
# 2. Prepraring a test set
# 3. A look at individual predictions on the test set
# 4. A look at all predictions on the test set

# In[ ]:


show_history(h)


# In[ ]:


test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
test_labels = names_to_ids(test_labels)


# In[ ]:


_ = model.evaluate(test_seq,test_labels)


# In[ ]:


i = random.randint(0, len(test_labels)-1)


# In[ ]:


print(test_tweets[i],'-',index_to_class[test_labels[i]])


# In[ ]:


p = model.predict(np.expand_dims(test_seq[i], axis = 0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]
print('Predicted Emotion : ',pred_class)


# In[ ]:


pred = np.argmax(model.predict(test_seq), axis=-1)


# In[ ]:


show_confusion_matrix(test_labels, pred, list(classes))

