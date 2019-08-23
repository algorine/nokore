import pandas as pd
import random


########### **Load Email Data**

### Read-in the emails and print some basic statistics

Nsamp = 5000
max_cells = 50
maxlen = 20

# Install Simon
#!pip install git+https://github.com/algorine/simon
#from Simon.LengthStandardizer import DataLengthStandardizerRaw

# Enron
EnronEmails = pd.read_csv('data/enron_emails_body.csv',dtype='str', header=0)
print("The size of the Enron emails dataframe is:")
print(EnronEmails.shape)
print("Ten Enron emails are:")
print(EnronEmails.loc[:10])

# Spam
SpamEmails = pd.read_csv('data/fraudulent_emails_body.csv',encoding="ISO-8859-1",dtype='str', header=0)
print("The size of the Spam emails dataframe is:")
print(SpamEmails.shape)
print("Ten Spam emails are:")
print(SpamEmails.loc[:10])

# Convert everything to lower-case, put one sentence per column in a tabular
# structure, truncate to max_cells...
ProcessedEnronEmails=[row.lower().split('\n')[:max_cells] for row in EnronEmails.iloc[:,1]]
#print("3 Enron emails after Processing (in list form) are:")
#print((ProcessedEnronEmails[:3]))
EnronEmails = pd.DataFrame(random.sample(ProcessedEnronEmails,Nsamp)).transpose()


#EnronEmails = DataLengthStandardizerRaw(EnronEmails,max_cells)


#print("Ten Enron emails after Processing (in DataFrame form) are:")
#print((EnronEmails[:10]))
print("Enron email dataframe after Processing shape:")
print(EnronEmails.shape)

ProcessedSpamEmails=[row.lower().split('/n')[:max_cells] for row in SpamEmails.iloc[:,1]]
#print("3 Spam emails after Processing (in list form) are:")
#print((ProcessedSpamEmails[:3]))
SpamEmails = pd.DataFrame(random.sample(ProcessedSpamEmails,Nsamp)).transpose()


#SpamEmails = DataLengthStandardizerRaw(SpamEmails,max_cells)


#print("Ten Spam emails after Processing (in DataFrame form) are:")
#print((SpamEmails[:10]))
print("Spam email dataframe after Processing shape:")
print(SpamEmails.shape)


################ **Preparation of Keras-Bert**

# Constants
SEQ_LEN = 128
BATCH_SIZE = 32 # 64 seems to be the maximum possible on Kaggle
EPOCHS = 10
LR = 1e-4

# Environment
import os

pretrained_path = '../../../BERT_EXPERIMENTS/bert_model'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'

# Load Basic Model
import codecs
from keras_bert import load_trained_model_from_checkpoint

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

import tensorflow as tf

############# **Convert Data to appropriate Array(s) for Keras-Bert input**
import numpy as np

raw_data = np.column_stack((SpamEmails,EnronEmails)).T
print("DEBUG::raw_data:")
print(raw_data.shape)

# corresponding labels
Categories = ['spam','notspam']
header = ([0]*Nsamp)
header.extend(([1]*Nsamp))

import os

from keras_bert import Tokenizer

tokenizer = Tokenizer(token_dict)

# function for processing data into the right format
def load_data(raw_data,header):
    global tokenizer
    indices, labels = [], []
    for i in range(raw_data.shape[0]):
        out=''
        for text in raw_data[i,:]:
            out = str(text)[:maxlen]+out
        ids, segments = tokenizer.encode(out, max_len=SEQ_LEN)
        indices.append(ids)
        labels.append(header[i])
        #print(i)
    items = list(zip(indices, labels))
    np.random.shuffle(items)
    indices, labels = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(labels)


# shuffle raw data first
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(b))
    data = a[p,:]
    header = np.asarray(b)[p]
    return data, list(header)

raw_data, header = unison_shuffled_copies(raw_data, header)


idx = int(0.9*raw_data.shape[0])
train_x, train_y = load_data(raw_data[:idx,:],header[:idx]) # 90% of data for training
test_x, test_y = load_data(raw_data[idx:,:],header[idx:]) # remaining 10% for testing

print("train_x/train_y list details, to make sure it is of the right form:")
print(len(train_x))
print(train_x)
print(train_y[:5])
print(train_y.shape)

############### **Freeze some layers (potentially), Train and Predict**
# Build Custom Model
from tensorflow.python import keras
from keras_bert import AdamWarmup, calc_train_steps

inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

decay_steps, warmup_steps = calc_train_steps(
    train_y.shape[0],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)

model = keras.models.Model(inputs, outputs)

#freeze some layers
FREEZE=True
if FREEZE:
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True

model.compile(
    AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

# @title Initialize Variables
import tensorflow as tf
import tensorflow.keras.backend as K

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)

from keras_bert import get_custom_objects

# Fit
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

# Predict
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    predicts = model.predict(test_x, verbose=True).argmax(axis=-1)

# Accuracy
print(np.sum(test_y == predicts) / test_y.shape[0])