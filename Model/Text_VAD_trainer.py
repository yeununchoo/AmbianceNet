import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

import itertools
import string
import os
import json

from Text_VAD import Text_VAD
from Text_VAD_functions import *


#########################################################
##
## Parameters
##
#########################################################


save_figures = True
data_path = "../Data"
weights_path = "Weights"
figure_path = "../Figures"

# ensures that we run only on cpu
# this environment variable is not permanent
# it is valid only for this session
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#########################################################
##
## Load preprocessed data
##
#########################################################


reddit_train_pd = pd.read_parquet(f"{data_path}/reddit_train.parquet")
reddit_test_pd = pd.read_parquet(f"{data_path}/reddit_test.parquet")
reddit_train_tokens = np.load(f"{data_path}/reddit_train_tokens.npy")
reddit_test_tokens = np.load(f"{data_path}/reddit_test_tokens.npy")

print(reddit_train_pd.head(n = 3))
print(reddit_train_pd.shape)
print(reddit_train_tokens[0:3])
print(reddit_train_tokens.shape)

print(reddit_test_pd.head(n = 3))
print(reddit_test_pd.shape)
print(reddit_test_tokens[0:3])
print(reddit_test_tokens.shape)


#########################################################
##
## Load prebuilt dictionary
##
#########################################################


with open(f"{data_path}/token_dictionary.json", "r") as readfile:
    word_to_index_dict = json.load(readfile)

print(len(word_to_index_dict))
print(list(word_to_index_dict.items())[0:30])

index_to_word_dict = {i:w for w,i in word_to_index_dict.items()}

print(reddit_test_pd.loc[365, "comment_text"])
for i in reddit_test_tokens[365]:
    print(index_to_word_dict[i], end = " ")
    
print(reddit_train_pd.loc[365, "comment_text"])
for i in reddit_train_tokens[365]:
    print(index_to_word_dict[i], end = " ")


#########################################################
##
## vad mapping
##
#########################################################


emotion_vad_mapping = np.array([
    [0.969,0.583,0.726],
    [0.929,0.837,0.803],
    [0.167,0.865,0.657],
    [0.167,0.718,0.342],
    [0.854,0.46,0.889],
    [0.635,0.469,0.5],
    [0.255,0.667,0.277],
    [0.75,0.755,0.463],
    [0.896,0.692,0.647],
    [0.115,0.49,0.336],
    [0.085,0.551,0.367],
    [0.052,0.775,0.317],
    [0.143,0.685,0.226],
    [0.896,0.684,0.731],
    [0.073,0.84,0.293],
    [0.885,0.441,0.61],
    [0.07,0.64,0.474],
    [0.98,0.824,0.794],
    [1,0.519,0.673],
    [0.163,0.915,0.241],
    [0.469,0.184,0.357],
    [0.949,0.565,0.814],
    [0.729,0.634,0.848],
    [0.554,0.51,0.836],
    [0.844,0.278,0.481],
    [0.103,0.673,0.377],
    [0.052,0.288,0.164],
    [0.875,0.875,0.562]])

emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", 
    "neutral", "optimism", "pride", "realization", "relief", 
    "remorse", "sadness","surprise"]

emotion_vad_mapping_pd = pd.DataFrame(columns = ["valence", "arousal", "dominance"], 
                                      data = emotion_vad_mapping, 
                                      index = emotion_columns)
print(emotion_vad_mapping_pd.T)

print(reddit_train_pd.loc[365, emotion_columns])
print(reddit_train_pd.loc[365, ["valence", "arousal", "dominance"]])
print(reddit_test_pd.loc[365, emotion_columns])
print(reddit_test_pd.loc[365, ["valence", "arousal", "dominance"]])


#########################################################
##
### Prepare the input tensors
##
#########################################################

text_input_train = reddit_train_tokens
text_input_test = reddit_test_tokens

VAD_true_train = tf.cast(reddit_train_pd[["valence", "arousal", "dominance"]].to_numpy(), 
                         dtype = tf.float32)
VAD_true_test = tf.cast(reddit_test_pd[["valence", "arousal", "dominance"]].to_numpy(), 
                        dtype = tf.float32)


print(text_input_train.shape)
# print(text_input_train[0:3])

print(text_input_test.shape)
# print(text_input_test[0:3])

print(VAD_true_train.shape)
# print(VAD_true_train[0:3])

print(VAD_true_test.shape)
# print(VAD_true_test[0:3])

sample_weights_train = tf.cast(reddit_train_pd["Weight"].to_numpy(), dtype = tf.float32)
print(sample_weights_train.shape)

sample_weights_test = tf.cast(reddit_test_pd["Weight"].to_numpy(), dtype = tf.float32)
print(sample_weights_test.shape)

onehot_emotion_labels_train = reddit_train_pd[emotion_columns].to_numpy()
print(onehot_emotion_labels_train.shape)
print("")

onehot_emotion_labels_test = reddit_test_pd[emotion_columns].to_numpy()
print(onehot_emotion_labels_test.shape)


#########################################################
##
## Initialize the model and train
##
#########################################################

n_epochs = 3 
vocab_size = len(word_to_index_dict)
window_size = 30

model = Text_VAD(vocab_size, 30)

for each_epoch in range(n_epochs):
    train_loss = train(model, text_input_train, VAD_true_train, sample_weights_train)
    print(f"train_loss = {train_loss}")


#########################################################
##
## test
##
#########################################################


test_loss = test(model, text_input_test, VAD_true_test, sample_weights_test)
print(f"test loss: {test_loss}")


#########################################################
##
## accuracy
## 
#########################################################

test_accuracy, test_accuracy_pd = accuracy(model, 
                                           text_input_test, sample_weights_test, 
                                           onehot_emotion_labels_test, emotion_vad_mapping_pd)
print(f"test accuracy: {100*test_accuracy:2.2f}%")

test_accuracy_pd.eval("precision = tp/(tp+fp)", inplace = True)
test_accuracy_pd.eval("recall = tp/(tp+fn)", inplace = True)
test_accuracy_pd.eval("f1 = 2*precision*recall/(precision+recall)", inplace = True)
macro_precision = test_accuracy_pd["precision"].mean()
print(f"macro_precision: {macro_precision}")


#########################################################
##
## Final Summary 
##
#########################################################


train_pad_proportion = np.sum(text_input_train == 0)/np.prod(text_input_train.shape)
train_unk_proportion = np.sum(text_input_train == 1)/np.prod(text_input_train.shape)
print(f"train_pad_proportion = {train_pad_proportion}")
print(f"train_unk_proportion = {train_unk_proportion}")


test_pad_proportion = np.sum(text_input_test == 0)/np.prod(text_input_test.shape)
test_unk_proportion = np.sum(text_input_test == 1)/np.prod(text_input_test.shape)
print(f"test_pad_proportion = {test_pad_proportion}")
print(f"test_pad_proportion = {test_unk_proportion}")

print(text_input_train.shape, text_input_test.shape)
print(model.summary())


#########################################################
##
## Weights 
##
#########################################################

model.save_weights(f"{weights_path}/Text_Weights")

