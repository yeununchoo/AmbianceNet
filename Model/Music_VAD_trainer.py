import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

import itertools
import os

from Music_VAD import Music_VAD
from Music_VAD_functions import *


#########################################################
##
## Parameters
##
#########################################################


save_figures = False
data_path = "../Data"
weights_path = "Weights"
figure_path = "../Figures"
eval_path = "Evaluations"

# ensures that we run only on cpu
# this environment variable is not permanent
# it is valid only for this session
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#########################################################
##
## Load preprocessed data
##
#########################################################


mood_dict = dict(
    zip(range(276,283),
        ['Happy music', 'Funny music', 'Sad music', 'Tender music', 
         'Exciting music', 'Angry music', 'Scary music']
    )
)
print(mood_dict)


bal_train_context_pd = pd.read_parquet(f"{data_path}/bal_train_music_contexts_full.parquet")
bal_train_embeddings = np.load(f"{data_path}/bal_train_music_embeddings.npy")

eval_context_pd = pd.read_parquet(f"{data_path}/eval_music_contexts_full.parquet")
eval_embeddings = np.load(f"{data_path}/eval_music_embeddings.npy")

unbal_train_context_pd = pd.read_parquet(f"{data_path}/unbal_train_music_contexts_full.parquet")
unbal_train_embeddings = np.load(f"{data_path}/unbal_train_music_embeddings.npy")

print(bal_train_embeddings.shape)
print(eval_embeddings.shape)
print(unbal_train_embeddings.shape)

print(bal_train_context_pd.head())
print(bal_train_context_pd.tail())

print(eval_context_pd.head())
print(eval_context_pd.tail())

print(unbal_train_context_pd.head())
print(unbal_train_context_pd.tail())


#########################################################
##
### VAD mapping
##
#########################################################


music_vad_mapping = np.array([
    [1,0.735,0.772],
    [0.918,0.61,0.566],
    [0.225,0.333,0.149],
    [0.63,0.52,0.509],
    [0.95,0.792,0.789],
    [0.122,0.83,0.604],
    [0.062,0.952,0.528],
])

music_vad_mapping_pd = pd.DataFrame(
    columns = ["valence", "arousal", "dominance"], 
    data = music_vad_mapping, 
    index = ['Happy music', 'Funny music', 'Sad music', 'Tender music', 
             'Exciting music', 'Angry music', 'Scary music'])

print(music_vad_mapping_pd)


#########################################################
##
### Prepare the input tensors
##
#########################################################


VGGish_input_train = tf.cast(unbal_train_embeddings.reshape(-1, 10, 128, 1)/128.0, 
                             dtype = tf.float32)
VGGish_input_test = tf.cast(eval_embeddings.reshape(-1, 10, 128, 1)/128.0, 
                            dtype = tf.float32)
VAD_true_train = tf.cast(unbal_train_context_pd[["valence", "arousal", "dominance"]].to_numpy(), 
                         dtype = tf.float32)
VAD_true_test = tf.cast(eval_context_pd[["valence", "arousal", "dominance"]].to_numpy(), 
                        dtype = tf.float32)
print(VGGish_input_train.shape)
print(VGGish_input_test.shape)
print(VAD_true_train.shape)
print(VAD_true_test.shape)


sample_weights_train = tf.cast(unbal_train_context_pd["Weight"].to_numpy(), 
                               dtype = tf.float32)
sample_weights_test = tf.cast(eval_context_pd["Weight"].to_numpy(), 
                              dtype = tf.float32)
emotion_labels_train = unbal_train_context_pd["mood"].to_numpy()
emotion_labels_test = eval_context_pd["mood"].to_numpy()
print(sample_weights_train.shape)
print(sample_weights_test.shape)
print(emotion_labels_train.shape)
print(emotion_labels_test.shape)

#########################################################
##
## Initialize the model and train
##
#########################################################

n_epochs = 5 ## how many epochs are needed???

model = Music_VAD()

for each_epoch in range(n_epochs):
    # slight input augmentation
    # add some gaussian noise
    # trying to prevent overfitting
    VGGish_input_train_augmented = (
        VGGish_input_train + tf.random.normal(VGGish_input_train.shape, 
                                              mean = 0, 
                                              stddev = 0.05, 
                                              dtype = tf.float32))
    
    train_loss = train(model, 
                       VGGish_input_train_augmented, VAD_true_train, 
                       sample_weights_train)
    print(f"train_loss = {train_loss}", "\n")


#########################################################
##
## test
##
#########################################################


test_loss = test(model, 
                 VGGish_input_test, VAD_true_test,
                 sample_weights_test)
print(f"test loss: {test_loss}")


#########################################################
##
## accuracy
## 
#########################################################

test_accuracy, test_accuracy_pd = accuracy(model, 
                                           VGGish_input_test, emotion_labels_test, 
                                           music_vad_mapping_pd, mood_dict)

print(f"test accuracy: {100*test_accuracy:2.2f}%")

test_accuracy_pd.eval("precision = tp/(tp+fp)", inplace = True)
test_accuracy_pd.eval("recall = tp/(tp+fn)", inplace = True)
test_accuracy_pd.eval("f1 = 2*precision*recall/(precision+recall)", inplace = True)
macro_precision = test_accuracy_pd["precision"].mean()
print(f"macro_precision = {macro_precision}")


#########################################################
##
## Final Summary and weights
##
#########################################################

print(model.summary())
model.save_weights(f"{weights_path}/Music_Weights")
test_accuracy_pd.to_parquet(f"{eval_path}/Music_pd.parquet")

