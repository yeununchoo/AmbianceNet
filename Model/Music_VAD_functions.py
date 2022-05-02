import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_datasets as tfds

# import adjustText
import itertools
import os
# import csv
# import collections
# import re

# from tqdm.auto import tqdm
save_figures = False
data_path = "../Data"
figure_path = "../Figures"


#########################################################
##
## Customized train, test, accuracy functions
##
#########################################################

def train(model, music_input, VAD_true, sample_weights):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
    :param VAD_true: float tensor, [batch_size x 3]
    :param sample_weights: float tensor, [batch_size x 1]
    
    :return: None
    """

    input_size = len(music_input)
    batch_size = model.batch_size
    
    # drop the rest of the data that do not fit in batches
    input_size = input_size - (input_size%batch_size)
    
    # shuffle
    shuffled_index = tf.random.shuffle(tf.range(input_size))
    shuffled_music_input  = tf.gather(music_input, shuffled_index)
    shuffled_VAD_true = tf.gather(VAD_true, shuffled_index)
    shuffled_sample_weights = tf.gather(sample_weights, shuffled_index)
    
    batch_loss_list = []
    
    for train_index in range(0, input_size, batch_size): 
        batch_music_input = shuffled_music_input[train_index:(train_index + batch_size)]
        batch_VAD_true   = shuffled_VAD_true[train_index:(train_index + batch_size)]
        batch_sample_weights = shuffled_sample_weights[train_index:(train_index + batch_size)]

        with tf.GradientTape() as tape:
            batch_VAD_pred = model(batch_music_input)
            batch_loss = model.loss_function(batch_VAD_true, 
                                             batch_VAD_pred, 
                                             batch_sample_weights)
            
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        batch_loss_list.append(batch_loss)
        
        if train_index%(batch_size*50) == 0:
            print(f"training on index {train_index}, "
                  f"progress {100*(train_index + batch_size)/(input_size):2.1f}%, "
                  f"loss {batch_loss:.4f}")
    
    print(f"last index {train_index}, "
          f"progress {100*(train_index + batch_size)/(input_size):2.1f}%, "
          f"loss {batch_loss:.4f}")    
    
    print("training over")
    
    return tf.reduce_mean(np.array(batch_loss_list))


def test(model, music_input, VAD_true, sample_weights):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
    :param VAD_true: float tensor, [batch_size x 3]
    
    :return: Loss
    """
    input_size = len(music_input)
    # the batch size for test must be different from that for train
    # there are 420 testing data
    batch_size = 84
    
    # drop the rest of the data that do not fit in batches
    input_size = input_size - (input_size%batch_size)
    
    # shuffle
    shuffled_index = tf.random.shuffle(tf.range(input_size))
    shuffled_music_input  = tf.gather(music_input, shuffled_index)
    shuffled_VAD_true = tf.gather(VAD_true, shuffled_index)
    shuffled_sample_weights = tf.gather(sample_weights, shuffled_index)
    
    batch_loss_list = []
    
    for test_index in range(0, input_size, batch_size): 
        batch_music_input = shuffled_music_input[test_index:(test_index + batch_size)]
        batch_VAD_true    = shuffled_VAD_true[test_index:(test_index + batch_size)]
        batch_sample_weights = shuffled_sample_weights[test_index:(test_index + batch_size)]
        
        batch_VAD_pred = model(batch_music_input)
        
        batch_loss = model.loss_function(batch_VAD_true, 
                                         batch_VAD_pred, 
                                         batch_sample_weights)
        batch_loss_list.append(batch_loss)
        
        if test_index%(batch_size) == 0:
            print(f"testing on index {test_index}, "
                  f"progress {100*(test_index + batch_size)/(input_size):2.1f}%, "
                  f"loss {batch_loss:.4f}")

    print(f"last index {test_index}, "
          f"progress {100*(test_index + batch_size)/(input_size):2.1f}%, "
          f"loss {batch_loss:.4f}")    
    
    print("testing over")

    return tf.reduce_mean(np.array(batch_loss_list))


def get_closest_mood(VAD_pred, music_vad_mapping):

    repeated_VAD_pred = np.repeat(VAD_pred, repeats = 7).reshape(3, 7).T
    distances = np.sum(np.square(music_vad_mapping - repeated_VAD_pred), axis = 1)
    closest_mood = 276 + np.argmin(distances)
    
    return closest_mood



def merge_mood(each_mood):
    if each_mood == 280:
        each_mood = 276
    elif each_mood == 282:
        each_mood = 281
    return each_mood


def accuracy(model, music_input, emotion_labels, music_vad_mapping, mood_dict):
    """
    Runs through one epoch - all training examples.

    Predict the VAD coordinate for the input music
    Find the closest emotion label on the VAD space
    See if the model has chosen the correct emotion label

    :param model: the initialized model to use for forward and backward pass
    :param VGGish_input: batched input of the VGGish embeddings, [batch_size x 10 x 128 x 1]
    :param emotion_labels: boolean tensor, [batch_size, ]
    
    :return: average accuracy
    """
    
    input_size = len(music_input)

    mood_pred_list = []
    for music_index, each_music in enumerate(music_input):
        each_VAD_pred = model(tf.reshape(each_music, [-1, 10, 128, 1]))
        
        each_closest_mood = get_closest_mood(each_VAD_pred, music_vad_mapping)
        mood_pred_list.append(each_closest_mood)
        
    mood_pred_array = np.array(mood_pred_list)

    
    mood_accuracy_list = []
    accuracy_pd = pd.DataFrame(
        columns = ["tp", "fp", "fn"], 
        index = ["Happy music", "Funny music", "Sad music", "Tender music", "Angry music"],
        data = np.zeros((5, 3), dtype = int))
    
    for each_pred_mood, each_true_mood in zip(mood_pred_array, emotion_labels):
        pred_label = mood_dict[merge_mood(each_pred_mood)]
        true_label = mood_dict[merge_mood(each_true_mood)]
        mood_accuracy_list.append(pred_label == true_label)
        
        if pred_label == true_label:
            accuracy_pd.loc[pred_label, "tp"] += 1
        else:
            accuracy_pd.loc[pred_label, "fp"] += 1
            accuracy_pd.loc[true_label, "fn"] += 1
            
        
    mood_accuracy_array = np.array(mood_accuracy_list)

    return np.mean(mood_accuracy_array), accuracy_pd
