import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_datasets as tfds

import itertools
import string
import os
# import adjustText 
# import collections
# import re
import json

# from tqdm.auto import tqdm

save_figures = True
data_path = "../Data"
figure_path = "../Figures"


def train(model, text_input, VAD_true, sample_weights):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
    :param VAD_true: float tensor, [batch_size x 3]
    :param sample_weights: float tensor, [batch_size x 1]
    
    :return: None
    """

    input_size = len(text_input)
    batch_size = model.batch_size
    
    # drop the rest of the data that do not fit in batches
    input_size = input_size - (input_size%batch_size)
    
    # shuffle
    shuffled_index = tf.random.shuffle(tf.range(input_size))
    shuffled_text_input  = tf.gather(text_input, shuffled_index)
    shuffled_VAD_true = tf.gather(VAD_true, shuffled_index)
    shuffled_sample_weights = tf.gather(sample_weights, shuffled_index)
    
    batch_loss_list = []
    
    for train_index in range(0, input_size, batch_size): 
        batch_text_input = shuffled_text_input[train_index:(train_index + batch_size)]
        batch_VAD_true   = shuffled_VAD_true[train_index:(train_index + batch_size)]
        batch_sample_weights = shuffled_sample_weights[train_index:(train_index + batch_size)]

        with tf.GradientTape() as tape:
            batch_VAD_pred = model(batch_text_input)
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



def test(model, text_input, VAD_true, sample_weights):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
    :param VAD_true: float tensor, [batch_size x 3]
    
    :return: Loss
    """
    input_size = len(text_input)
    batch_size = model.batch_size
    
    # drop the rest of the data that do not fit in batches
    input_size = input_size - (input_size%batch_size)
    
    # shuffle
    shuffled_index = tf.random.shuffle(tf.range(input_size))
    shuffled_text_input = tf.gather(text_input, shuffled_index)
    shuffled_VAD_true = tf.gather(VAD_true, shuffled_index)
    shuffled_sample_weights = tf.gather(sample_weights, shuffled_index)
    
    batch_loss_list = []
    
    for test_index in range(0, input_size, batch_size): 
        batch_text_input = shuffled_text_input[test_index:(test_index + batch_size)]
        batch_VAD_true   = shuffled_VAD_true[test_index:(test_index + batch_size)]
        batch_sample_weights = shuffled_sample_weights[test_index:(test_index + batch_size)]
        
        batch_VAD_pred = model(batch_text_input)
        batch_loss = model.loss_function(batch_VAD_true, 
                                         batch_VAD_pred, 
                                         batch_sample_weights)
        batch_loss_list.append(batch_loss)
        
        if test_index%(batch_size*50) == 0:
            print(f"testing on index {test_index}, "
                  f"progress {100*(test_index + batch_size)/(input_size):2.1f}%, "
                  f"loss {batch_loss:.4f}")

    print(f"last index {test_index}, "
          f"progress {100*(test_index + batch_size)/(input_size):2.1f}%, "
          f"loss {batch_loss:.4f}")    
    
    print("testing over")

    return tf.reduce_mean(np.array(batch_loss_list))


def accuracy(model, text_input, sample_weights, onehot_emotion_labels, emotion_vad_mapping_pd):
    """
    Runs through one epoch - all training examples.

    Predict the VAD coordinate for the input text
    Find the closest emotion label on the VAD space
    See if the model has chosen the correct emotion label

    :param model: the initialized model to use for forward and backward pass
    :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
    :param onehot_emotion_labels: boolean tensor, [batch_size x 28]
    
    :return: average accuracy
    """
    input_size = len(text_input)
    
    # map out ekman categories as the average of the included vectors
    emo_list = emotion_vad_mapping_pd.index
    emo_vecs = emotion_vad_mapping_pd.to_numpy()
    
    reverse_ekman_dict = {
        "anger": "anger", "annoyance": "anger","disapproval": "anger",
        "disgust": "disgust",
        "fear": "fear", "nervousness": "fear",
        "joy": "joy", "amusement": "joy", "approval": "joy", 
        "excitement": "joy", "gratitude": "joy", "love": "joy", 
        "optimism": "joy", "relief": "joy", "pride": "joy", 
        "admiration": "joy", "desire": "joy","caring": "joy",
        "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness", 
        "grief": "sadness", "remorse": "sadness",
        "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
        "curiosity": "surprise",
        "neutral": "neutral",}
    
    accuracy_pd = pd.DataFrame(
        columns = ["tp", "fp", "fn"], 
        index = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
        data = np.zeros((7, 3), dtype = int))
    
    accuracy = 0
    for i, text in enumerate(text_input):
        VAD_pred = model(np.array([text]))
            
        repeated_VAD_pred = np.repeat(VAD_pred, repeats = 28).reshape(3, 28).T
        euc_dists = np.sum(np.square(emo_vecs - repeated_VAD_pred), 
                           axis = 1)
            
        smallest_dist = np.argmin(euc_dists, axis=0)
        closest_emotion = emo_list[smallest_dist]
        
        label_vec = onehot_emotion_labels[i]
        label_inds = np.nonzero(label_vec)
        
        closest_ekman = reverse_ekman_dict[closest_emotion]
        label_ekman_set = set([reverse_ekman_dict[each_emo] 
                               for each_emo in emo_list[label_inds]])
        
        if closest_ekman in label_ekman_set:
            accuracy += 1
            

        if closest_ekman in label_ekman_set:
            accuracy_pd.loc[closest_ekman, "tp"] += 1
        else:
            accuracy_pd.loc[closest_ekman, "fp"] += 1
            
        for each_label_ekman in label_ekman_set:
            if closest_ekman != label_ekman_set:
                accuracy_pd.loc[each_label_ekman, "fn"] += 1
            

        if i%750 == 0:
            print(f"evaluating on index {i}, "
                  f"progress {100*(i/input_size):2.1f}%")

    print(f"last index {i}, "
          f"progress {100*(i/input_size):2.1f}%")
    
    print("evaluation over")
        
    accuracy = accuracy / input_size

    return accuracy, accuracy_pd

