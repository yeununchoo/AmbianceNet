import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

import itertools
import string
import json
import os

from Text_VAD import Text_VAD
from Music_VAD import Music_VAD


#########################################################
##
## Parameters
##
#########################################################

save_figures = True
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
## Load Preprocessed Data
##
#########################################################


###############
##
## Preprocessed Reddit Comments
##
###############


reddit_test_pd = pd.read_parquet(f"{data_path}/reddit_test.parquet")
reddit_test_tokens = np.load(f"{data_path}/reddit_test_tokens.npy")

with open(f"{data_path}/token_dictionary.json", "r") as readfile:
    word_to_index_dict = json.load(readfile)

print(len(word_to_index_dict))
print(list(word_to_index_dict.items())[0:30])


###############
##
## Reddit VAD mapping
##
###############

text_vad_mapping = np.array([
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

text_vad_mapping_pd = pd.DataFrame(columns = ["valence", "arousal", "dominance"], 
                                      data = text_vad_mapping, 
                                      index = emotion_columns)
print(text_vad_mapping_pd.T)


###############
##
## Preprocessed AudioSet music embeddings
##
###############



eval_context_pd = pd.read_parquet(f"{data_path}/eval_music_contexts_full.parquet")
eval_embeddings = np.load(f"{data_path}/eval_music_embeddings.npy")

print(eval_embeddings.shape)


###############
##
## Audio VAD mapping
##
###############

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

print(music_vad_mapping_pd.T)

music_mood_five_dict = dict(
    zip(range(276,283),
        ['Happy', 'Funny', 'Sad', 'Tender', 
         'Happy', 'Angry', 'Angry']
    )
)

#########################################################
##
## Load Pre-trained Models
##
#########################################################

vocab_size = len(word_to_index_dict)
window_size = 30

text_model = Text_VAD(vocab_size, 30)
text_model.load_weights(f"{weights_path}/Text_Weights")

music_model = Music_VAD()
music_model.load_weights(f"{weights_path}/Music_Weights")



#########################################################
##
## Construct the evaluation set
##
#########################################################

text_input_eval = tf.cast(reddit_test_tokens, 
                          dtype = tf.float32)
text_label_eval_pd = reddit_test_pd[emotion_columns].copy()

music_input_eval = tf.cast(eval_embeddings.reshape(-1, 10, 128, 1)/128.0, 
                           dtype = tf.float32)

music_label_eval = [music_mood_five_dict[each_m] 
                    for each_m 
                    in eval_context_pd["mood"].to_numpy()] 



#########################################################
##
## Map Text and Music to VAD
##
#########################################################

text_VAD_predictions = text_model(text_input_eval)
print(text_VAD_predictions.shape)


music_VAD_predictions = music_model(music_input_eval)
print(music_VAD_predictions.shape)

#########################################################
##
## Match Labels for Text and Music
##
#########################################################

music_vad_mapping_five_pd = music_vad_mapping_pd.loc[[
    'Happy music', 'Funny music', 'Sad music', 
    'Tender music', 'Angry music']].copy()

music_vad_mapping_five = music_vad_mapping_five_pd.to_numpy()

# Now we have to creating a mapping between the 28 text emotion labels 
# and the 5 music emotion labels. 

# The way that Won et al. did was to match the labels with the shortest Euclidean distance in the VAD space.

def match_closest_mood(each_text_vad, music_vad_mapping_five):
    """
    :param each_text_vad: the VAD coordinates of each text mood label in np.array of shape (3,)
    :param music_vad_mapping_five: the VAD coordinates of five major music moods in np.array of shape (5, 3)

    :return: closest_mood: the closest music emotion label to the text emotion label
    """
    
    repeated_text_vad = np.repeat(each_text_vad, repeats = 5).reshape(3, 5).T
    # this just repeats the text emotion label's VAD coordinates five times
    # For example 
    # array([[0.875, 0.875, 0.562],
    #    [0.875, 0.875, 0.562],
    #    [0.875, 0.875, 0.562],
    #    [0.875, 0.875, 0.562],
    #    [0.875, 0.875, 0.562]])
    
    distances = np.sum(np.square(music_vad_mapping_five - repeated_text_vad), 
                       axis = 1)
    # squared distances from the five music moods
    # For example 
    # array([0.079325, 0.07209 , 0.886833, 0.188859, 0.570798])
    
    five_text_moods = ['Happy', 'Funny', 'Sad', 
                        'Tender', 'Angry']
    closest_mood = five_text_moods[np.argmin(distances)]
    
    return closest_mood

# Construct the mapping between the original label and the new five-emotion label

text_emotion_five = []
for each_text_vad in text_vad_mapping:
    text_emotion_five.append(match_closest_mood(each_text_vad, music_vad_mapping_five))
text_label_five_dict = dict(zip(emotion_columns, text_emotion_five))
print(text_label_five_dict)


#########################################################
##
## Create matching label lists
##
#########################################################

text_label_five_list = []
for each_i, each_r in text_label_eval_pd.iterrows():
    emotion_set = set()
    for each_e in emotion_columns:
        if each_r[each_e]:
            emotion_set.add(text_label_five_dict[each_e])
    text_label_five_list.append(list(emotion_set))

print(text_label_five_list[0:5])


#########################################################
##
## Match Text Predictions to Music Predictions
##
#########################################################


def find_closest_music(each_text_vad, music_VAD_predictions):
    music_VAD_prediction_size = len(music_VAD_predictions)
    repeated_text_vad = tf.transpose(tf.reshape(tf.repeat(each_text_vad, 
                                            repeats = music_VAD_prediction_size), 
                                 shape = [3, music_VAD_prediction_size]))
    distances = tf.reduce_sum(tf.square(repeated_text_vad - music_VAD_predictions), 
                          axis = 1)
    return tf.argmin(distances).numpy()


def mapping_accuracy(text_VAD_predictions, music_VAD_predictions, 
                     text_label_five_list, music_label_eval):
    accuracy_list = []
    accuracy_pd = pd.DataFrame(
        columns = ["tp", "fp", "fn"], 
        index = ['Happy', 'Funny', 'Sad', 'Tender', 'Angry'],
        data = np.zeros((5, 3), dtype = int))

    five_emotion_list = ['Happy', 'Funny', 'Sad', 'Tender', 'Angry']

    for each_text_index, each_text_vad in enumerate(text_VAD_predictions):
        each_music_index = find_closest_music(each_text_vad, music_VAD_predictions)

        each_text_emotion_label_list = text_label_five_list[each_text_index]
        each_music_emotion_pred = music_label_eval[each_music_index]

        # print(f"{each_text_emotion_label_list}, {each_music_emotion_pred}")
        # print(f"{each_music_emotion_pred in each_text_emotion_label_list}")

        is_correct = each_music_emotion_pred in each_text_emotion_label_list

        accuracy_list.append(is_correct)

        if each_music_emotion_pred in each_text_emotion_label_list:
            accuracy_pd.loc[each_music_emotion_pred, "tp"] += 1
        else:
            accuracy_pd.loc[each_music_emotion_pred, "fp"] += 1

        for each_major_emotion in each_text_emotion_label_list:
            if each_music_emotion_pred != each_major_emotion:
                accuracy_pd.loc[each_major_emotion, "fn"] += 1


    micro_accuracy = np.mean(np.array(accuracy_list))
    return micro_accuracy, accuracy_pd
    


#########################################################
##
## Calculate Accuracy
##
#########################################################

micro_accuracy, accuracy_pd = mapping_accuracy(
    text_VAD_predictions, music_VAD_predictions, 
    text_label_five_list, music_label_eval)


print(f"micro_accuracy: {100*micro_accuracy:2.2f}%")


accuracy_pd.eval("precision = tp/(tp+fp)", inplace = True)
accuracy_pd.eval("recall = tp/(tp+fn)", inplace = True)
accuracy_pd.eval("f1 = 2*precision*recall/(precision+recall)", inplace = True)
macro_precision = accuracy_pd["precision"].mean()
print(f"macro_precision: {100*macro_precision:2.2f}%")

print(accuracy_pd)
accuracy_pd.to_parquet(f"{eval_path}/Connection_pd.parquet")

