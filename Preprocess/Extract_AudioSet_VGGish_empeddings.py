#########################################################################
# AudioSet VGGish Embeddings
#########################################################################

# Rule of the Internet: if you want to do something, someone else has already done it before.

# We should reference this random guy at Google

# - https://groups.google.com/g/audioset-users/c/wyyH4MqaboM
# - https://colab.research.google.com/drive/1BeORlWolTKw3noASvW94OXXqcQZ8PZEQ#scrollTo=r7BmTVx1tn_8

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_datasets as tfds

import itertools
import os
import csv
# import collections
# import re

# from tqdm.auto import tqdm

# Before you run the cells below, you need to go to the AudioSet website first,
# and then you need to manually download the tar.gz file.

# - https://research.google.com/audioset/download.html
# - storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz

## Load TF record files

# tfrecord_directory = "bal_train"
# tfrecord_directory = "eval"
tfrecord_directory = "unbal_train"
tfrecord_filenames = os.listdir(tfrecord_directory)
tfrecord_filenames = [(tfrecord_directory + "/" + each_fname) 
                      for each_fname 
                      in tfrecord_filenames]

print(f"Number of TF record files: {len(tfrecord_filenames)}")
print(tfrecord_filenames[:20])

# NOT a problem on a Linux machine
#
# # practice with a small number of files first
# # it is not possible to load the entire dataset in a Windows machine anyway

# # bal_train: all 1444 available on Windows --> 7970 observations
# # eval     : all 1444 available on Windows --> 7329 observations
# # unbal_train:   first 20 files on Windows --> 19502 observations
tfrecord_filenames = tfrecord_filenames[:20]

# This is the dataset before parsing.

raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames)

# %%time
# for i, example in enumerate(raw_dataset):
#     if (i%100_000) == 0:
#         print(i)
    
# print(i)

# for i, raw_record in enumerate(raw_dataset.take(1)):
#     example = tf.train.SequenceExample()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
#     print("\n\n")

############################################################
## Parse the TF record files
############################################################

# Create a description of the features.
context = {
    'end_time_seconds': tf.io.FixedLenFeature([], tf.float32, 
                                              default_value = 0.0),
    'video_id': tf.io.FixedLenFeature([], tf.string, 
                                      default_value = ''),
    'start_time_seconds': tf.io.FixedLenFeature([], tf.float32, 
                                                default_value = 0.0),
    'labels': tf.io.VarLenFeature(tf.int64),
}

sequence = {
    'audio_embedding': tf.io.FixedLenSequenceFeature([], tf.string, 
                                                     default_value = None,
                                                     allow_missing = True)
}

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_sequence_example(example_proto, 
                                               context_features = context, 
                                               sequence_features = sequence)

parsed_dataset = raw_dataset.map(_parse_function)

############################################################
## Extract the music embeddings
############################################################

############################################################
### Class code to mood name
############################################################

with open('class_labels_indices.csv', encoding='utf-8') as class_map_csv:
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header

#class_names = np.array(class_names)
print(class_names[276:283])


############################################################
### Visualize a few examples
############################################################

# def visualize_embedding(music_class_label):
#     music_count = 0
    
#     for i, example in enumerate(parsed_dataset):
#         if (i%100_000) == 0:
#             print(f"i = {i}")

#         context, sequence = example
#         labels = context['labels'].values.numpy()

#         if (music_class_label in labels):
#             raw_embedding = sequence['audio_embedding'].numpy()
#             embedding = tf.io.decode_raw(raw_embedding, tf.int8).numpy()

#             plt.title(str(class_names[labels]))
#             plt.imshow(embedding, 
#                        cmap = 'BrBG')
#             plt.xlabel("VGGish Embedding")
#             plt.ylabel("Seconds")
#             plt.show()

#             music_count += 1
        
#         # this will give us THREE, not tow, visualizations
#         # because Python counts from zero
#         if music_count > 2:
#             break

#     print(f"i = {i}")
#     print(f"music_count = {music_count}")
    

# %%time

# for each_label in range(276, 283):
#     visualize_embedding(each_label)
#     print("\n\n")

def extract_embedding(music_class_label):
    music_contexts = []
    music_embeddings = []

    for i, example in enumerate(parsed_dataset):
        context, sequence = example
        labels = context['labels'].values.numpy()

        if (music_class_label in labels):
            raw_embedding = sequence['audio_embedding'].numpy()
            embedding = tf.io.decode_raw(raw_embedding, tf.int8).numpy()

            if embedding.shape != (10, 128):
                # ideally, all audio clips are exactly 10 seconds long
                # however, some are shorter than 10 seconds. like 7 seconds for example
                # so we pad the embedding with zero values to make the shape uniform

                # print(f"embedding old shape = {embedding.shape}")

                zero_padding = np.zeros((10 - embedding.shape[0], 128), dtype = 'int8')
                embedding = np.concatenate((embedding, zero_padding), axis = 0)

                # print(f"embedding new shape = {embedding.shape}")

            music_embeddings.append(embedding)

            music_context = (context['video_id'].numpy(), 
                             context['start_time_seconds'].numpy(), 
                             context['end_time_seconds'].numpy())

            music_contexts.append(music_context)

        it is a long process
        good go know how much of the for loop is completed
        if (i%10_000) == 0:
            print(f"i = {i}")
        
    print(f"i = {i}")
    music_embeddings = np.array(music_embeddings)
    
    return music_contexts, music_embeddings

############################################################
### VAP mapping
############################################################

# NRC Word-Emotion Association Lexicon

# Source: https://saifmohammad.com/WebPages/nrc-vad.html

# We can't include the lexicon itself, 
# because the non-commercial research license forbids redistributing.

VAP_mapping = {
    276:(1,0.735,0.772),
    277:(0.918,0.61,0.566),
    278:(0.225,0.333,0.149),
    279:(0.63,0.52,0.509),
    280:(0.95,0.792,0.789),
    281:(0.122,0.83,0.604),
    282:(0.062,0.952,0.528),
}

############################################################
## Put all extractions together
############################################################

music_context_pd_before_concat = []
music_embeddings_before_concat = []

for each_class in range(276, 283):
    music_contexts, music_embeddings = extract_embedding(each_class)
    
    (music_youtube_ids, music_start_times, music_end_times) = tuple(zip(*music_contexts))

    music_context_pd = pd.DataFrame(data = {"youtube_id": music_youtube_ids, 
                                            "start_time": music_start_times, 
                                            "end_time"  : music_end_times})
    music_context_pd["mood"] = each_class
    
    valence, arousal, dominance = VAP_mapping[each_class]
    music_context_pd["valence"] = valence
    music_context_pd["arousal"] = arousal
    music_context_pd["dominance"] = dominance
    
    print("music embeddings")
    print(music_embeddings.shape)
    print(music_embeddings.dtype)

    print("")
    print("music contexts")
    print(music_context_pd.shape)
    print(music_context_pd.dtypes)
    print(music_context_pd.head())
    
    music_context_pd_before_concat.append(music_context_pd)
    music_embeddings_before_concat.append(music_embeddings)

music_context_all_moods_pd = pd.concat(music_context_pd_before_concat, 
                                       axis = 0, 
                                       ignore_index = True)

music_embbedings_all_moods = np.concatenate(music_embeddings_before_concat, 
                                            axis = 0)

print("music_embbedings_all_moods")
print(music_embbedings_all_moods.shape)
print(music_embbedings_all_moods.dtype)

print("")
print("music_context_all_moods_pd")
print(music_context_all_moods_pd.shape)
print(music_context_all_moods_pd.dtypes)
print(music_context_all_moods_pd.head())
print(music_context_all_moods_pd.tail())

## Export the extracted embeddings

context_filename = f"music_mood/{tfrecord_directory}_music_contexts.parquet"
embeddings_filename = f"music_mood/{tfrecord_directory}_music_embeddings.npy"

music_context_all_moods_pd.to_parquet(context_filename)

with open(embeddings_filename, "wb") as f:
    np.save(f, 
            music_embbedings_all_moods, 
            allow_pickle = False)
    
