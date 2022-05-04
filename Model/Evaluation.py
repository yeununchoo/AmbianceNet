################################################
#
# Model Evaluation
#
################################################
import numpy as np
import pandas as pd

import os
import time

eval_path = "Evaluations"


################################################
#
# Repeat
#
################################################


Music_pd_list = []
Text_pd_list = []
Connection_pd_list = []

for each_repeat in range(10):
    print(f"repeat = {each_repeat}")
    #
    # Music
    #
    music_start = time.time()
    
    os.system("python Music_VAD_trainer.py")
    Music_pd = pd.read_parquet(f"{eval_path}/Music_pd.parquet")
    Music_pd_list.append(Music_pd)

    music_end = time.time()
    print(f"Music, time = {music_end - music_start:.3f} seconds")
    
    #
    # Text
    #
    text_start = time.time()

    os.system("python Text_VAD_trainer.py")
    Text_pd = pd.read_parquet(f"{eval_path}/Text_pd.parquet")
    Text_pd_list.append(Text_pd)
    
    text_end = time.time()
    print(f"Text Model, time = {text_end - text_start:.3f} seconds")
    
    #
    # Connection
    #
    connection_start = time.time()

    os.system("python VAD_Connection.py")
    Connection_pd = pd.read_parquet(f"{eval_path}/Connection_pd.parquet")
    Connection_pd_list.append(Connection_pd)
    
    connection_end = time.time()
    print(f"Connection, time = {connection_end - connection_start:.3f} seconds")


################################################
#
# Music Model
#
################################################


music_micro_list = []
music_macro_list = []

for each_Music_pd in Music_pd_list:
    music_micro = (each_Music_pd["tp"].sum()
                   /(each_Music_pd["tp"].sum() + each_Music_pd["fp"].sum()))
    music_macro = each_Music_pd["precision"].mean()

    print(f"music_micro: {100*music_micro:2.2f}")
    print(f"music_macro: {100*music_macro:2.2f}\n")
    
    music_micro_list.append(music_micro)
    music_macro_list.append(music_macro)


################################################
#
# Text Model
#
################################################


text_micro_list = []
text_macro_list = []

for each_Text_pd in Text_pd_list:
    text_micro = (each_Text_pd["tp"].sum()
                  /(each_Text_pd["tp"].sum() + each_Text_pd["fp"].sum()))
    text_macro = each_Text_pd["precision"].mean()

    print(f"text_micro: {100*text_micro:2.2f}")
    print(f"text_macro: {100*text_macro:2.2f}\n")
    
    text_micro_list.append(text_micro)
    text_macro_list.append(text_macro)


################################################
#
# Connection
#
################################################


connection_micro_list = []
connection_macro_list = []

for each_Connection_pd in Connection_pd_list:
    connection_micro = (each_Connection_pd["tp"].sum()
                        /(each_Connection_pd["tp"].sum() + each_Connection_pd["fp"].sum()))
    connection_macro = each_Connection_pd["precision"].mean()

    print(f"connection_micro: {100*connection_micro:2.2f}")
    print(f"connection_macro: {100*connection_macro:2.2f}\n")
    
    connection_micro_list.append(connection_micro)
    connection_macro_list.append(connection_macro)


################################################
#
# Export Everything
#
################################################

    
for index, each_Music_pd in enumerate(Music_pd_list):
    each_Music_pd.to_csv(f"{eval_path}/Music_accuracy{index:02d}.csv")

for index, each_Text_pd in enumerate(Text_pd_list):
    each_Text_pd.to_csv(f"{eval_path}/Text_accuracy{index:02d}.csv")

for index, each_Connection_pd in enumerate(Connection_pd_list):
    each_Connection_pd.to_csv(f"{eval_path}/Connection_accuracy{index:02d}.csv")


precisions_pd = pd.DataFrame(
    {"music_micro": np.array(music_micro_list),
     "music_macro": np.array(music_macro_list),
     "text_micro": np.array(text_micro_list),
     "text_macro": np.array(text_macro_list),
     "connection_micro": np.array(connection_micro_list),
     "connection_macro": np.array(connection_macro_list)})

precisions_pd.loc["mean"] = precisions_pd.mean()
precisions_pd.loc["std"] = precisions_pd.std()

precisions_pd.to_csv(f"{eval_path}/Precisions.csv")

