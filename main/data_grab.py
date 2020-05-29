import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import pandas as pd
import sqlite3
import ast
import re

from main.Discord_Scraper_master.discord import Discord

def get_data(name):
    cnx = sqlite3.connect(f'{name}/text.db')

    res = cnx.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print(name[0])

    df = pd.read_sql_query("SELECT * FROM text_337694725056364544_337694725056364544", cnx)
    return df


def get_org(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H:%M:%S')
    data = data.sort_values(by='timestamp', ascending=True)
    data['delta'] = data['timestamp'].diff().dt.seconds.div(60, fill_value=0)
    data['cat'] = 1
    data['uid'] = data.index.astype(str) + "L"
    return data


def change_cat(df):
    df['cat'] = df['delta'].gt(20).cumsum()
    return df


def get_conversation(data):
    df = pd.DataFrame(columns=['name1', 'name2', 'conversation'])
    for i in list(set(data.cat.values)):
        cut = data[data.cat == i]
        if (len(cut) > 1) & (len(set(cut.name)) > 1):
            name1 = cut.name.iloc[0]
            name2 = cut.name.iloc[1]
            if (name1 == name2) & (len(data) > 2):
                name2 = cut.name.iloc[2]
            convo = '->'.join(cut.uid.values.tolist())
        df.loc[i] = [name1, name2, convo]
    return df


def get_extra_data():
    path_to_zip = tf.keras.utils.get_file(
        'cornell_movie_dialogs.zip',
        origin=
        'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
        extract=True)

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
    path_to_movie_conversations = os.path.join(path_to_dataset,
                                               'movie_conversations.txt')
    with open(path_to_movie_lines, errors='ignore') as file:
        lines_one = file.readlines()

    df_one = pd.DataFrame([sub.split("+++$+++") for sub in lines_one])
    df_one.columns = ['uid', 'nn', 'bb', 'name', 'content']
    df_one['content'] = df_one['content'].str.replace('\n', '')
    df_one = df_one[['uid', 'name', 'content']]

    with open(path_to_movie_conversations, 'r') as file:
        lines_two = file.readlines()

    df_two = pd.DataFrame([sub.split("+++$+++") for sub in lines_two])
    df_two.columns = ['name1', 'name2', 'm', 'conversation']
    df_two['conversation'] = df_two['conversation'].str.replace('\n', '')

    def remove_list(x):
        x = re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', x, flags=re.M)
        x = ast.literal_eval(x)
        x = '->'.join(x)
        return x

    df_two['conversation'] = df_two['conversation'].apply(remove_list)
    df_two = df_two[['name1', 'name2', 'conversation']]
    return df_one, df_two


def all_data(name='Bot Scrapes', is_data=False):
    if is_data==False:
        print(False)
        discords = Discord()
        discords.grab_server_data()
    else:
        print(True)
    data = get_data(name)
    data = data.drop_duplicates()
    data = get_org(data)
    data = change_cat(data)
    convo = get_conversation(data)
    speach_lines = data.groupby('cat').filter(lambda x: len(x) > 1)[['name', 'content', 'uid']]
    df_one, df_two = get_extra_data()
    convo = pd.concat([convo, df_two.sample(len(convo))])
    speach_lines = pd.concat([speach_lines, df_one])
    return convo, speach_lines