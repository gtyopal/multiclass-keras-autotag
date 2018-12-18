# coding: utf-8
# encoding = utf8
import sys
import os
import random
import re
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
import string
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas
import itertools
from collections import Counter
from sklearn.utils import shuffle
import json

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_from_disk(input_dataset_path, output_dataset_path):
    df = pd.read_csv(input_dataset_path, sep=",", encoding='latin-1', low_memory=False, error_bad_lines = False)
    df = df[['community', 'title', 'body']]
    df['body'] = df['body'].fillna(value='')
    df['title'] = df['title'].fillna(value='')
    df['message'] = df.title.str.cat(df.body)
    df['label'] = df['community'].fillna(value='')
    df.drop(['body', 'title'], axis=1, inplace=True)
    df = shuffle(df)
    df['label'].dropna()
    df.dropna()
    df = df[df['message'].apply(lambda x: len(x) < 1000)]
    df = df.groupby("label").filter(lambda x: len(x) > 3000)
    nb_classes = len(list(set(df['label'].tolist())))
    df['message'] = df['message'].str.slice(0, 1000)
    df['message'] = df['message'].apply((lambda x: clean_str(x)))
    print('number of sentences:', df.shape[0])
    print("number of tags is", nb_classes)
    df.to_csv(output_dataset_path, sep="\t", index=False, columns=["label", "message"], encoding="utf-8")
    return df, nb_classes


 # Word dictionary prepare and data split
def generate_words(df_dataset):
    """ generate words dict """
    words_dict = defaultdict(int)
    for item in df_dataset["message"]:
        if isinstance(item, float) or isinstance(item, int):
            continue
        for word in item.split():
            word = word.lower()
            if re.findall(r'^[a-zA-Z0-9\.\?\!\`\"\;\:\.\,\@\#\$\(\)\-\_\+\=\^\%\&\*]+$', word):
                words_dict[word] += 1
    count_sort = sorted(words_dict.items(), key=lambda e:-e[1])
    word2id = {'<pad>': 0}
    idx = 1
    for w in count_sort:
        if w[1] > 1:
            word2id[w[0]] = idx
            idx += 1
    return word2id


def save_words(word2id, words_path):
    """ save words and id """
    with open(words_path, "w") as fw:
        for w in word2id:
            fw.write(w + "\t" + str(word2id[w]) + "\n")
    return 0


def load_words(words_path):
    """ load words and id """
    word2id = {}
    with open(words_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            word2id[line[0]] = int(line[1])
    return word2id


# add 0831
def generate_category2id(df_dataset):
    category2id = {}
    ilabel = 0
    for item in df_dataset["label"]:
        item = item.strip().lower()
        if item and item != "nan" and item not in category2id:
            category2id[item] = ilabel
            ilabel += 1
    return category2id

# add 0831
def save_category2id(category2id, category2id_path):
    with open(category2id_path, "w") as fw:
        s = sorted(category2id.items(), key=lambda e:e[1])
        for i in s:
            fw.write(i[0] + "\t" + str(i[1]) + "\n")


# add 0831
def load_category2id(category2id_path):
    category2id = {}
    with open(category2id_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")
            category2id[line[0]] = int(line[1])
    return category2id


# add 0831
def split_train_test(df_dataset, word2id, split_rate, category2id):
    """ generate train and test dataset, then map words to index """
    item2id = []  # [(label, message2id)]
    max_len_sent = 0
    icount = 0
    for i in df_dataset.index:
        label, message = df_dataset.loc[i].values[0], df_dataset.loc[i].values[1]
        if isinstance(label, float) or isinstance(message, int) or isinstance(label, int) or isinstance(message, float):
            continue
        label = label.lower()
        message = message.lower()
        if label == "nan" or message == "nan" or label not in category2id:
            continue
        label = category2id[label]
        message2id = [word2id.get(w, 0) for w in message.split()]
        item2id.append((label, message2id))
        icount += 1
        #max_len_sent += len(message2id)
        if len(message2id) > max_len_sent:
            max_len_sent = len(message2id)
    # max_len_sent /= icount
    random.shuffle(item2id)
    x_train, y_train, x_test, y_test = [], [], [], []
    for item in item2id[:int(len(item2id)*split_rate)]:
        x_test.append(item[1])
        y_test.append(item[0])
    for item in item2id[int(len(item2id)*split_rate):]:
        x_train.append(item[1])
        y_train.append(item[0])
    return x_train, y_train, x_test, y_test, max_len_sent

# # FastText Data Processing by adding n-grams:
def create_ngram_set(input_list, ngram_value):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def fasttext_data(x_train, x_test, max_features, ngram_range):
    if ngram_range > 1:
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        #print max_features
        #print ngram_set
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        max_features_ngram = 0
        #try:
        #print(indice_token)
        max_features_ngram = np.max(list(indice_token.keys()))
        #except ValueError:
        #    print("ERROR:max features ngram")
        #    pass
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        return x_train, x_test, max_features_ngram, token_indice
    return x_train, x_test, 0, {}


def save_ngram_words(token_indice, words_ngram_path):
    with open(words_ngram_path, "w") as fw:
        for item in token_indice:
            fw.write(str(item[0])+","+str(item[1])+","+str(token_indice[item])+"\n")
    return 0


def load_ngram_words(words_ngram_path):
    token_indice = {}
    with open(words_ngram_path, "r") as fr:
        for item in fr.readlines():
            item = item.strip().split(",")
            token_indice[(int(item[0]), int(item[1]))] = int(item[2])
    return token_indice


def predict_fasttext_data(input_list, token_indice, ngram_range):
    ngram_set = set()
    for i in range(2, ngram_range + 1):
        set_of_ngram = create_ngram_set(input_list, ngram_value=i)
        ngram_set.update(set_of_ngram)
    new_seq = input_list
    for ngram in ngram_set:
        if ngram in token_indice:
            new_seq.append(token_indice[ngram])
    return new_seq


if __name__ == '__main__' :
    df, nb_classes = load_data_from_disk("../data/en_community_content.csv", "../data/en_community_content_clean.csv")
    print("nb_classes:", nb_classes)
