import os
import pandas as pd
from tqdm import tqdm


def get_labelmap():
    dataframe = pd.read_csv(os.path.join(os.getcwd(), 'librispeech_clean_train_100h_phoneme.csv'))
    textlist = sorted(list(set(dataframe['text'])))
    labelmap = {}
    for i, phoneme in tqdm(enumerate(textlist)):
        labelmap[phoneme] = i

    return labelmap

def get_indexmap(labelmap):
    indexmap = {}
    for key in labelmap:
        indexmap[labelmap[key]] = key

    return indexmap


def print_map(maps):
    for key in maps:
        if(type(key) == str):
            print("%d | %s"%(maps[key], key))
        
        else:
            print("%s | %d"%(maps[key], key))

def label2index(label, labelmap):
    return labelmap[label]

def index2label(index, indexmap):
    return indexmap[index]





labelmap = get_labelmap()
print_map(labelmap)

indexmap = get_indexmap(labelmap)
print_map(indexmap)
