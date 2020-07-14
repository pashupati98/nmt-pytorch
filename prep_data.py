import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from load_data import *
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print("Data Loaded....See Example below - ")
print(random.choice(pairs))
print("Saving Dictionary")
pickle.dump(input_lang, open('save/input.pkl', 'wb'))
pickle.dump(output_lang, open('save/output.pkl', 'wb'))


def indexesFromSentence(lang, sentence):
    words = sentence.split(' ')
    indexes = []
    for word in words:
        if word not in lang.word2index.keys():
            indexes.append(lang.word2index["UNK"])
        else:
            indexes.append(lang.word2index[word])

    return indexes
    #return [lang.word2index[word] for word in words]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

