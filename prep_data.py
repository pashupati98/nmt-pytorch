import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from load_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print("Data Loaded....See Example below - ")
print(random.choice(pairs))



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

