import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
UNK_token = 2

encoder1 = pickle.load(open("save/encV1.pkl", "rb"))
attn_decoder1 = pickle.load(open("save/adecV1.pkl", "rb"))
input_lang = pickle.load(open("save/input.pkl", "rb"))
output_lang = pickle.load(open("save/output.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSentence(lang, sentence):
    words = sentence.split(' ')
    indexes = []
    for word in words:
        if word not in lang.word2index.keys():
            indexes.append(lang.word2index["UNK"])
        else:
            indexes.append(lang.word2index[word])

    return indexes
    # return [lang.word2index[word] for word in words]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def predict(encoder, decoder, sent):
    output_words, attentions = evaluate(encoder, decoder, sent)
    output_sentence = ' '.join(output_words)
    return output_sentence


def run():
    print("English to Hindi Translator")
    while True:
        sent = input("Please enter your input sentence, Enter exit to close this program \n")
        sent = str(sent)
        if sent == "exit":
            break
        print("English Sentence : ", sent)
        ans = predict(encoder1, attn_decoder1, sent)
        print("Hindi Sentence : ", ans)


if __name__ == "__main__":
    run()