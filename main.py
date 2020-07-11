from Encoders.lstm_encoder import *
from AttnDecoders.attn_lstm_decoder import *

# from Encoders.basic_encoder import *
# from AttnDecoders.attn_decoder import *

from train import *
from evaluation import *
from visualize import *
import matplotlib.pyplot as plt
import pickle
#plt.switch_backend('agg')
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    hidden_size = 256
    print("Building Encoder...")
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, 1).to(device)
    print(encoder1)
    eparam = count_parameters(encoder1)
    print("Building Decoder...")
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, layers=1).to(device)
    print(attn_decoder1)
    dparam = count_parameters(attn_decoder1)
    print("Total parameters in encoder + decoder", eparam+dparam)
    print("Starting Training...")
    trainIters(encoder1, attn_decoder1, 750, print_every=50)
    print("Finished Training...")
    # Evaluation and Visualization

    print("Evaluating Rabdomly...")
    evaluateRandomly(encoder1, attn_decoder1)

    print("Saving Encoder...")
    pickle.dump(encoder1, open('save/encV1.pkl', 'wb'))
    print("Saving Decoder...")
    pickle.dump(attn_decoder1, open('save/adecV1.pkl', 'wb'))
    print("Saved model successfully...")

    print("Evaluating on a sentence...")
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())

    evaluateAndShowAttention(encoder1, attn_decoder1, "elle a cinq ans de moins que moi .")

    evaluateAndShowAttention(encoder1, attn_decoder1, "elle est trop petit .")

    evaluateAndShowAttention(encoder1, attn_decoder1, "je ne crains pas de mourir .")

    evaluateAndShowAttention(encoder1, attn_decoder1, "c est un jeune directeur plein de talent .")

    return


if __name__ == "__main__":
    main()




