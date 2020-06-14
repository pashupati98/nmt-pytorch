from Encoders.basic_encoder import *
from AttnDecoders.attn_decoder import *
from train import *
from evaluation import *
from visualize import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    hidden_size = 256
    print("Building Encoder...")
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    print("Building Decoder...")
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    print("Starting Training...")
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    print("Finished Training...")
    # Evaluation and Visualization
    print("Evaluating Rabdomly...")
    evaluateRandomly(encoder1, attn_decoder1)

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




