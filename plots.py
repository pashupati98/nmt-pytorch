import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import pickle


def showPlot(gru11, lstm1, gru3, lstm3):
    print("function called")
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    # plt.plot(gru1, label="GRU-1L")
    # plt.plot(lstm1, label="LSTM-1L")
    # plt.plot(gru3, label="GRU-3L")
    plt.plot(lstm1, label="LSTM Single Layers")
    plt.xlabel("iter/100")
    plt.ylabel("loss")
    plt.legend()
    plt.ylim(0, 5)
    # plt.text(200, 4.7, "Encoder-Decoder architecture")
    # plt.text(350, 4.4, "with attention")
    plt.show()
    fig.savefig('save/images/l1.png')


gru1 = pickle.load(open("./save/gru1n.pkl", 'rb'))
lstm1 = pickle.load(open("./save/lstm1n.pkl", 'rb'))
gru3 = pickle.load(open("./save/gru3n.pkl", 'rb'))
lstm3 = pickle.load(open("./save/lstm3n.pkl", 'rb'))
showPlot(gru1, lstm1, gru3, lstm3)

