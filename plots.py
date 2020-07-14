import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import pickle


def showPlot(gru11):
    print("function called")
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(gru1, label="GRU Single Layer")
    #plt.plot(lstm1, label="LSTM Single Layer")
    plt.xlabel("iter/1000")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    fig.savefig('save/images/g1.png')


gru1 = pickle.load(open("./save/gru1n.pkl", 'rb'))
lstm1 = pickle.load(open("./save/lstm1n.pkl", 'rb'))
showPlot(gru1)

