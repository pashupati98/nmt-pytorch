import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import pickle


def showPlot(points):
    print("function called")
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.show()
    #fig.savefig('loss.png')


gru1 = pickle.load(open("./save/gru1/gru1.pkl", 'rb'))
showPlot(gru1)