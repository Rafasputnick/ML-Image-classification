
from matplotlib import pyplot as plt

# Plota os graficos sobre o fit
def plot_grafico(historico, tipo):
    plt.figure(0)
    plt.plot(historico['accuracy'], label='training accuracy')
    plt.plot(historico['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(f"{tipo}_epochs_accuracy.png")
    
    #plt.show()
    plt.figure(1)
    plt.plot(historico['loss'], label='training loss')
    plt.plot(historico['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    #plt.show()
    plt.savefig(f"{tipo}_epochs_loss.png")