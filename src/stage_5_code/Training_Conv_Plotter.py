"""
Plot loss v.s. epoch
"""
import matplotlib.pyplot as plt

class Plotter:
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    plot_name = "Neural Network Plot"
    file_name = "NN_plot.png"
    
    def __init__(self, plot_name, file_name):
        self.plot_name = plot_name
        self.file_name = file_name
        
    def plot(self, validation=True):
        fig, ax = plt.subplots(1,2,figsize = (16,4))
        ax[0].plot(self.train_loss, color='#EFAEA4', label = 'Training Loss')
        ax[1].plot(self.train_acc, color='#EFAEA4',label = 'Training Accuracy')
        if validation:
            ax[0].plot(self.val_loss, color='#B2D7D0', label = 'Validation Loss')
            ax[1].plot(self.val_acc, color='#B2D7D0', label = 'Validation Accuracy')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs');
        ax[0].set_ylabel('Loss')
        ax[1].set_ylabel('Accuracy %');
        fig.suptitle(self.plot_name, fontsize = 24)
        
        plt.savefig(self.file_name)
        