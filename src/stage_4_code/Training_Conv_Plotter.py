"""
Plot loss v.s. epoch
"""
import matplotlib.pyplot as plt

class Plotter:
    xs = []
    ys = []
        
    def plot(self, filename):
        plt.plot(self.xs, self.ys, linewidth=2.0)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(filename)
        