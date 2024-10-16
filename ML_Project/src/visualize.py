# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot_data(self):
        sns.pairplot(self.data, hue='target')
        plt.show()

    def plot_correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.show()
