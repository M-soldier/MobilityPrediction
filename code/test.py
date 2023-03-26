import torch

from dataPreprocess.Dataset import Dataset_Trajectory

# from model.RNN import RNN
# from dataPreprocess.DataFoursquare import DataFoursquare

from model.Markov import Markov_1, Markov_2


def runMarkov():
    dataset = Dataset_Trajectory("../data/processedData/", "Markov").dataset[0]
    Markov_1(dataset)
    Markov_2(dataset)

runMarkov()
