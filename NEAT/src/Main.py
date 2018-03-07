from mvc.Model import Model
import pygame
import sys
sys.path.append("..")

from tools.DEBUG import *

class Main:
    def __init__(self):
        pygame.init()
        self.model = Model(300)

    def getModel(self):
        return self.model

if __name__ == '__main__':
    m = Main()
