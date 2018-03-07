import time
import random
import pandas as pd
import numpy as np
from sklearn import svm, linear_model, ensemble, pipeline, decomposition, calibration, metrics, isotonic, preprocessing, naive_bayes, grid_search, model_selection
from scipy import interpolate, stats

from tools.DEBUG         import *
from tools.MyThread      import *
from genetics.Genome     import *
from genetics.Evolution  import *
from network.Brain       import *

import os

class Model:
    print_fps   = False
    print_score = False
    gen_idx = 0

    
    def __init__(self,maxframes):
        self.settings   = self.initSettings()
        self.X, self.Y  = self.loadData()
        self.crashed    = False
        self.reader     = GenomeReader(self)
        self.evolution  = Evolution(self)
        self.evolution.loadGenomes('../res/genomes',10)
        self.score = 0
        self.best  = 999999


        self.startThreads()
        
        
    """INITIALIZATION"""
        
    def initSettings(self):
        settings = {}
        with open('../res/settings.txt','r') as f:
            for line in f:
                if len(line)>1 and not line[0]=='#':
                    line_data = (line.rstrip()).split("=")
                    line_data = [(d.rstrip()).lstrip() for d in line_data]
                    settings[line_data[0]] = line_data[1] if not line_data[1].isdigit() else float(line_data[1])
        return settings

    def loadData(self):
        df = pd.read_csv('../../train.csv')
        X = df[[c for c in df.columns if c != 'SalePrice']]
        X_float = X.select_dtypes(exclude=['object']).fillna(0)
        y = df['SalePrice']
        X.fillna('None', inplace=True)
        one_hot = pd.get_dummies(df)
        return (one_hot, y)

    def loadGenome(self):
        reader = GenomeReader(self)
        genome = reader.makeGenome('../res/genomes/best.dna',self)
        return genome

    def loadNetwork(self,genome):
        return Brain(genome,self)

    def startThreads(self):
        m_thread = MyThread( 3, "ModelThread", self.loop )
        m_thread.start()
        m_thread.join()

    def reset(self):
        (self.gen_idx,genome) = self.evolution.getNextGenome()
        self.genome           = genome
        self.network          = self.loadNetwork(self.genome)
        self.score            = 0
    
    """GAME LOOP"""

    def loop(self):
        x = [ row.tolist() for _,row in self.X.iterrows() ]
        y = [ price for price in self.Y ]
        errs = [0]*len(x)
        epoch = 0
        while not self.crashed:
            epoch += 1
            if epoch % 10 == 0:
                self.settings = self.initSettings()
                self.sim_frames = int(self.settings['sim_frames'])
            self.reset()
            errs = [0]*len(x)
            

            for i,e in enumerate(x):
                pred = max(0,self.network.query( e )[0])
                true = y[i]
                errs[i] = (np.log(pred+1) - np.log(true+1))**2

            rmsle = np.sqrt(np.mean(errs))
            self.evolution.reportFitness(self.gen_idx, 10. / rmsle )
            print(str(epoch) + " -> " + str(rmsle),end='\r')
            if rmsle < self.best:
                self.best = rmsle
                print("BEST: %s                                           " % (str(rmsle)),end='\r')

            

    """GETTER METHODS"""

    def getSettings(self):
        return self.settings

    def getCrashed(self):
        return self.crashed


    def getGenome(self,name='player'):
        return self.entities[name].getGenome()

    def getNetwork(self,name='player'):
        return self.entities[name].getNetwork()

    def getScore(self):
        return self.score

    """SETTER METHODS"""
    def setCrashed(self, boolean):
        self.crashed = boolean

