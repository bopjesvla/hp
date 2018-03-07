import numpy as np

class Node:
    def __init__(self, n_id, method, n_type):
        self.n_id       = n_id
        self.method     = method
        self.parents    = []
        self.weights    = []
        self.activity   = 0
        self.n_type     = n_type

    def update(self):
        if not (self.n_type == 'input' or self.n_type == 'bias'):
            activation = sum( self.parents[i].getActivity()*self.weights[i] for i in range(len(self.parents)) )
            self.activity = self.method( activation )

    def addParent( self, p_node, p_weight ):
        self.parents.append(p_node)
        self.weights.append(p_weight)

    def getId(self):
        return self.n_id

    def getAF(self):
        return self.method

    def setActivity(self, value):
        self.activity = value

    def getActivity(self):
        return self.activity

    def getType(self):
        return self.n_type

    def setType(self, n_type):
        self.n_type = n_type

    def toString(self):
        return "Node " + str(self.n_id)
