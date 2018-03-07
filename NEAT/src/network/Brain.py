import numpy as np

from network.ActivationFunctions import ActivationFunctions as af
from genetics.Genome import *
from network.Node import *

class Brain:
    def __init__(self, genome, model):
        self.nodes = []
        self.build( genome )
        settings = model.getSettings()
        self.input_nodes = [ self.nodes[i] for i in range(int(settings['input_min_idx']),int(settings['input_max_idx'])+1) ]
        self.output_nodes = [ self.getNode(i) for i in range(int(settings['output_min_idx']),int(settings['output_max_idx'])+1) ]
        self.nodes[0].setType('bias')
        for n in self.input_nodes: n.setType('input')
        for n in self.output_nodes: n.setType('output')

        #print(len([ n.getId() for n in self.input_nodes]))
    
    def build(self, genome):
        #Add all nodes from the genome to the network
        for n_gene in genome.getNgenes():
            self.nodes.append( Node( n_gene.getId(),
                                     n_gene.getMethod(),
                                     'hidden'))
        #Sort execution order based on topologies.
        self.nodes.sort(key=lambda n: n.getId())
        #Add all axons
        for gene in genome.getCgenes():
            if gene.getEnabled():
                self.getNode( gene.getOut() ).addParent( self.getNode(gene.getIn()), gene.getWeight())

    def query(self, inputs):
        self.nodes[0].setActivity(1)
        #feed inputs
        for i in range(len(inputs)):
            self.input_nodes[i].setActivity( inputs[i] )
        #feed through network
        for node in self.nodes:
            node.update()
        #measure outputs
        out = [ n.getActivity() for n in self.output_nodes ]
        #return output
        return out
        
            
    def getNode( self, n_id ):
        try:
            return [n for n in self.nodes if n.getId()==n_id][0]
        except:
            return None

    def getNodes(self):
        return self.nodes
