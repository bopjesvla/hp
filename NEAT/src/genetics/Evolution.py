import math
import copy
import random
import numpy as np

from .Genome import *
from tools.DEBUG import *

from os import listdir
from os.path import isfile, join

class Evolution():
    genomes = []
    fitness = []
    idx     = 0
    innovations = []
    generation = 0

    best_fitness = 0
    
    representatives = []
    species_size    = {}

    c1 = 1.
    c2 = 1.
    c3 = 0.4
    delta_t = 3.

    Pr_mutate_weights = 0.8
    Pr_mutate_uniform = 0.99
    Pr_add_new_node   = 0.03
    Pr_add_new_link   = 0.1

    only_mutate = 0.25
    Pr_interspecies_mating = 0.0001
    
    
    def __init__(self,model):
        self.model  = model
        self.settings = model.getSettings()
        self.reader = GenomeReader(self.model)

        self.idx = 0

    def loadGenomes(self, genome_directory, N=0):
        self.genomes = []
        files = [f for f in listdir(genome_directory) if isfile(join(genome_directory,f))]
        if N==0:
            for f in files:
                self.genomes.append( self.reader.makeGenome(join(genome_directory,f),self.model) )
        else:
            for n in range(N):
                self.genomes.append( self.reader.makeGenome(join(genome_directory,files[0]),self.model))
        self.fitness = [-math.inf]*len(self.genomes)

    def initInnovations(self):
        for genome in self.genomes:
            for g in genome.getCgenes():
                if not (g.getInnov(),g.getIn(),g.getOut()) in self.innovations:
                    self.innovations.append((g.getInnov(),g.getIn(),g.getOut()))

    def getNextGenome(self):
        if self.idx >= len(self.genomes):
            self.nextGen()
        self.idx += 1
        return (self.idx-1,self.genomes[self.idx-1])
                
    def reportFitness(self,idx,fitness):
        self.fitness[idx] = fitness
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.storeGenome('../res/genomes/best.dna',idx)

    def storeGenome(self,path,idx):
        gen = self.genomes[idx]
        gen.store(path)
        

    def nextGen(self):
        self.generation += 1
        print("Generation: " + str(self.generation) + " - Best: " + str(10./self.best_fitness))
        
        species = self.makeSpecies()
        shared_fitness = self.sharedFitness(species)

        self.genomes = [g for _,g in sorted(zip(shared_fitness,self.genomes),key=lambda pair: pair[0],reverse=True)]
        species      = [s for _,s in sorted(zip(shared_fitness,species),key=lambda pair: pair[0],reverse=True)]
        shared_fitness.sort(reverse=True)

        new_genomes = []
        N = len(self.genomes)
        # a percentage of the generation is only mutated
        for n in range(int(N*self.only_mutate)):
            #sample a random individual based on fitness
            new_genome = copy.deepcopy(self.genomes[ self.sample(shared_fitness) ])
            #randomly apply mutations to new genome
            if random.random() < self.Pr_mutate_weights: new_genome = self.mutateWeights(new_genome)
            if random.random() < self.Pr_add_new_node:   new_genome = self.mutateAddNode(new_genome)
            if random.random() < self.Pr_add_new_link:   new_genome = self.mutateAddLink(new_genome)
            #append a copy of the mutated genome to the genomes for next generation
            new_genomes.append(copy.deepcopy(new_genome))

        # the rest is created through crossover
        while len(new_genomes) < len(self.genomes):
            #sample a random individual based on fitness
            index = self.sample( shared_fitness )
            parent1 = self.genomes[index]
            specie1 = species[index]
            fitnes1 = shared_fitness[index]
            #there's a small chance that individuals between species can mate
            if random.random() < self.Pr_interspecies_mating:
                mating_pool = [g for g in self.genomes if g != parent1]
            else:
                mating_pool = [g for i,g in enumerate(self.genomes) if g != parent1 and species[i] == specie1]

            #if no individual wants to reproduce with parent1 it can go fuck itself
            if len(mating_pool) == 0:
                new_genome = copy.deepcopy(parent1)
            else:
                pool_fitness = [ shared_fitness[ self.genomes.index(g) ] for g in mating_pool ]
                index = self.sample( pool_fitness )
                parent2 = mating_pool[index]
                fitnes2 = pool_fitness[index]
                new_genome = self.crossover(parent1,parent2,fitnes1,fitnes2)

            #randomly apply mutations to new genome
            if random.random() < self.Pr_mutate_weights: new_genome = self.mutateWeights(new_genome)
            if random.random() < self.Pr_add_new_node:   new_genome = self.mutateAddNode(new_genome)
            if random.random() < self.Pr_add_new_link:   new_genome = self.mutateAddLink(new_genome)
            #append a copy of the mutated genome to the genomes for next generation
            new_genomes.append(copy.deepcopy(new_genome))
            
        self.genomes = copy.deepcopy(new_genomes)
        
        self.idx = 0

    def sample(self, fitness_list):
        threshold = random.random() * sum( fitness_list )
        i = 0
        running_total = fitness_list[i]
        while threshold > running_total:
            i += 1
            running_total += fitness_list[i]
        return i

    def makeSpecies(self):
        species = [0]*len(self.genomes)
        for i,genome in enumerate(self.genomes):
            new_species = True
            for j,representative in enumerate(self.representatives):
                if self.SH(genome,representative):
                    new_species = False
                    species[i] = j
            if new_species:
                self.representatives.append( genome )
                species[i] = len(self.representatives)-1

        #update representatives of this generation for the next generation
        self.representatives = []
        self.species_size = {}
        for specie in set(species):
            self.representatives.append( self.genomes[ species.index(specie) ] )
            try:
                self.species_size[specie] += 1
            except KeyError:
                self.species_size[specie] = 1
        
        return species

    def delta(self, genome1, genome2):
        n1 = genome1.getNgenes()
        c1 = genome1.getCgenes()
        n2 = genome2.getNgenes()
        c2 = genome2.getCgenes()
        

        N = max( len(c1), len(c2) ) + 1
        max_innov1 = (0 if len(c1)==0 else max( g.getInnov() for g in c1 ))
        max_innov2 = (0 if len(c2)==0 else max( g.getInnov() for g in c2 ))
        # E := the number of excess genes
        E = sum( 1. for g in c1 if g.getInnov() > max_innov2 ) + sum( 1. for g in c2 if g.getInnov() > max_innov1 )
        # D := the number of disjoint genes
        D = sum( 1. for g1 in c1 if g1.getInnov() < max_innov2 and g1.getInnov() not in [ g2.getInnov() for g2 in c2 ] ) + \
            sum( 1. for g2 in c2 if g2.getInnov() < max_innov1 and g2.getInnov() not in [ g1.getInnov() for g1 in c1 ] )
        # W := the average weight difference of matching genes
        W = np.mean( [ np.sqrt(([g1 for g1 in c1 if g1.getInnov()==innov][0].getWeight()-[g2 for g2 in c2 if g2.getInnov()==innov][0].getWeight())**2) \
                       for innov in set( [ g1.getInnov() for g1 in c1 ] ) & set( [ g2.getInnov() for g2 in c2 ] ) ] )
        
        return (self.c1*E)/N + (self.c2*D)/N + self.c3*W

    def SH(self, genome1, genome2):
        return self.delta(genome1,genome2) < self.delta_t

    def sharedFitness(self,species):
        shared_fitness = []
        for i,g_i in enumerate(self.genomes):
            shared_fitness.append( self.fitness[i] / self.species_size[ species[i] ] )
            #shared_fitness.append( self.fitness[i] / (sum( 1.*self.SH(g_i,g_j) for g_j in self.genomes )+1) )
        return shared_fitness

    """CROSSOVER AND MUTATION METHODS"""

    def crossover(self,parent1,parent2,fitness1,fitness2):
        dprt(self,"crossover")
        c_genes = []
        n_genes = [NGene(i,af.getAF( self.settings['function'] )) for i in range(0,int(self.settings['input_max_idx'])+1)]
        for i in range(int(self.settings['output_min_idx']),int(self.settings['output_max_idx'])+1):
            n_genes.append(NGene(i,af.getAF( self.settings['function'] )))

        C1 = copy.deepcopy( parent1.getCgenes() if fitness1 > fitness2 else parent2.getCgenes() )
        C2 = copy.deepcopy( parent2.getCgenes() if fitness1 > fitness2 else parent1.getCgenes() )
        C1.sort(key=lambda gene: gene.getInnov())
        C2.sort(key=lambda gene: gene.getInnov())

        for gene1 in C1:
            if gene1.getInnov() in [g.getInnov() for g in C2]:
                if fitness1 == fitness2 or random.random() > fitness2/(fitness1+fitness2):
                    c_genes.append( gene1 )
                else:
                    c_genes.append( [g2 for g2 in C2 if g2.getInnov() == gene1.getInnov() ][0] )
            else:
                c_genes.append( gene1 )

        for g in c_genes:
            if len(list(filter (lambda n: n.getId() == g.getIn(), n_genes))) == 0:
                n_genes.append( NGene(g.getIn(), af.getAF( self.settings['function'] )))
            if len(list(filter (lambda n: n.getId() == g.getOut(), n_genes))) == 0:
                n_genes.append( NGene(g.getOut(), af.getAF( self.settings['function'] )))

        n_genes.sort(key=lambda n: n.getId())
        return Genome( n_genes, c_genes )

    def mutateAddLink(self,genome):
        dprt(self,"mutate add link")
        #select the nodes that are valid as IN and OUT for the new link
        in_lst = [ n.getId() for n in genome.getNgenes() if n.getId() >= self.settings['bias_idx'] and n.getId() <= self.settings['hidden_max_idx']]
        out_lst   = [ n.getId() for n in genome.getNgenes() if n.getId() >= self.settings['hidden_min_idx'] and n.getId() <= self.settings['output_max_idx']]

        #select IN and OUT nodes for the new link from the valid lists
        in_id  = in_lst[ random.randint(0,len(in_lst)-1) ]
        out_id = out_lst[ random.randint(0,len(out_lst)-1) ]

        #make sure that the suggested link does not already exist
        i = 0
        while len(list(filter (lambda cgene: (cgene.getIn() == in_id and cgene.getOut() == out_id), genome.getCgenes()))) > 0:
            i += 1
            if i > 100: # if nothing works, don't mutate
                return copy.deepcopy(genome)
            else:
                in_id  = in_lst[ random.randint(0,len(in_lst)-1) ]
                out_id = out_lst[ random.randint(0,len(out_lst)-1) ]
        
        #find the corresponding innovation, if it does not yet exist, add new innovation
        innov_nr = self.getInnovNumber( in_id, out_id )

        #randomly generate weight for link
        weight = np.random.normal(0,1)

        n_genes = copy.deepcopy( genome.getNgenes() )
        c_genes = copy.deepcopy( genome.getCgenes() )
        c_genes.append( CGene( in_id, out_id, weight, True, innov_nr ))

        return Genome(n_genes,c_genes)

    def mutateAddNode(self,genome):
        dprt(self,"mutate add node")
        if len(genome.getCgenes()) == 0:
            return copy.deepcopy(genome)
        else:
            n_genes = copy.deepcopy(genome.getNgenes())
            c_genes = copy.deepcopy(genome.getCgenes())
            link = c_genes[random.randint(0,len(c_genes)-1)]

            old_in     = link.getIn()
            old_out    = link.getOut()
            old_weight = link.getWeight()

            #if there's no room to insert a new node in this link, don't mutate
            if old_out - 1 < old_in + 1: return copy.deepcopy(genome)

            #select a node between in and out to insert in the link
            new_idx = random.randint( old_in + 1, old_out - 1)
            i = 0
            while len(list(filter (lambda ngene: ngene.getId() == new_idx, n_genes))) > 0:
                i += 1
                if i > 100: return copy.deepcopy(genome) # if we can't find a valid node, don't mutate
                else: new_idx = random.randint( old_in + 1, old_out - 1 )

            #if we have found a valid node to add, disable the old link
            link.setEnabled( False )

            #add the new node to n_genes
            n_genes.append( NGene( new_idx, af.getAF( self.settings['function'] ) ) )
            n_genes.sort(key=lambda n: n.getId())

            #add the link between old in and new node
            innov_nr = self.getInnovNumber( old_in, new_idx )
            weight   = 1
            c_genes.append( CGene( old_in, new_idx, weight, True, innov_nr ))

            #add the link between new node and old out
            innov_nr = self.getInnovNumber( new_idx, old_out )
            weight = old_weight
            c_genes.append( CGene( new_idx, old_out, weight, True, innov_nr ))

            c_genes.sort(key=lambda n: n.getInnov())

            return Genome(n_genes,c_genes)

    def mutateWeights(self,genome):
        dprt(self,"mutate weights")
        if len(genome.getCgenes()) == 0:
            return copy.deepcopy(genome)
        else:
            n_genes = copy.deepcopy(genome.getNgenes())
            c_genes = copy.deepcopy(genome.getCgenes())

            for link in c_genes:
                if random.random() < self.Pr_mutate_weights:
                    if random.random() < self.Pr_mutate_uniform:
                        link.setWeight( link.getWeight() + np.random.normal(0,0.1) )
                    else:
                        link.setWeight( np.random.normal(0,1) )

            return Genome( n_genes, c_genes )

    """Find the innovation corresponding to specified IN and OUT nodes,
       If it does not yet exist, add new innovation
    """
    def getInnovNumber(self, in_id, out_id):
        for innov in self.innovations:
            if innov[1] == in_id and innov[2] == out_id:
                return innov[0]
        self.innovations.append((len(self.innovations),in_id,out_id))
        return len(self.innovations)-1
                    
            
