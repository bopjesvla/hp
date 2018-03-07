from network.ActivationFunctions import ActivationFunctions as af

class GenomeReader:
    def __init__(self,model):
        self.model = model

    def makeGenome(self, genome_path, model):
        settings = model.getSettings()
        n_genes = [NGene(i,af.getAF(settings['function'])) for i in range(int(settings['bias_idx']),int(settings['input_max_idx'])+1)]
        for i in range(int(settings['output_min_idx']),int(settings['output_max_idx'])+1): n_genes.append(NGene(i,af.getAF(settings['function'])))
        c_genes = []
        with open(genome_path,'r') as f:
            for line in f:
                data = (line.rstrip()).split(':')
                innov = int(data[0])
                in_id = int(data[1])
                ou_id = int(data[2])
                weigh = float(data[3])
                enabl = (data[4] == '1')
                c_genes.append( CGene( in_id, ou_id, weigh, enabl, innov ) )

                if len(list(filter (lambda n: n.getId() == in_id, n_genes))) == 0:
                    n_genes.append( NGene(in_id, af.getAF(settings['function'])) )
                if len(list(filter (lambda n: n.getId() == ou_id, n_genes))) == 0:
                    n_genes.append( NGene(ou_id, af.getAF(settings['function'])) )
        n_genes.sort(key=lambda n: n.getId())
        return Genome( n_genes, c_genes )    

class Genome:
    def __init__(self, n_genes, c_genes):
        self.n_genes = n_genes
        self.c_genes = c_genes

    def getNgenes(self):
        return self.n_genes

    def getCgenes(self):
        return self.c_genes

    def store(self,path):
        with open(path,'w+') as f:
            for cgen in self.c_genes:
                f.write("%s:%s:%s:%.8f:%s\n" % (str(cgen.getInnov()),
                                                str(cgen.getIn()),
                                                str(cgen.getOut()),
                                                cgen.getWeight(),
                                                '1' if cgen.getEnabled() else '0'))

class NGene:
    def __init__(self, n_id, method):
        self.n_id     = n_id
        self.method   = method

    def getId(self):
        return self.n_id

    def getMethod(self):
        return self.method

class CGene:
    def __init__(self, in_id, out_id, weight, enabled, innov):
        self.in_id   = in_id
        self.out_id  = out_id
        self.weight  = weight
        self.enabled = enabled
        self.innov   = innov


    def getCopy(self):
        return CGene( self.in_id,
                      self.out_id,
                      self.weight,
                      self.enabled,
                      self.innov )

    """GETTER METHODS"""
    def getIn(self):
        return self.in_id

    def getOut(self):
        return self.out_id

    def getWeight(self):
        return self.weight

    def getEnabled(self):
        return self.enabled

    def getInnov(self):
        return self.innov

    """SETTER METHODS"""
    def setWeight(self, new_val):
        self.weight = new_val

    def setEnabled(self, new_val):
        self.enabled = new_val
