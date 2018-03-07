"""def initSettings():
        settings = {}
        with open('../res/settings.txt','r') as f:
            for line in f:
                if len(line)>1 and not line[0]=='#':
                    line_data = (line.rstrip()).split("=")
                    line_data = [(d.rstrip()).lstrip() for d in line_data]
                    settings[line_data[0]] = line_data[1] if not line_data[1].isdigit() else float(line_data[1])
        return settings

settings = initSettings()

with open('000000.dna','w+') as f:
    innov_nr = 0
    for i_node in range(int(settings['bias_idx']),int(settings['input_max_idx'])+1):
        for o_node in range(int(settings['output_min_idx']),int(settings['output_max_idx'])+1):
            f.write("%s:%s:%s:1:1\n" % (innov_nr, i_node, o_node))
            innov_nr += 1
"""

import os

string = 'SIGMOID'
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk('./') for f in filenames]
for p in result:
        try:
                if string in open(p).read():
                        with open(p,'r') as f:
                                i = 1
                                for line in f:
                                        if string in line:
                                                print( str(i) + " > " + p )
                                        i += 1
                                
        except:
                continue
