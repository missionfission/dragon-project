import os
import graphviz as gv
def kotana_reader():
    file = open("../../vis-oos-16simd-32op-192-192reg.c0.txt",'r')
    lines = []
    for line in file.readlines():
        instruction = line.split("\t")[-1]
        if len(instruction.split("="))>1:
            lines.append(instruction.split("=")[-1].split("(")[0].strip())
    return lines
# {"I", "L", "S", "W","C", "E","R", "D"}
stage =  {"F" "Rn" "Wat", "Sr", "Sw", "Wb", "Cm"}
riscv_instruction_set = ['fmv.d.x','fmadd.d','vle64','vfmul.vv','vfmacc.vv','VFALU','VFXLD','vlxei64' 'addi','fld','addi','bne','add','slli','ld', "iLD", "iALU", "fLD", "iSFT", "fMUL"]
inst_energy = ['0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1']
stage_energy = []
total_cycles = 5066
pattern1 = ['ld', 'fld', 'addi', 'slli', 'add', 'fld', 'fmadd.d','add']
pattern2 = ['fLD', 'iALU', 'iSFT', 'iALU', 'fLD', 'fMUL', 'iALU', 'iLD']
pattern3 = ['vfmacc.vv','VFALU','VFXLD','vlxei64']
pattern4 = ['addi', 'addi', 'iALU', 'iALU']

def kotana_graph(lines):
    energy = 0
    for inst in lines:
        if inst in riscv_instruction_set:
            index = riscv_instruction_set.index(inst)
            energy += float(inst_energy[index])
        # if inst in stage:
        #     index = riscv_instruction_set.index(inst)
        #     energy += stage_energy[index]
    # dummy energy numbers
    print("Total energy Consumption:", energy)
    print("Total Power consumption:", energy/total_cycles)

class Processor_Graph(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.name = 'Processor_Graph'
        
    def _build_visual(self, format='pdf', calls=True):
        graph = gv.Digraph(name='cluster'+self.name, format=format,
                           graph_attr={'label': self.name})
        id = 0
        i = 0
        while i < len(self.nodes):
            node = self.nodes[i]
            if node in pattern3:
                count = 0
                while self.nodes[i+1]==node:
                    count += 1
                    i+=1
                graph.node(str(id),label=node+":"+str(count))
                id +=1
                if id>=2:
                    graph.edge(str(id-2), str(id-1), label="next",_attributes={'style': 'dashed'})
            i+=1
        return graph

    def build_visual(self, filepath, format, calls=True, show=True):
        """
        Build a visualisation of the CFG with graphviz and output it in a DOT
        file.

        Args:
            filename: The name of the output file in which the visualisation
                      must be saved.
            format: The format to use for the output file (PDF, ...).
            show: A boolean indicating whether to automatically open the output
                  file after building the visualisation.
        """
        graph = self._build_visual(format, calls)
        graph.render(filepath, view=False)