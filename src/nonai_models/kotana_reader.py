import os

def kotana_reader():
    file = open(os.path.join(os.path.dirname(__file__),"processor_output.txt"),'r')
    lines = []
    for line in file:
        inst, cycle, number, instruction = line.split(" ")
        lines.append(instruction)
# {"I", "L", "S", "W","C", "E","R", "D"}
stage =  {"F" "Rn" "Wat", "Sr", "Sw", "Wb", "Cm"}
riscv_instruction_set = {'fmadd.d', 'addi','fld','addi','bne','add','slli','ld', "iLD", "iALU", "fLD", "iSFT", "fMUL"}
inst_energy = {}
total_cycles = 5066

def kotana_graph(lines):
    energy = 0
    for inst in lines:
        if inst in riscv_instruction_set:
            index = riscv_instruction_set.find(inst)
            energy += inst_energy[index]
        if inst in stage:
            index = riscv_instruction_set.find(inst)
            energy += inst_energy[index]

    print("Total energy Consumption:", energy)
    print("Total Power consumption:", energy/total_cycles)
