# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit

def CreateCircuitFromQASM(file, path):
    #QASM_file = open('inputs/QASM example/' + file, 'r')
    QASM_file = open(path + file,
                     'r')
    iter_f = iter(QASM_file)
    QASM = ''
    for line in iter_f: #遍历文件，一行行遍历，读取文本
#         if line.split(' ')[0] in {'barrier','measure'}:
#             continue
        QASM = QASM + line
#     print(QASM)
    cir = QuantumCircuit.from_qasm_str(QASM)
    QASM_file.close
    
    return cir