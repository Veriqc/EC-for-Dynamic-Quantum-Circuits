import random
import numpy as np
from TDD.TDD import Index,Find_Or_Add_Unique_table,get_int_key,set_index_order,diag_matrix_2_TDD2,Matrix2TDD,if_then_else,get_index_order
from TDD.TDD import Ini_TDD,TDD,Single_qubit_gate_2TDD,diag_matrix_2_TDD,cnot_2_TDD,Slicing,contraction,TDD_2_matrix,Two_qubit_gate_2TDD
from TDD.TDD import measurement_2_TDD,condition_2_TDD,change_neig_index,single_qubit_state_2_TDD
from TDD.TDD_show import TDD_show
from cir_input.qasm import CreateCircuitFromQASM
from cir_input.circuit_DG import CreateDGfromQASMfile
from cir_input.gate_operation import OperationCNOT, OperationSingle,OperationU,OperationTwo
from qiskit.quantum_info.operators import Operator
import time
import datetime
import copy
import pandas as pd
from func_timeout import func_set_timeout
import func_timeout
import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_degree,treewidth_min_fill_in

def is_diagonal(U):
    i, j = np.nonzero(U)
    return np.all(i == j)

def get_classical_control_gate_num(cir):
    #classical bits are named c0,c1,...
    classical_control=dict()
    
    for k in range(len(cir._clbits)):
        c_idx='c'+str(k)
        classical_control[c_idx]=0

    gates=cir.data
    for k in range(len(gates)):
        if gates[k][0].condition:
            q_idx_con = gates[k][0].condition[0].name
            if len(q_idx_con)>=2:
                classical_control[q_idx_con]+=1
            else:
                bits = [int(k1) for k1 in list(bin(gates[k][0].condition[1])[2:])]
                num=1
                for k1 in range(len(bits)-1,-1,-1):
                    if bits[k1]==1:
                        q_idx_con = gates[k][0].condition[0].name+str(gates[k][0].condition[0].size-num)
                        classical_control[q_idx_con]+=1
                    num+=1
        else:
            continue
    return classical_control  

def add_hyper_index(var_list,hyper_index):
    for var in var_list:
        if not var in hyper_index:
            hyper_index[var]=0
            
def add_index_2_node(var_list,node,index_2_node):
    for var in var_list:
        if not var in index_2_node:
            index_2_node[var]=set([node])
        else:
            index_2_node[var].add(node)

def add_index_set(var_list,index_set):
    for var in var_list:
        index_set.add(var)
        
#do not use hyper edge for noisy gate
def get_tensor_index(cir):
    """return a dict that link every quantum gate to the corresponding index"""
    node_2_index=dict()
    index_2_node=dict()
    index_set = set()

    qubits_index = dict()
    hyper_index=dict()
    measure_position=dict()
    
    classical_control = get_classical_control_gate_num(cir)
    
    for k in range(len(cir._clbits)):
        c_idx=cir._clbits[k].register.name+str(cir._clbits[k].index)
        qubits_index[c_idx]=0
        measure_position[c_idx]=0
    for k in range(len(cir._qubits)):
        q_idx=cir._qubits[k].register.name+str(cir._qubits[k].index)
        qubits_index[q_idx]=0

    
    gates=cir.data
    for k in range(len(gates)):
        if gates[k][0].name == "barrier":
            continue
        if gates[k][0].name == "measure":
            if len(gates[k][2][0].register.name)>=2:
                q_idx_con = gates[k][2][0].register.name
            else:
                q_idx_con = gates[k][2][0].register.name+str(gates[k][2][0].index)
            q_idx_tar = gates[k][1][0].register.name+str(gates[k][1][0].index) 
            var_tar_in=q_idx_tar+str(0)+str(qubits_index[q_idx_tar])         
            add_hyper_index([var_tar_in],hyper_index)                
            gate_index = [Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_in,hyper_index[var_tar_in]+1)]
            hyper_index[var_tar_in]+=2
            measure_position[q_idx_con]=var_tar_in
            if classical_control[q_idx_con]==0:
                node_2_index[k] = gate_index
            else:
                node_2_index[k] = gate_index
                for k1 in range(classical_control[q_idx_con]):
                    node_2_index[k].append(Index(var_tar_in,hyper_index[var_tar_in]+k1))
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)
            continue
            
        if gates[k][0].condition:
            q_idx_tar = gates[k][1][0].register.name+str(gates[k][1][0].index)
            var_tar_in=q_idx_tar+str(0)+str(qubits_index[q_idx_tar])
            var_tar_out=q_idx_tar+str(0)+str(qubits_index[q_idx_tar]+1)
            add_hyper_index([var_tar_in,var_tar_out],hyper_index)            
            gate_index =[Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_out,hyper_index[var_tar_out])]
            qubits_index[q_idx_tar]+=1 
            node_2_index[k] = gate_index  
            
            if len(gates[k][0].condition[0].name)>=2:
                q_idx_con=gates[k][0].condition[0].name
                var_con_in=measure_position[q_idx_con]
                add_hyper_index([var_con_in],hyper_index)
                node_2_index[k].append(Index(var_con_in,hyper_index[var_con_in]))
                hyper_index[var_con_in]+=1
            else:
                bits = [int(k1) for k1 in list(bin(gates[k][0].condition[1])[2:])]
                num=1
                for k1 in range(len(bits)-1,-1,-1):
                    if bits[k1]==1:
                        q_idx_con = gates[k][0].condition[0].name+str(gates[k][0].condition[0].size-num)
                        var_con_in=measure_position[q_idx_con]
                        add_hyper_index([var_con_in],hyper_index)
                        node_2_index[k].append(Index(var_con_in,hyper_index[var_con_in]))
                        hyper_index[var_con_in]+=1 
                    num+=1
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)            
            continue
            
        if gates[k][0].name in {"cx","cu1","cz"}:
            q_idx_con = gates[k][1][0].register.name+str(gates[k][1][0].index)
            q_idx_tar = gates[k][1][1].register.name+str(gates[k][1][1].index)
            var_con=q_idx_con+str(0)+str(qubits_index[q_idx_con])
            var_tar_in=q_idx_tar+str(0)+str(qubits_index[q_idx_tar])
            var_tar_out=q_idx_tar+str(0)+str(qubits_index[q_idx_tar]+1)
            add_hyper_index([var_con,var_tar_in,var_tar_out],hyper_index)                
            gate_index =[Index(var_con,hyper_index[var_con]),Index(var_con,hyper_index[var_con]+1),Index(var_con,hyper_index[var_con]+2),Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_out,hyper_index[var_tar_out])]
            hyper_index[var_con]+=2
            qubits_index[q_idx_tar]+=1
            node_2_index[k] = gate_index
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)
            continue
        
        if len(gates[k][1])==1:
            q_idx = gates[k][1][0].register.name+str(gates[k][1][0].index)
            var_in = q_idx + str(0) + str(qubits_index[q_idx])
            var_out = q_idx + str(0) + str(qubits_index[q_idx]+1)
            add_hyper_index([var_in,var_out],hyper_index)
            if is_diagonal(Operator(gates[k][0]).data):
                gate_index = [Index(var_in,hyper_index[var_in]),Index(var_in,hyper_index[var_in]+1)]
                hyper_index[var_in]+=1
            else:
                gate_index = [Index(var_in,hyper_index[var_in]),Index(var_out,hyper_index[var_out])]
                qubits_index[q_idx] += 1
            node_2_index[k] = gate_index
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)            
            continue
            
        if len(gates[k][1])==2:
            q_idx_con = gates[k][1][0].register.name+str(gates[k][1][0].index)
            q_idx_tar = gates[k][1][1].register.name+str(gates[k][1][1].index)            
            var_con_in=q_idx_con+str(0)+str(qubits_index[q_idx_con])
            var_con_out=q_idx_con+str(0)+str(qubits_index[q_idx_con]+1)
            var_tar_in=q_idx_tar+str(0)+str(qubits_index[q_idx_tar])
            var_tar_out=q_idx_tar+str(0)+str(qubits_index[q_idx_tar]+1)
            add_hyper_index([var_con_in,var_con_out,var_tar_in,var_tar_out],hyper_index) 
            if is_diagonal(Operator(gates[k][0]).data):
                gate_index=[Index(var_con_in,hyper_index[var_con_in]),Index(var_con_in,hyper_index[var_con_in]+1),Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_in,hyper_index[var_tar_in]+1)]
                hyper_index[var_con_in]+=1
                hyper_index[var_tar_in]+=1
            else:                      
                gate_index=[Index(var_con_in,hyper_index[var_con_in]),Index(var_con_out,hyper_index[var_con_out]),Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_out,hyper_index[var_tar_out])]
                qubits_index[q_idx_con]+=1
                qubits_index[q_idx_tar]+=1
                
            node_2_index[k] = gate_index
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)  
            continue
            
        if gates[k][0].name == "ccx":
            q_idx_con1 = gates[k][1][0].register.name+str(gates[k][1][0].index)
            q_idx_con2 = gates[k][1][1].register.name+str(gates[k][1][1].index)
            q_idx_tar = gates[k][1][2].register.name+str(gates[k][1][2].index)
            var_con1=q_idx_con1+str(0)+str(qubits_index[q_idx_con1])
            var_con2=q_idx_con2+str(0)+str(qubits_index[q_idx_con2])
            var_tar_in=q_idx_tar+str(0)+str(qubits_index[q_idx_tar])
            var_tar_out=q_idx_tar+str(0)+str(qubits_index[q_idx_tar]+1)
            add_hyper_index([var_con1,var_con2,var_tar_in,var_tar_out],hyper_index)                
            gate_index =[Index(var_con1,hyper_index[var_con1]),Index(var_con1,hyper_index[var_con1]+1),Index(var_con1,hyper_index[var_con1]+2),
                         Index(var_con2,hyper_index[var_con2]),Index(var_con2,hyper_index[var_con2]+1),Index(var_con2,hyper_index[var_con2]+2),Index(var_tar_in,hyper_index[var_tar_in]),Index(var_tar_out,hyper_index[var_tar_out])]
            hyper_index[var_con1]+=2
            hyper_index[var_con2]+=2
            qubits_index[q_idx_tar]+=1
            node_2_index[k] = gate_index
            add_index_2_node([index.key for index in node_2_index[k]],k,index_2_node)
            add_index_set([index.key for index in node_2_index[k]],index_set)
            continue
        
        
        if len(gates[k][1])>2:
            print("Not supported yet: multi-qubits gates")
            break

    for k in range(len(cir._qubits)):
        last1='q'+str(k)+str(0)+str(qubits_index['q'+str(k)])
        new1='y'+str(k)
        last2='q'+str(k)+str(0)+str(0)
        new2='x'+str(k)
        if qubits_index['q'+str(k)]!=0:
            index_set.remove(last1)
            index_set.add(new1)
            index_set.remove(last2)
            index_set.add(new2)
            index_2_node[new1]=index_2_node[last1]
            index_2_node[new2]=index_2_node[last2]
            index_2_node.pop(last1)
            index_2_node.pop(last2)
        elif last1 in index_set:
            index_set.remove(last1)
            index_set.add(new1)
            index_2_node[new1]=index_2_node[last1]
            index_2_node.pop(last1)
        for m in node_2_index:
            node_2_index[m]=[Index(new1,item.idx) if item.key ==last1 else item for item in node_2_index[m]]
            node_2_index[m]=[Index(new2,item.idx) if item.key ==last2 else item for item in node_2_index[m]]    
    
    return node_2_index,index_2_node,index_set

def get_tdd(gate,var_list,involve_qubits):
    """get the TDD of the correct part of quantum gate"""
    nam=gate[0].name
    
    if nam == "barrier":
        node=Find_Or_Add_Unique_table(1,0,0,None,None)
        return TDD(node)
    if nam == "measure":
        node=Find_Or_Add_Unique_table(1,0,0,None,None)
        tdd=TDD(node)
        tdd.index_set=var_list
        return tdd
        
    if gate[0].condition:
        return condition_2_TDD(Operator(gate[0]).data,var_list)
    
    u_matrix=Operator(gate[0]).data
    if nam in {'cx','cu1','cz'}:
        if gate[1][0].index in involve_qubits and not gate[1][1].index in involve_qubits:
            node=Find_Or_Add_Unique_table(1,0,0,None,None)
            tdd=TDD(node)
            tdd.index_set=var_list[:3]
            return tdd
        U=np.array([[u_matrix[1,1],u_matrix[1,3]],[u_matrix[3,1],u_matrix[3,3]]])
        if not gate[1][0].index in involve_qubits and gate[1][1].index in involve_qubits:
            low=Single_qubit_gate_2TDD(np.eye(2),var_list[3:])    
            high=Single_qubit_gate_2TDD(U,var_list[3:])
            tdd=if_then_else(var_list[2].key,low,high)
            v=min(var_list[3],var_list[4])
            v_key=v.key
            if var_list[1]>v:
                tdd=change_neig_index(tdd,var_list[1].key,v_key)
            v=max(var_list[3],var_list[4])
            v_key=v.key
            if var_list[1]>v:
                tdd=change_neig_index(tdd,var_list[1].key,v_key)
            tdd.index_set=[var_list[1],var_list[3],var_list[4]]
            return tdd
    if nam =='ccx':
        if gate[1][0].index in involve_qubits:
            node=Find_Or_Add_Unique_table(1,0,0,None,None)
            tdd=TDD(node)
            tdd.index_set=var_list[:3]
            return tdd    
        if gate[1][1].index in involve_qubits:
            node=Find_Or_Add_Unique_table(1,0,0,None,None)
            tdd=TDD(node)
            tdd.index_set=var_list[3:6]
            return tdd     
        if gate[1][2].index in involve_qubits:
            low=Single_qubit_gate_2TDD(np.eye(2),var_list[6:])
            X=np.array([[0,1],[1,0]])
            high=Single_qubit_gate_2TDD(X,var_list[6:])
            x=max(var_list[2],var_list[5])
            x_key=x.key
            high=if_then_else(x_key,low,high)
            v=min(var_list[6],var_list[7])
            v_key=v.key
            if x>v:
                high=change_neig_index(high,x_key,v_key)
            v=max(var_list[6],var_list[7])
            v_key=v.key
            if x>v:
                high=change_neig_index(high,x_key,v_key)
            x=min(var_list[1],var_list[4])
            x_key=x.key
            high=if_then_else(x_key,low,high)
            v=min(var_list[6],var_list[7])
            v_key=v.key
            if x>v:
                high=change_neig_index(high,x_key,v_key)
            v=max(var_list[6],var_list[7])
            v_key=v.key
            if x>v:
                high=change_neig_index(high,x_key,v_key)                
            tdd=high
            tdd.index_set=[var_list[1],var_list[4],var_list[6],var_list[7]]
            return tdd       
    
    if len(gate[1]) ==1:
        if is_diagonal(u_matrix) and nam != 'noisy':
            return diag_matrix_2_TDD(u_matrix,var_list)
        else:
            return Single_qubit_gate_2TDD(u_matrix,var_list)
    if len(gate[1])== 2:
        if is_diagonal(u_matrix):
            return diag_matrix_2_TDD2(u_matrix,var_list)
        else:
            return Two_qubit_gate_2TDD(u_matrix,var_list)
        
def get_tdd_of_a_part_circuit(involve_nodes,involve_qubits,cir,node_2_index):
    """get the TDD of a part of circuit"""
#     print('involve_nodes',involve_nodes)
    compute_time = time.time()
    node=Find_Or_Add_Unique_table(1,0,0,None,None)
    tdd=TDD(node)
    max_node_num = 0
    
    gates=cir.data
    
    for k in involve_nodes:
        if gates[k][0].name == "barrier":
            continue
        temp_tdd = get_tdd(gates[k],node_2_index[k],involve_qubits)
        
        tdd=contraction(tdd,temp_tdd)
        max_node_num=max(max_node_num,tdd.node_number())
    #print('get_part_time:',time.time()-compute_time)
    return tdd,max_node_num

def get_gates_per_bit(cir):
    res=[]
    for k in range(len(cir._qubits)):
        res.append([])
    gates=cir.data
    for k in range(len(gates)):
        for q in range(len(gates[k][1])):
            res[gates[k][1][q].index].append(k)
    return res

def get_TDD_of_DQC(cir,ini_tdd=True):
    
    max_idx=20
    
    if ini_tdd:
        var_order=[]
        for k in range(len(cir._clbits)):
            for k1 in range(max_idx):
                c_idx='c'+str(k)+str(0)+str(k1)
                var_order.append(c_idx)
        for k in range(len(cir._qubits)):
            for k1 in range(max_idx):
                q_idx=cir._qubits[k].register.name+str(cir._qubits[k].index)+str(0)+str(k1)
                var_order.append(q_idx)
            q_idx=cir._qubits[k].register.name+str(cir._qubits[k].index)+str(0)+'y'
            var_order.append(q_idx)
        Ini_TDD(var_order)

    max_node_num = 0
    
    node = Find_Or_Add_Unique_table(1,0,0,None,None)
    tdd = TDD(node)
    
    node_2_index,index_2_node,index_set = get_tensor_index(cir)

    cir_partition=get_gates_per_bit(cir)

    for k in range(len(cir_partition)):
        temp_tdd,max_node1=get_tdd_of_a_part_circuit(cir_partition[k],[k],cir,node_2_index)
        max_node_num=max(max_node_num,max_node1)
        tdd=contraction(temp_tdd,tdd)
        max_node_num=max(max_node_num,tdd.node_number())
        
    return tdd,max_node_num

def Equivalence_checking_of_DQC(cir1,cir2):
    max_idx=20
    var_order=[]
    for k in range(len(cir1._clbits)):
        for k1 in range(max_idx):
            c_idx='c'+str(k)+str(0)+str(k1)
            var_order.append(c_idx)
            
    for k in range(len(cir2._clbits)):      
        for k1 in range(max_idx):
            c_idx='c'+str(k)+str(0)+str(k1)
            if not c_idx in var_order:
                var_order.append(c_idx)
                
    for k in range(len(cir1._qubits)):
        q_idx='x'+str(k)
        var_order.append(q_idx)          
        for k1 in range(max_idx):
            q_idx=cir1._qubits[k].register.name+str(cir1._qubits[k].index)+str(0)+str(k1)
            var_order.append(q_idx)
        q_idx=cir1._qubits[k].register.name+str(cir1._qubits[k].index)+str(0)+'y'
        var_order.append(q_idx)
        q_idx='y'+str(k)
        var_order.append(q_idx)          
    for k in range(len(cir2._qubits)):
        for k1 in range(max_idx):
            q_idx=cir2._qubits[k].register.name+str(cir2._qubits[k].index)+str(0)+str(k1)
            if not q_idx in var_order:
                var_order.append(q_idx)
        q_idx=cir2._qubits[k].register.name+str(cir2._qubits[k].index)+str(0)+'y'
        if not q_idx in var_order:
            var_order.append(q_idx)
        
    tdd1 = Ini_TDD(var_order)
    tdd2 = tdd1.self_copy()
    max_node_num=0        
    node_2_index1,index_2_node1,index_set1 = get_tensor_index(cir1)
    cir_partition1=get_gates_per_bit(cir1)        
    node_2_index2,index_2_node2,index_set2 = get_tensor_index(cir2)
    cir_partition2=get_gates_per_bit(cir2)  
    
    if not len(cir_partition1)==len(cir_partition2):
        return tdd1,tdd2,max_node_num,False
    for k in range(len(cir_partition1)):
        temp_tdd1,max_node1=get_tdd_of_a_part_circuit(cir_partition1[k],[k],cir1,node_2_index1)
        temp_tdd2,max_node2=get_tdd_of_a_part_circuit(cir_partition2[k],[k],cir2,node_2_index2)
        max_node_num=max(max_node_num,max_node1)
        max_node_num=max(max_node_num,max_node2)
        if not temp_tdd1==temp_tdd2:
#             print('Not identical of this qubit!')
            tdd1=contraction(temp_tdd1,tdd1)
            tdd2=contraction(temp_tdd2,tdd2)
            max_node_num=max(max_node_num,tdd1.node_number())
            max_node_num=max(max_node_num,tdd2.node_number())

    return tdd1,tdd2,max_node_num,tdd1==tdd2

def m_eq(tdd1,tdd2,index_set):
    if tdd1==tdd2:
        return True
    v1=tdd1.node.key
    v2=tdd2.node.key
    if var_order.index(v1)<=var_order.index(v2):
        x=v1
    else:
        x=v2
    if x in index_set:
        return m_eq(Slicing(tdd1,x,0),Slicing(tdd2,x,0),index_set) and m_eq(Slicing(tdd1,x,1),Slicing(tdd2,x,1),index_set)
    else:
        return get_norm(tdd1)==get_norm(tdd2)
    
    
@func_set_timeout(3600)
def m_Equivalence(cir1,cir2,index_set,input_state1={},input_state2={}):
    #index_set are input and measured index,input_state1,input_state2 are initialised input of the two circuit, given in the TDD form
    max_idx=20
    var_order=copy.copy(index_set)
    for k in range(len(cir1._clbits)):
        c_idx="c"+str(k)
        if not c_idx in var_order:
            var_order.append(c_idx)
    
    for k in range(len(cir1._qubits)):
        q_idx='x'+str(k)
        if not q_idx in var_order:
            var_order.append(q_idx)
        for k1 in range(max_idx):
            q_idx=cir1._qubits[k].register.name+str(cir1._qubits[k].index)+str(0)+str(k1)
            if not q_idx in var_order:
                var_order.append(q_idx)
        q_idx='y'+str(k)
        if not q_idx in var_order:
            var_order.append(q_idx)
        
    tdd1 = Ini_TDD(var_order)
    tdd2 = tdd1.self_copy()
    max_node_num=0        
    node_2_index1,index_2_node1,index_set1 = get_tensor_index(cir1)
    cir_partition1=get_gates_per_bit(cir1)        
    node_2_index2,index_2_node2,index_set2 = get_tensor_index(cir2)
    cir_partition2=get_gates_per_bit(cir2)  
    
    if not len(cir_partition1)==len(cir_partition2):
        return tdd1,tdd2,max_node_num,False
    for k in range(len(cir_partition1)):
        temp_tdd1,max_node1=get_tdd_of_a_part_circuit(cir_partition1[k],[k],cir1,node_2_index1)
        temp_tdd2,max_node2=get_tdd_of_a_part_circuit(cir_partition2[k],[k],cir2,node_2_index2)
        max_node_num=max(max_node_num,max_node1)
        max_node_num=max(max_node_num,max_node2)
        if k in input_state1:
            temp=single_qubit_state_2_TDD(input_state1[k],[Index('x'+str(k),0)])
            temp_tdd1= contraction(temp_tdd1,temp)
            max_node_num=max(max_node_num,temp_tdd1.node_number())
        if k in input_state2:
            temp=single_qubit_state_2_TDD(input_state2[k],[Index('x'+str(k),0)])
            temp_tdd2= contraction(temp_tdd2,temp)
            max_node_num=max(max_node_num,temp_tdd2.node_number())            
        if not temp_tdd1==temp_tdd2:
#             print('Not identical of this qubit!')
            tdd1=contraction(temp_tdd1,tdd1)
            tdd2=contraction(temp_tdd2,tdd2)
            max_node_num=max(max_node_num,tdd1.node_number())
            max_node_num=max(max_node_num,tdd2.node_number())
    
    return tdd1,tdd2,max(max_node1,max_node2),m_eq(tdd1,tdd2,index_set)

def if_only_one_state(tdd,index_set):
    #index_set are measurement index
    v=tdd.node.key
    if v in index_set:
        res1,state1 = if_only_one_state(Slicing(tdd,v,0),index_set)
        if res1==False:
            return False,None
        res2 ,state2= if_only_one_state(Slicing(tdd,v,1),index_set)
        if res2 ==False:
            return False,None
        if state1.node==state2.node:
            return True,state1
        else:
            return False,None
    return True,tdd
    
    
def q_eq(tdd1,tdd2,index_set):
    res1,state1=if_only_one_state(tdd1,index_set)
    if res1==False:
        return False
    res2,state2=if_only_one_state(tdd2,index_set)
    if res2==False:
        return False    
    if state1.node==state2.node:
        return True
    else:
        return False    

@func_set_timeout(3600)
def q_Equivalence(cir1,cir2,index_set,input_state1={},input_state2={}):
    max_idx=20
    var_order=copy.copy(index_set)
    for k in range(len(cir1._clbits)):
        c_idx="c"+str(k)
        if not c_idx in var_order:
            var_order.append(c_idx)
    
    for k in range(len(cir1._qubits)):
        q_idx='x'+str(k)
        if not q_idx in var_order:
            var_order.append(q_idx)
        for k1 in range(max_idx):
            q_idx=cir1._qubits[k].register.name+str(cir1._qubits[k].index)+str(0)+str(k1)
            if not q_idx in var_order:
                var_order.append(q_idx)
        q_idx='y'+str(k)
        if not q_idx in var_order:
            var_order.append(q_idx)
            
    tdd1 = Ini_TDD(var_order)
    tdd2 = tdd1.self_copy()
    max_node_num=0        
    node_2_index1,index_2_node1,index_set1 = get_tensor_index(cir1)
    cir_partition1=get_gates_per_bit(cir1)        
    node_2_index2,index_2_node2,index_set2 = get_tensor_index(cir2)
    cir_partition2=get_gates_per_bit(cir2)
    
    if not len(cir_partition1)==len(cir_partition2):
        return tdd1,tdd2,max_node_num,False
    for k in range(len(cir_partition1)):
        temp_tdd1,max_node1=get_tdd_of_a_part_circuit(cir_partition1[k],[k],cir1,node_2_index1)
        temp_tdd2,max_node2=get_tdd_of_a_part_circuit(cir_partition2[k],[k],cir2,node_2_index2)
        max_node_num=max(max_node_num,max_node1)
        max_node_num=max(max_node_num,max_node2)
        if k in input_state1:
            temp=single_qubit_state_2_TDD(input_state1[k],[Index('x'+str(k),0)])
            temp_tdd1= contraction(temp_tdd1,temp)
            max_node_num=max(max_node_num,temp_tdd1.node_number())
        if k in input_state2:
            temp=single_qubit_state_2_TDD(input_state2[k],[Index('x'+str(k),0)])
            temp_tdd2= contraction(temp_tdd2,temp)
            max_node_num=max(max_node_num,temp_tdd2.node_number())
            
        temp_qqq=temp_tdd1.self_copy()
        if not q_eq(temp_tdd1,temp_tdd2,index_set):
#             print('Not identical of this qubit!')
            tdd1=contraction(temp_tdd1,tdd1)
            tdd2=contraction(temp_tdd2,tdd2)
            max_node_num=max(max_node_num,tdd1.node_number())
            max_node_num=max(max_node_num,tdd2.node_number())
    
    return tdd1,tdd2,max(max_node1,max_node2),q_eq(tdd1,tdd2,index_set)

def get_gate_num(cir):
    gates=cir.data
    gate_num=0
    for k in range(len(gates)):
        if gates[k][0].name in {"barrier","measure"}:
            continue
        gate_num+=1
    return gate_num

if __name__=="__main__":
    path='Benchmarks/'
    file_name='teleportation.qasm'
    file_name2='DQC_teleportation.qasm'
    cir1= CreateCircuitFromQASM(file_name, path)
    cir2= CreateCircuitFromQASM(file_name2, path)    
    index_set=['y0','y1']
    v=np.array([1,0])
    input_state1={1:v,2:v}
    input_state2={1:v,2:v}
    t_start=time.time()
    tdd1,tdd2,max_nodes_num,res = q_Equivalence(cir1,cir2,index_set,input_state1,input_state2)
    t_end=time.time()
    print('Is equal:',res)
    print('time:',t_end-t_start)
    print('tdd1_node_num_final:',tdd1.node_number())
    print('tdd2_node_num_final:',tdd2.node_number())
    print('node_num_max:',max_nodes_num)
#     TDD_show(tdd1)
#     cir1.draw()