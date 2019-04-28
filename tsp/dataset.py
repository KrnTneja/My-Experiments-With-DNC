import itertools
import numpy as np
import torch as T
from torch.autograd import Variable as var

def get_bit_sequence(x, bits=7):
    assert len(x.shape) == 1
    bit_seq = np.zeros((len(x), bits), dtype=np.float32)
    for i in range(len(x)):
        bit_seq[i,:] = [float(bi) for bi in "{:0{bits}b}".format(x[i],bits=bits)]
    return bit_seq

def get_distance(matrix,path):
    output = 0;
    for index in range(1,len(path)):
        output += matrix[path[index-1]][path[index]]
    return output

def get_example(N, bits=7):
    max_val = 2**bits
    dist_matrix = np.random.randint(1,max_val,size = (N,N))
    dist_matrix = np.floor_divide(dist_matrix + dist_matrix.transpose(),2)
    np.fill_diagonal(dist_matrix,0)
    
    all_paths = itertools.permutations(range(N))
    curr_min = float('inf')
    curr_path = []

    for path in all_paths:
        dist = get_distance(dist_matrix,path)
        if(dist < curr_min):
            curr_min = dist
            curr_path = path

    return dist_matrix, curr_path

def generate_data(batch_size, bits=7, cuda=-1):
    

    for i in range(batch_size):
        M,path = get_example(10)
        print(M,path)

generate_data(1)
