import numpy as np
import torch as T
from torch.autograd import Variable as var

def get_bit_sequence(x, bits=7):
    assert len(x.shape) == 1
    bit_seq = np.zeros((len(x), bits), dtype=np.float32)
    for i in range(len(x)):
        bit_seq[i,:] = [float(bi) for bi in "{:0{bits}b}".format(x[i],bits=bits)]
    return bit_seq

def get_example(N, k, bits=7):
    max_N = 2**bits
    inp_sequence = np.arange(max_N)
    np.random.shuffle(inp_sequence)
    inp_sequence = inp_sequence[:N]
    sorted_top_k = np.sort(inp_sequence)[:k]
    inp_sequence = get_bit_sequence(inp_sequence, bits)
    sorted_top_k = get_bit_sequence(sorted_top_k, bits)
    return inp_sequence, sorted_top_k

# ---- Top-k sort ----
# 7 bits inputs + ?? flags
# Input: inp1, inp2, inp3, inp4, inp5, ..., inpN, input_end_flag, k, query_end_flag, zeroes
# Target: zeroes, ..., zeroes, top1, top2, ..., topk, end_output_flag
def generate_data(batch_size, bits=7, cuda=-1):
    max_N = 2**(bits-1)+1
    N = np.random.randint(1,max_N)
    k = np.random.randint(1,N+1)

    input_data = np.zeros((batch_size, N+3+k+1, bits+2), dtype=np.float32)
    target_output = np.zeros((batch_size, N+3+k+1, bits+2), dtype=np.float32)

    for i in range(batch_size):
        inp_sequence, sorted_top_k = get_example(N,k,bits)
        input_data[i, :N, :bits] = inp_sequence
        target_output[i, N+3:N+3+k, :bits] = sorted_top_k
        input_data[i, N+1, :bits] = get_bit_sequence(np.array([k]),bits)

    input_data[:, N, -2] = 1
    input_data[:, N+2, -1] = 1
    target_output[:, N+3+k, -1] = 1

    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)

    if cuda != -1:
        input_data = input_data.cuda(cuda)
        target_output = target_output.cuda(cuda)

    return var(input_data), var(target_output)

