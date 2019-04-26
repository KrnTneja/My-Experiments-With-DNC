import numpy as np
import torch as T
from torch.autograd import Variable as var
import networkx as nx

# generate number of nodes randomly between [10,25]
# generate a Gnm graph 
# Check if the the graph is connected
# generate two nodes randomly
# find the shortest path between them

def get_bit_sequence(x, bits=6):
    assert len(x.shape) == 1
    bit_seq = np.zeros((len(x), bits), dtype=np.float32)
    for i in range(len(x)):
        bit_seq[i,:] = [float(bi) for bi in "{:0{bits}b}".format(x[i],bits=bits)]
    return bit_seq

def expand_shortest_path(p):
    pe = np.array([[el,el] for el in p]).flatten()
    pe = pe[1:-1]
    return pe

def construct_example(n, bits = 6):
    done = False
    length_of_shortest_path = 0
    while not done:
        m = int(np.ceil(n*np.log(n)))
        G = nx.gnm_random_graph(n,m)
        edges = np.array(G.edges())
        # print("edges",edges)
        edges = edges.flatten()
        query = np.random.choice(n, 2, replace=False)
        # print("query",query)
        try:
            shortest_paths = np.array([expand_shortest_path(p) for p in nx.all_shortest_paths(G, query[0], query[1])])
            done = True
        except:
            continue
        # print("shortest_paths",shortest_paths)
        length_of_shortest_path = len(shortest_paths[0])//2
        # print("length_of_shortest_path",length_of_shortest_path)
        shortest_paths = np.array([get_bit_sequence(shortest_paths[i], bits) for i in range(len(shortest_paths))])

        edges = get_bit_sequence(edges, bits)
        query = get_bit_sequence(query)
    return edges, query, shortest_paths, length_of_shortest_path


def generate_data(batch_size, bits=6, cuda =-1):
    n = np.random.randint(10,25+1)
    sample_edges = []
    sample_queries = []
    sample_shortest_paths = []
    sample_lengths = []

    for i in range(batch_size):
        edges, query, shortest_paths, length_of_shortest_path = construct_example(n, bits)
        sample_edges.append(edges)
        sample_queries.append(query)
        sample_shortest_paths.append(shortest_paths)
        sample_lengths.append(length_of_shortest_path)

    m = sample_edges[0].shape[0]//2
    # print("m",m)
    length = m + 1 + 1 + 1 + np.max(sample_lengths) + 1 # edges + eoe + query + eoq + output + eoo 
    size = 2*bits+2
    input_data = np.zeros((batch_size, length, size), dtype=np.float32)

    for i in range(batch_size):
        input_data[i, :m, :bits] = sample_edges[i][::2]
        input_data[i, :m, bits:2*bits] = sample_edges[i][1::2]
        input_data[i, m+1, :bits] = sample_queries[i][0]
        input_data[i, m+1, bits:2*bits] = sample_queries[i][1]

        
    input_data[:,m,-2] = 1
    input_data[:,m+2,-1] = 1

    def target_output(actual_output):
        actual_output = actual_output.cpu().detach().numpy()
        target_array = np.zeros((batch_size, length, size), dtype=np.float32)
        for i in range(batch_size):
            # find closest target output
            curr_output = actual_output[i,m+3:m+3+sample_lengths[i],:-2]
            # print("sample_lengths[i]",sample_lengths[i])
            curr_targets = sample_shortest_paths[i].reshape((len(sample_shortest_paths[i]),-1,2*bits))
            # print("curr_output",curr_output)
            # print("curr_targets",curr_targets)
            best_target = np.argmin(np.sum(np.abs(curr_targets - curr_output),axis=(1,2)))
            best_target = curr_targets[best_target]
            # constrct target array
            target_array[i,m+3:m+3+sample_lengths[i],:-2] = best_target
            target_array[i,m+3+sample_lengths[i]:,-1] = 1
        
        target_array = T.from_numpy(target_array)
        if cuda != -1:
            target_array = target_array.cuda(cuda)
        return var(target_array)

    input_data = T.from_numpy(input_data)
    if cuda != -1:
        input_data = input_data.cuda(cuda)
    return var(input_data), target_output

if __name__ == "__main__":
    sample_edges, sample_queries = generate_data(2)
    print("sample_edges[0]",sample_edges[0])
    print("functionreturn",sample_queries(sample_edges))






