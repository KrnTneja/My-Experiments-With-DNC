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

def construct_example(n, q, bits = 6):
    done = False

    m = int(np.ceil(n*np.log(n)/4))
    G = nx.gnm_random_graph(n,m)
    edges = np.array(G.edges())
    # print("edges",edges)
    edges = edges.flatten()
    queries = np.array([np.random.choice(n, 2, replace=False) for _ in range(q)])
    connected = np.zeros(q, dtype=np.float32)
    # print("queries",queries)
    for qi in range(q):
        try:
            shortest_paths = np.array([expand_shortest_path(p) for p in 
                                       nx.all_shortest_paths(G, queries[qi,0], queries[qi,1])])
            connected[qi] = 1.0
        except:
            connected[qi] = 0.0

    # print("shortest_paths",shortest_paths)
    # print("length_of_shortest_path",length_of_shortest_path)
    # print("connected", connected)

    edges = get_bit_sequence(edges, bits)
    queries = get_bit_sequence(queries.flatten())
    return edges, queries, connected


def generate_data(batch_size, bits = 6, cuda = -1):
    n = np.random.randint(10,25+1)
    q = np.random.randint(2,n)
    sample_edges = []
    sample_queries = []
    sample_connected = []

    for i in range(batch_size):
        edges, queries, connected = construct_example(n, q, bits)
        sample_edges.append(edges)
        sample_queries.append(queries)
        sample_connected.append(connected)

    m = sample_edges[0].shape[0]//2
    length = m + 1 + q + 1 + q + 1 # edges + eoe + queries + eoq + outputs + eoo 
    size = 2*bits+2
    input_data = np.zeros((batch_size, length, size), dtype=np.float32)
    target_data = np.zeros((batch_size, length, size), dtype=np.float32)

    for i in range(batch_size):
        input_data[i, :m, :bits] = sample_edges[i][::2]
        input_data[i, :m, bits:2*bits] = sample_edges[i][1::2]
        input_data[i, m+1:m+1+q, :bits] = sample_queries[i][::2]
        input_data[i, m+1:m+1+q, bits:2*bits] = sample_queries[i][1::2]
        target_data[i, m+1+q+1:m+1+q+1+q, 0] = sample_connected[i]

    input_data[:,m,-2] = 1
    input_data[:,m+1+q,-1] = 1
    target_data[:,-1,-1] = 1
        
    input_data = T.from_numpy(input_data)
    target_data = T.from_numpy(target_data)
    if cuda != -1:
        input_data = input_data.cuda(cuda)
        target_data = target_data.cuda(cuda)
    return var(input_data), var(target_data)

if __name__ == "__main__":
    input_d, target_d = generate_data(1)
    print("input_d[0]",target_d[0])
    print("input_d[0]",target_d[0])






