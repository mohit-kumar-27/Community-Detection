import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations, permutations
from collections import defaultdict
import time

def import_facebook_data(path):

    data = np.genfromtxt(
    path,
    delimiter=" ",
    skip_header=False,
    dtype="int",)
    data1 = data.copy()
    data1[:,[0, 1]] = data1[:,[1, 0]]
    data = np.concatenate((data, data1), axis=0)
    data = data[data[:,0] != data[:,1]]
    data = np.unique(data, axis=0)
    
    return data

def import_bitcoin_data(path):
  
    data = np.genfromtxt(
    path,
    delimiter=",",
    skip_header=False,
    dtype="int",)
    data = data[:,[0,1]]
    data1 = data.copy()
    data1[:,[0, 1]] = data1[:,[1, 0]]
    data = np.concatenate((data, data1), axis=0)
    data = data[data[:,0] != data[:,1]]
    data = np.unique(data, axis=0)
    data = data-np.amin(data)

    return data

def spectralDecomp_OneIter(edgelist):

    nodes, indices = np.unique(edgelist.ravel(), return_inverse=True)
    nids = np.array(np.arange(nodes.shape[0]))
    edges = nids[indices].reshape(-1,2)
    adj_mat = coo_matrix((np.ones(edges.shape[0]),(edges[:,0], edges[:,1])))
    degrees = np.array(np.sum(adj_mat, axis = 1)).flatten()
    deg_mat = coo_matrix((degrees, (np.arange(adj_mat.shape[0]), np.arange(adj_mat.shape[0]))))
    L = deg_mat - adj_mat
    L = L.toarray()
    adj_mat = adj_mat.toarray()
    _, eigvecs = np.linalg.eigh(L)
    fv = eigvecs[:,1]
    cid0 = np.amin(nodes[fv<=0])
    cid1 = np.amin(nodes[fv>0])
    cids = np.zeros_like(nodes)
    cids[fv<=0] = cid0
    cids[fv>0] = cid1
    graph_partition =  np.hstack((nodes.reshape(-1,1), cids.reshape(-1,1)))

    return fv, adj_mat, graph_partition

def spectralDecomposition(edgelist):

    nodes, indices = np.unique(edgelist.ravel(), return_inverse=True)
    nids = np.array(np.arange(nodes.shape[0]))
    graph_partition = np.hstack((nids.reshape(-1,1), np.zeros((nids.shape[0],1))))
    edgelist = nids[indices].reshape(-1,2)
    tmp = [np.array(edgelist, copy=True)]
    while tmp:
        edges = tmp.pop(0)
        try:
          _, _, partition = spectralDecomp_OneIter(edges)
        except:
          continue
        c01edges = np.zeros((edges.shape[0]), dtype= bool)
        c0edges = np.zeros((edges.shape[0]), dtype= bool)
        c1edges = np.zeros((edges.shape[0]), dtype= bool)
        cid, count = np.unique(partition[:,1],return_counts=True)
        com0 = partition[partition[:,1] == cid[0],:]
        com1 = partition[partition[:,1] == cid[1],:]
        c0nodes = set(list(com0[:,0]))
        c1nodes = set(list(com1[:,0])) 
        for i in range(edges.shape[0]):
            s,t = edges[i][0], edges[i][1]
            c0edges[i] = s in c0nodes and t in c0nodes
            c1edges[i] = s in c1nodes and t in c1nodes
            c01edges[i] = (not c0edges[i]) and (not c1edges[i])
        c01edges = edges[c01edges,:]
        ratio_cut = len(c01edges)/len(c0nodes) + len(c01edges)/len(c1nodes)
        if (ratio_cut > 0.75):
            continue
        graph_partition[com0[:,0],:] = com0
        graph_partition[com1[:,0],:] = com1
        c0edges = edges[c0edges,:]
        c1edges = edges[c1edges,:]
        if c0edges.shape[0] > 10:
            tmp.append(c0edges)
        if c1edges.shape[0] > 10:
            tmp.append(c1edges)
    graph_partition[:,0] = nodes
    
    return graph_partition

def createSortedAdjMat(graph_partition, edgelist):

    adj_mat = coo_matrix((np.ones(edgelist.shape[0]),(edgelist[:,0], edgelist[:,1]))).toarray()
    idx = np.argsort(graph_partition[:,1])
    adj_mat = adj_mat[:,idx][idx,:]
    
    return adj_mat

# code taken from https://github.com/multinetlab-amsterdam/network_TDA_tutorial/blob/main/1-network_analysis.ipynb 
# (for visualization of communities within the network)

def community_layout(g, partition):

    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)
    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))
    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)
    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()
    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]
        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]
    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def calc_node_wts(edge_wts):
    # Calculates node wts (sum of edge wts of a node)

    node_wts = defaultdict(float)
    for node in edge_wts.keys():
        node_wts[node] = sum([w for w in edge_wts[node].values()])

    return node_wts

def get_neighbor_nodes(node, edge_wts):
    # Returns neighbors of a node along with edge wts

    if node not in edge_wts:
        return 0

    return edge_wts[node].items()

def get_node_wt_in_cluster(node, node2com, edge_wts):
    # Calculates node wts (sum of edge wts of a node) within a cluster/community

    node_com = node2com[node]
    nei_nodes = get_neighbor_nodes(node, edge_wts)
    wts = 0.
    for nei_node in nei_nodes:
        if node_com == node2com[nei_node[0]]:
            wts += nei_node[1]

    return wts

def get_tot_wt(node, node2com, edge_wts):
    # Calculates total weight of nodes in a community 

    nodes = [n for n, cid in node2com.items() if cid == node2com[node] and node != n]
    wt = 0.
    for n in nodes:
        wt += sum(list(edge_wts[n].values()))

    return wt

def get_cluster_deg(nodes, edge_wts):
    # Calculates the cluster degree

    return sum([sum(list(edge_wts[n].values())) for n in nodes])

def first_phase(node2com, edge_wts):

    node_wts = calc_node_wts(edge_wts)
    all_edge_wts = sum([wt for start in edge_wts.keys() for end, wt in edge_wts[start].items()]) / 2
    flag = True
    while flag:
        flags = []
        for node in node2com.keys():
            flags = []
            nei_nodes = [edge[0] for edge in get_neighbor_nodes(node, edge_wts)]
            max_delta = 0.
            cid = node2com[node]
            mcid = cid
            communities = {}
            for nei_node in nei_nodes:
                node2com_copy = node2com.copy()
                if node2com_copy[nei_node] in communities:
                    continue
                communities[node2com_copy[nei_node]] = 1
                node2com_copy[node] = node2com_copy[nei_node]
                delta_q = 2 * get_node_wt_in_cluster(node, node2com_copy, edge_wts) - (get_tot_wt(node, node2com_copy, edge_wts) * node_wts[node] / all_edge_wts)
                if delta_q > max_delta:
                    mcid = node2com_copy[nei_node]
                    max_delta = delta_q           
            flags.append(cid != mcid)
            node2com[node] = mcid
        if sum(flags) == 0:
            break

    return node2com, node_wts

def second_phase(node2com, edge_wts):

    new_edge_wts = defaultdict(lambda : defaultdict(float))
    node2com_new = {}
    com2node = defaultdict(list)
    for node, cid in node2com.items():
        com2node[cid].append(node)
        if cid not in node2com_new:
            node2com_new[cid] = cid
    nodes = list(node2com.keys())
    node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
    for edge in node_pairs:
        new_edge_wts[node2com_new[node2com[edge[0]]]][node2com_new[node2com[edge[1]]]] += edge_wts[edge[0]][edge[1]]

    return node2com_new, new_edge_wts

def partition_update(node2com_new, partition):
    # Updates the community id of the nodes

    reverse_partition = defaultdict(list)
    for node, cid in partition.items():
        reverse_partition[cid].append(node)
    for old_cid, new_cid in node2com_new.items():
        for old_com in reverse_partition[old_cid]:
            partition[old_com] = new_cid

    return partition

def louvain_one_iter(edgelist):

  G = nx.Graph()
  G.add_edges_from(edgelist)
  node2com = {}
  edge_wts = defaultdict(lambda : defaultdict(float))
  for idx, node in enumerate(G.nodes()):
      node2com[node] = idx
      for edge in G[node].items():
          edge_wts[node][edge[0]] = 1.0
  node2com, node_wts = first_phase(node2com, edge_wts)
  partition = node2com.copy()
  node2com_new, new_edge_wts = second_phase(node2com, edge_wts)  
  partition = partition_update(node2com_new, partition)
  
  return np.array(list(partition.items()))

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    print('---------------------------------------------------------------------------------------')
    print('https://snap.stanford.edu/data/ego-Facebook.html dataset')
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)


    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    print('Spectral Decomposition Technique')
    start = time.time()
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected: {np.unique(graph_partition_fb[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_fb}
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G, pos=community_layout(G, part), font_size=8, node_size=clust, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)
    plt.figure(figsize=(5,5))
    plt.imshow(clustered_adj_mat_fb, cmap="Greys", interpolation="none")
    plt.show()
    print('Sorted Adjacency Matrix')

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    print('Louvain algorithm')
    start = time.time()
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected: {np.unique(graph_partition_louvain_fb[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_louvain_fb}
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G, pos=community_layout(G, part), font_size=8, node_size=clust, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')

    print('---------------------------------------------------------------------------------------')

    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    print('https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html dataset')
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    print('Spectral Decomposition Technique')
    start = time.time()
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected:{np.unique(graph_partition_btc[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_btc}
    G = nx.Graph(adj_mat_btc, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    try:
      nx.draw(G, pos=community_layout(G, part), font_size=8, node_size=clust, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
      plt.show()
      print('visualization of communities in the network')
    except:
      print("")

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)
    plt.figure(figsize=(5,5))
    plt.imshow(clustered_adj_mat_btc, cmap="Greys", interpolation="none")
    plt.show()
    print('Sorted Adjacency Matrix')

    # Question 4
    print('Louvain algorithm')
    start = time.time()
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected:{np.unique(graph_partition_louvain_btc[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_louvain_btc}
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    try:
      nx.draw(G, pos=community_layout(G, part), font_size=8, node_size=clust, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
      plt.show()
      print('visualization of communities in the network')
    except:
      print("")