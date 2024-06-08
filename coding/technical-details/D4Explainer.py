
# All code copied from: 
# - https://github.com/Graph-and-Geometric-Learning/D4Explainer/blob/main/utils/dist_helper.py
# - https://github.com/Graph-and-Geometric-Learning/D4Explainer/blob/main/evaluation/in_distribution/ood_stat.py

import concurrent.futures
from functools import partial

import copy
import os
import subprocess as sp
from datetime import datetime

import networkx as nx
import numpy as np
import pyemd
from scipy.linalg import eigvalsh
from scipy.linalg import toeplitz

# dist_helper.py

# NOTES:
# EMD stands for earth move distance, i.e. Wasserstein metric,
# (\inf_{\gama \in \Gama(\mu, \nu) \int_{M*M} d(x,y)^p d\gama(x,y))^(1/p)

def emd(x, y, distance_scaling=1.0):
    """
    Earth Mover's Distance (EMD) between two 1D pmf
    :param x: 1D pmf
    :param y: 1D pmf
    :param distance_scaling: scaling factor for distance matrix
    :return: EMD distance
    """
    x = x.astype(float)
    y = y.astype(float)
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)  # diagonal-constant matrix
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def l2(x, y):
    """
    L2 distance between two 1D pmf
    :param x: 1D pmf
    :param y: 1D pmf
    :return: L2 distance
    """
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """
    Gaussian kernel with squared distance in exponential term replaced by EMD
    :param x: 1D pmf
    :param y: 1D pmf
    :param sigma: standard deviation
    :param distance_scaling: scaling factor for distance matrix
    :return: Gaussian kernel with EMD
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """
    Discrepancy between 2 samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_parallel: whether to use parallel computation
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """
    MMD between two samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_hist: whether the samples are histograms or pmf
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """
    EMD between average of two samples
    :param samples1: list of samples
    :param samples2: list of samples
    :param kernel: kernel function
    :param is_hist: whether the samples are histograms or pmf
    :param args: args for kernel
    :param kwargs: kwargs for kernel
    """
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


def test():
    s1 = np.array([0.2, 0.8])
    s2 = np.array([0.3, 0.7])
    samples1 = [s1, s2]

    s3 = np.array([0.25, 0.75])
    s4 = np.array([0.35, 0.65])
    samples2 = [s3, s4]

    s5 = np.array([0.8, 0.2])
    s6 = np.array([0.7, 0.3])
    samples3 = [s5, s6]

    # print(
    #     "between samples1 and samples2: ",
    #     compute_emd(samples1, samples2, kernel=gaussian_emd, is_parallel=False, sigma=1.0),
    # )
    # print(
    #     "between samples1 and samples3: ",
    #     compute_emd(samples1, samples3, kernel=gaussian_emd, is_parallel=False, sigma=1.0),
    # )
    print(
        "between samples1 and samples2: ",
        compute_mmd(samples1, samples2, kernel=gaussian, is_parallel=True, sigma=1.0),
    )
    print(
        "between samples1 and samples3: ",
        compute_mmd(samples1, samples3, kernel=gaussian, is_parallel=True, sigma=1.0),
    )
    print(
        "between samples1 and samples2: ",
        compute_mmd(samples1, samples2, kernel=gaussian, is_parallel=True, sigma=1.0),
    )
    print(
        "between samples1 and samples3: ",
        compute_mmd(samples1, samples3, kernel=gaussian, is_parallel=True, sigma=1.0),
    )


def process_tensor(x, y):
    """
    Helper function to pad tensors to the same size
    :param x: tensor
    :param y: tensor
    :return: padded tensors
    """
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y

# ood_stat.py 

PRINT_TIME = False
ORCA_DIR = "orca"  # the relative path to the orca dir


def degree_worker(G):
    """
    Compute the degree distribution of a graph.
    :param G: a networkx graph
    :return: a numpy array of the degree distribution
    """
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    """
    Add two tensors. If unequal shape, pads the smaller one with zeros.
    :param x: a tensor
    :param y: a tensor
    :return: x + y
    """
    x, y = process_tensor(x, y)
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two degree distributions
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def spectral_worker(G):
    """
    Compute the spectral pmf of a graph.
    :param G: a networkx graph
    :return: a numpy array of the spectral pmf
    """
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two degree distributions
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    """
    Compute the clustering coefficient distribution of a graph.
    :param param: a tuple of (graph, number of bins)
    :return: a numpy array of the clustering coefficient distribution
    """
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    """
    Compute the distance between the clustering coefficient distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param bins: number of bins for the histogram
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two clustering coefficient distributions
    """
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {"3path": [1, 2], "4cycle": [8]}
COUNT_START_STR = "orbit counts: \n"


def edge_list_reindexed(G):
    """
    Convert a graph to a list of edges, where the nodes are reindexed to be integers from 0 to n-1.
    :param G: a networkx graph
    :return: a list of edges, where each edge is a tuple of integers
    """
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """
    Compute the orbit counts of a graph.
    :param graph: a networkx graph
    :return: a numpy array of shape (n, 2), where n is the number of nodes in the graph. The first column is the node index, and the second column is the orbit count.
    """
    tmp_file_path = os.path.join(ORCA_DIR, "tmp.txt")
    f = open(tmp_file_path, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, "orca"), "node", "4", tmp_file_path, "std"])
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array(
        [list(map(int, node_cnts.strip().split(" "))) for node_cnts in output.strip("\n").split("\n")]
    )

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    """
    Compute the distance between the orbit counts of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :return: the distance between the two orbit counts
    """
    total_counts_ref = []
    total_counts_pred = []

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception:
            continue

        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except Exception:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)

    return mmd_dist


def adjs_to_graphs(adjs):
    """
    Convert a list of adjacency matrices to a list of networkx graphs.
    :param adjs: a list of adjacency matrices
    :return: a list of networkx graphs
    """
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not (lobster -> caterpillar -> path)
    :param G: a networkx graph
    :return: True if the graph is a lobster graph, False otherwise
    """
    # Check if G is a tree
    if nx.is_tree(G):
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def eval_acc_lobster_graph(G_list):
    """
    Compute the accuracy of a list of graphs being lobster graphs.
    :param G_list: a list of networkx graphs
    :return: the accuracy of the list of graphs being lobster graphs
    """
    G_list = [copy.deepcopy(gg) for gg in G_list]

    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(G_list))


METHOD_NAME_TO_FUNC = {
    "degree": degree_stats,
    "cluster": clustering_stats,
    "orbit": orbit_stats_all,
    "spectral": spectral_stats,
}


def eval_graph_list(graph_ref_list, grad_pred_list, methods=None):
    """
    Compute the evaluation metrics for a list of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param grad_pred_list: a list of networkx graphs
    :param methods: a list of evaluation methods to be used
    :return: a dictionary of evaluation results
    """
    if methods is None:
        methods = ["degree", "cluster", "spectral", "orbit"]
    results = {}
    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list)
    if "orbit" not in methods:
        results["orbit"] = 0.0
    print(results)
    return results


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    """
    Compute the evaluation metrics for a batch of graphs.
    :param ref_batch: a batch of adjacency matrices
    :param pred_batch: a batch of adjacency matrices
    :param methods: a list of evaluation methods to be used
    :return: a dictionary of evaluation results
    """
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    grad_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, grad_pred_list, methods=methods)
    return results
