import warnings
import pandas as pd
import numpy as np

from torch_geometric.nn import knn_graph, radius_graph
from anndata import AnnData

import torch
from typing import List

def match_cluster_labels(true_labels,est_labels):
    import networkx as nx

    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)      





def batch_split(adata: AnnData,
                method: str,
                num_batch_x: int,
                num_batch_y: int,
                min_spots: int = 10) -> List[AnnData]:
    """Create spatial batches based on the method specified by the user."""
    if method == 'percentile':
        return percentile_split(adata, num_batch_x, num_batch_y, min_spots)
    elif method == 'interval':
        return interval_split(adata, num_batch_x, num_batch_y, min_spots)
    else:
        raise ValueError(f"Invalid method: {method}")


def percentile_split(adata: AnnData,
                     num_batch_x: int,
                     num_batch_y: int,
                     min_spots: int = 10) -> List[AnnData]:
    """Create spatial batches based on percentile boundaries of coordinates."""
    sp = adata.obsm['spatial']
    num_x, num_y = max(1, num_batch_x), max(1, num_batch_y)
    n_points = sp.shape[0]

    # Precompute percentile boundaries (vectorized)
    x_bounds = np.percentile(sp[:, 0], np.linspace(0, 100, num_x + 1))
    y_bounds = np.percentile(sp[:, 1], np.linspace(0, 100, num_y + 1))

    # Vectorized grid mask generation (avoid nested loops)
    x_in = np.logical_and(sp[:, 0, None] >= x_bounds[:-1], 
                          sp[:, 0, None] <= x_bounds[1:])  # Shape: (n_points, num_x)
    y_in = np.logical_and(sp[:, 1, None] >= y_bounds[:-1], 
                          sp[:, 1, None] <= y_bounds[1:])  # Shape: (n_points, num_y)

    # Generate all grid masks (shape: (n_points, num_x * num_y))
    grid_masks = np.logical_and(
        x_in[:, :, None],  # Shape: (n_points, num_x, 1)
        y_in[:, None, :]   # Shape: (n_points, 1, num_y)
    ).reshape(n_points, -1)

    # Filter valid batches and create AnnData objects
    batches = [
        adata[mask].copy() 
        for mask in grid_masks.T  # Transpose to iterate over batches
        if mask.sum() > min_spots
    ]

    return batches

def interval_split(adata: AnnData,
                   num_batch_x: int = 2,
                   num_batch_y: int = 2,
                   min_spots: int = 10) -> List[AnnData]:
    if "spatial" not in adata.obsm or adata.obsm["spatial"].shape[1] != 2:
        raise ValueError("adata.obsm['spatial'] must be 2D spatial coordinates (x, y)")
    
    sp = adata.obsm["spatial"].astype(np.float32)  # unify type to reduce computation cost
    n_spots = adata.n_obs
    num_x, num_y = max(1, num_batch_x), max(1, num_batch_y)  # auto-correct invalid block numbers
    
    if num_x * num_y > n_spots:
        raise ValueError(f"Total number of blocks ({num_x}×{num_y}) exceeds total number of spots ({n_spots}), cannot generate valid sub-batches")

    group_x = np.argsort(sp[:, 0]).argsort() % num_x  
    group_y = np.argsort(sp[:, 1]).argsort() % num_y

    x_groups, y_groups = np.meshgrid(range(num_x), range(num_y), indexing="ij")
    batch_masks = (group_x[:, None, None] == x_groups) & (group_y[:, None, None] == y_groups)
    batches = [
        adata[mask].copy()
        for mask in batch_masks.reshape(-1, n_spots)
        if mask.sum() >= min_spots
    ]
    
    return batches

def plot_batch_stats(batches: List[AnnData], figsize: tuple = (3, 5), title: str = None) -> None:
    """
    Plot the statistics of the number of spots per batch.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if not batches:
        warnings.warn("No batches to plot")
        return
    
    # Extract the number of spots in each sub-batch
    plot_df = pd.DataFrame(
        [batch.shape[0] for batch in batches],
        columns=['#spots per batch']
    )
    
    # Plot
    f, ax = plt.subplots(figsize=figsize)
    sns.boxplot(y='#spots per batch', data=plot_df, ax=ax)
    sns.stripplot(y='#spots per batch', data=plot_df, ax=ax, color='red', size=5)
    
    # Set title
    if title is None:
        title = f'Total batches: {len(batches)}'
    plt.title(title)
    plt.savefig('./batch_stats.png')
    plt.show()


# Refactored original batch processing function based on the above functions
def Batch_Data(adata: AnnData, num_batch_x: int, num_batch_y: int, plot_Stats: bool = False) -> List[AnnData]:
    """Split a single dataset into sub-batches by spatial coordinates (original functionality)"""
    batches = batch_split(adata, method='percentile', num_batch_x=num_batch_x, num_batch_y=num_batch_y)
    if plot_Stats:
        plot_batch_stats(batches)
    return batches


def Batch_Data_By_Slice(adata: AnnData, batch_name: str, 
                        num_batch_x: int, num_batch_y: int, 
                        plot_Stats: bool = False,
                        spatial_net_args = {}
                    ) -> List[AnnData]:
    """Split by slice and combine sub-batches (ensure each batch contains data from all slices)"""
    all_slices = adata.obs[batch_name].unique()
    if len(all_slices) == 0:
        raise ValueError(f"Column {batch_name} not found in adata.obs")
    
    # Split each slice separately
    slice_batches = []
    for slice_name in all_slices:
        sub_adata = adata[adata.obs[batch_name] == slice_name].copy()
        if sub_adata.shape[0] == 0:
            warnings.warn(f"Slice {slice_name} has no data, skipped")
            continue

        # slice_subbatches = batch_split(sub_adata, method='percentile', num_batch_x=num_batch_x, num_batch_y=num_batch_y)
        slice_subbatches = batch_split(sub_adata, method='interval', 
                                       num_batch_x=num_batch_x, num_batch_y=num_batch_y)
        slice_batches.append(slice_subbatches)
    
    # Merge corresponding sub-batches
    n_blocks = num_batch_x * num_batch_y
    res_batches = []
    res_names = []
    for i in range(n_blocks):
        # Collect the i-th sub-batch from each slice
        to_merge = []
        for sb in slice_batches:
            if i < len(sb): 
                to_merge.append(sb[i])
        if to_merge:
            adj_list = []
            merged = to_merge[0][0:0].copy()
            for b in to_merge:
                Cal_Spatial_Net(b, **spatial_net_args)
                merged = merged.concatenate(b, join='outer') 
                adj_list.append(b.uns['adj'])
            merged.uns['edgeList'] = adj_concat(adj_list)
            res_batches.append(merged)
            res_names.append(merged.obs[batch_name].tolist())
    if plot_Stats:
        plot_batch_stats(res_batches)
    return res_batches, res_names


def adj_concat(adj_list):
    edge_list = []
    offset = 0
    for adj in adj_list:
        rows, cols = adj
        global_rows = rows + offset
        global_cols = cols + offset
        batch_edges = np.stack([global_rows, global_cols], axis=0)
        edge_list.append(batch_edges)
        offset += max(np.max(rows), np.max(cols)) + 1
    edge_list = np.concatenate(edge_list, axis=1)
    return edge_list


def Cal_Spatial_Net(
    adata: AnnData,
    rad_cutoff: float = None,
    k_cutoff: int = None,
    max_neigh: int = 50,
    model: str = 'Radius',
    verbose: bool = True
):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. 
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['adj']
    """
    # 1. Input validation
    assert model in ['Radius', 'KNN'], f"model must be 'Radius' or 'KNN', but got {model}"
    if model == 'KNN' and k_cutoff is None:
        raise ValueError("k_cutoff (number of neighbors) must be specified for KNN mode")
    if model == 'Radius' and rad_cutoff is None:
        raise ValueError("rad_cutoff (distance threshold) must be specified for Radius mode")
    if 'spatial' not in adata.obsm:
        raise ValueError("Spatial coordinates not found! Please ensure adata.obsm['spatial'] exists")

    if verbose:
        print('------Calculating spatial graph (PyG backend)...')

    # 2. Extract spatial coordinates
    coor = pd.DataFrame(
        data=adata.obsm['spatial'],
        index=adata.obs.index,
        columns=['imagerow', 'imagecol']
    )
    n_cells = coor.shape[0]
    coor_tensor = torch.tensor(coor.values, dtype=torch.float32)

    # 3. Neighbor search using PyG
    if model == 'KNN':
        edge_index = knn_graph(
            x=coor_tensor,
            k=k_cutoff,
            loop=False,
            flow='source_to_target'
        )
    elif model == 'Radius':
        edge_index = radius_graph(
            x=coor_tensor,
            r=rad_cutoff,
            loop=False,
            max_num_neighbors=max_neigh,
            flow='source_to_target'
        )
    else:
        raise ValueError(f"Invalid model: {model}")

    # 4. Directly construct adjacency matrix indices
    edge_index_np = edge_index.numpy()
    self_loops = np.arange(n_cells)
    
    # Combine original edges and self-loops
    all_src = np.concatenate([edge_index_np[0], self_loops])
    all_tgt = np.concatenate([edge_index_np[1], self_loops])
    
    adata.uns['adj'] = (all_src, all_tgt)

    # 5. Logging
    if verbose:
        print(f'The graph contains {edge_index_np.shape[1]} edges, {n_cells} cells.')
        print('%.3f neighbors per cell on average.' % (edge_index_np.shape[1] / n_cells))

    return adata


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STIntg', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df

