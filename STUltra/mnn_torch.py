import torch
from torch_cluster import knn
import itertools



def create_dictionary_mnn(use_rep, batch_idx, k=50, iter_comb=None, verbose=0):
    """
    Create a dictionary of mutual nearest neighbors between batches, 
    and record the batch information of each anchor.
    
    Args:
        use_rep: Feature representation
        batch_idx: Dictionary, key is batch name, value is the index array corresponding to that batch
        k: Number of neighbors
        iter_comb: Batch combinations
        verbose: Whether to print detailed information
    
    Returns:
        mnns: Dictionary of MNN pairs (same as before)
        anchor_to_batch: Dictionary mapping each anchor to its batch name
    """
    batch_names = list(batch_idx.keys())
    mnns = {}
    anchor_to_batch = {}  # Record the batch of each anchor
    iter_comb = iter_comb or list(itertools.combinations(range(len(batch_names)), 2))

    for i, j in iter_comb:
        if verbose > 0:
            print(f'Processing datasets {batch_names[i]} and {batch_names[j]}')

        key = f"{batch_names[i]}_{batch_names[j]}"
        
        # Get the local index of two batches (already on device)
        cells_i = torch.tensor(batch_idx[batch_names[i]], device=use_rep.device)
        cells_j = torch.tensor(batch_idx[batch_names[j]], device=use_rep.device)
        
        # Calculate mutual nearest neighbors (MNN)
        idx_i, idx_j = mnn(use_rep[cells_j], use_rep[cells_i], k=k)  # Assume mnn function returns indices
        
        # Map back to global index (index in batch_pair)
        idx_i_global = cells_i[idx_i]  # Anchor belongs to batch_names[i]
        idx_j_global = cells_j[idx_j]  # Anchor belongs to batch_names[j]
        
        # Record the batch of each anchor (key new logic)
        for anchor in idx_i_global.tolist():
            anchor_to_batch[anchor] = batch_names[i]
        for anchor in idx_j_global.tolist():
            anchor_to_batch[anchor] = batch_names[j]
        
        # Merge bidirectional MNN pairs (same as original logic)
        all_anchors = torch.cat([idx_i_global, idx_j_global])
        all_positives = torch.cat([idx_j_global, idx_i_global])
        
        # Remove duplicates and group
        unique_anchors, inverse = torch.unique(all_anchors, return_inverse=True)
        counts = torch.bincount(inverse)
        sorted_indices = torch.argsort(inverse)
        grouped = torch.split(all_positives[sorted_indices], counts.tolist())
        
        mnns[key] = dict(zip(unique_anchors.tolist(), grouped))
    
    return mnns, anchor_to_batch  # Return MNN dictionary and anchor-batch mapping


def generate_triplets(mnn_dict, anchor_to_batch, batch_idx, neg_k=10):
    # Preallocate tensor sizes
    total_pairs = sum(len(pairs) for pairs in mnn_dict.values())
    anchor_idx = torch.empty(total_pairs, dtype=torch.long)
    positive_idx = torch.empty(total_pairs, dtype=torch.long)
    negative_idxs = torch.empty((total_pairs, neg_k), dtype=torch.long)

    start_idx = 0

    for key, mnn_pairs in mnn_dict.items():
        end_idx = start_idx + len(mnn_pairs)
        
        # Extract anchor and positive
        anchors = torch.tensor(list(mnn_pairs.keys()), dtype=torch.long)
        positives = torch.tensor([vals[0] for vals in mnn_pairs.values()], dtype=torch.long)
        
        anchor_idx[start_idx:end_idx] = anchors
        positive_idx[start_idx:end_idx] = positives
        
        # Get batch from anchor_to_batch
        anchor_batches = [anchor_to_batch[anchor.item()] for anchor in anchors]
        
        # Sample negative samples
        negatives = torch.stack([
            batch_idx[batch][torch.randint(len(batch_idx[batch]), (neg_k,))]
            for batch in anchor_batches
        ], dim=0)
        negative_idxs[start_idx:end_idx] = negatives
        
        start_idx = end_idx

    return anchor_idx, positive_idx, negative_idxs

def mnn(ds1, ds2, k=20):
    """
    For each point in ds1, find its k nearest neighbors in ds2 using torch_cluster.knn.
    This matches the direction of sklearn's NearestNeighbors kneighbors(ds1).
    This function is used to find the mutual nearest neighbors between two datasets.
    """
    edges1 = knn(ds1, ds2, k)
    edges2 = knn(ds2, ds1, k)
    hash1 = (edges1[0] << 32) | edges1[1]
    hash2 = (edges2[1] << 32) | edges2[0]
    mask = torch.isin(hash1, hash2)
    mutual_edges = edges1[:, mask]
    return mutual_edges

