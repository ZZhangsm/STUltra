import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import random

from .loss import InfoNCE
from .mnn_torch import create_dictionary_mnn, generate_triplets
from .net import Model
from .ST_utils import Batch_Data_By_Slice #, Cal_Spatial_Net, adj_concat

cudnn.deterministic = True
cudnn.benchmark = True

def seed_everything(seed=666):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def return_sparse_data(x, edge_index, 
                       prune_edge_index=[], 
                       batch_name=None,
                    ):
    if hasattr(x, 'toarray'):
        # Process sparse matrix
        x_sparse = x.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor([x_sparse.row, x_sparse.col]),
            torch.FloatTensor(x_sparse.data),
            torch.Size(x_sparse.shape)
        )
    else:
        # Process dense matrix
        x = torch.FloatTensor(x)
    
    data = Data(edge_index=torch.LongTensor(edge_index),
            prune_edge_index=torch.LongTensor(prune_edge_index),
            x=x)
    if batch_name is not None:
        data.batch_name = batch_name
    return data



def train(
        adata,
        hidden_dims=[512, 30], lr=0.001, gradient_clipping=5., weight_decay=0.0001, 
        pretrain_epochs=500, n_epochs=1000, device='cpu', 
        key_added='STUltra', save_recon=False, 
        random_seed=666, 
        num_batch=2, batch_data=False, spatial_net_args={},  
        iter_comb=None, knn_neigh=100, alpha=0.2, 
        verbose=False 
    ):
    """\
    Train graph auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for Optimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for Optimizer.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.
    save_recon
        Whether to save the reconstructed expression matrix.
    num_batch
        The number of batches in the x and y directions.
    batch_data
        Whether to perform batch correction in the embedding space.
    spatial_net_args
        Arguments for Cal_Spatial_Net.
    random_seed
        Random seed for reproducibility.
    pretrain_epochs (default: 500)
        Number of epochs for pretraining the graph auto-encoder.
    n_epochs (default: 10)
        Number of epochs for training the graph auto-encoder.

    Returns
    -------
    adata
        AnnData object of scanpy package.
    """
    seed_everything(random_seed)


    alpha = alpha if alpha is not None else 0.2
    device = device if device is not None else 'cpu'
    
    data = return_sparse_data(adata.X, adata.uns['edgeList'])

    if batch_data:
        # data = data.to('cpu')
        Batch_list, Batch_names = Batch_Data_By_Slice(adata, batch_name='batch_name', 
                                                      num_batch_x=num_batch, num_batch_y=num_batch, 
                                                      spatial_net_args=spatial_net_args)
        data_list = [return_sparse_data(b.X, b.uns['edgeList'], batch_name=Batch_names[i]) for i, b in enumerate(Batch_list)]
        trainloader = DataLoader(data_list, batch_size=1, shuffle=True)
    else:
        
        data = data.to(device)
        batches = adata.obs['batch_name']
        batch_idx = {batch: torch.tensor(np.where(batches == batch)[0], dtype=torch.long) 
                    for batch in batches.unique()}

    input_dim = adata.X.shape[1]
    model = Model(hidden_dims=[input_dim, hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    info_nce_loss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')

    if verbose:
        print(model, end='\n')

    print('Pretrain with Graph Auto-Encoder...')
    pbar = tqdm(range(0, pretrain_epochs))
    model.train()
    for epoch in pbar:
        if batch_data:
            loss_list = []
            for batch in trainloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss, _ = model.train_step(batch.x, batch.edge_index)
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                del batch, loss, _
                torch.cuda.empty_cache()
            pbar.set_postfix({'loss': f'{np.mean(loss_list):.3f}'})

        else:   
            optimizer.zero_grad()
            loss, _ = model.train_step(data.x, data.edge_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

    print('Integrate Multi-slice data with contrastive learning...')
    pbar = tqdm(range(pretrain_epochs, n_epochs))
    for epoch in pbar:
        if batch_data:
            if epoch % 100 == 0 or epoch == pretrain_epochs:
                if verbose:
                    print(f'\nNo. {epoch} epoch update spot triplets')
                model.eval()
                eval_list = []
                for data_tmp in tqdm(data_list, desc=f'Update triplets at epoch {epoch}', leave=False):
                    data_tmp = data_tmp.to(device)
                    with torch.no_grad():
                        enc_rep, _ = model(data_tmp.x, data_tmp.edge_index)
                    
                    batch_names = np.array(data_tmp.batch_name)
                    batch_idx = {batch: torch.tensor(np.where(batch_names == batch)[0], dtype=torch.long) 
                                for batch in batch_names}
                    mnn_dict, anchor_to_batch = create_dictionary_mnn(enc_rep, batch_idx, knn_neigh, iter_comb, verbose=0)
                    anchor_idx, pos_idx, neg_idxs = generate_triplets(mnn_dict, anchor_to_batch, batch_idx, neg_k=10)
                    eval_data = Data(x=data_tmp.x, edge_index=data_tmp.edge_index, 
                                 anchor_ind=anchor_idx, positive_ind=pos_idx, negative_ind=neg_idxs)
                    eval_list.append(eval_data)
                eval_loader = DataLoader(eval_list, batch_size=1, shuffle=True)
                del data_tmp, enc_rep, batch_names, batch_idx, mnn_dict, anchor_to_batch, anchor_idx, pos_idx, neg_idxs, eval_data


            model.train()
            loss_list, mse_loss_list, tri_loss_list = [], [], []
            for batch in eval_loader:
                batch = batch.to(device)
                mse_loss, z = model.train_step(batch.x, batch.edge_index)
                anchor_arr = z[batch.anchor_ind,]
                pos_arr = z[batch.positive_ind,]
                neg_arr = z[batch.negative_ind,]
                tri_loss = info_nce_loss(anchor_arr, pos_arr, neg_arr)
                loss = mse_loss + alpha * tri_loss
                loss_list.append(loss.item())
                mse_loss_list.append(mse_loss.item())
                tri_loss_list.append(tri_loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            pbar.set_postfix({'loss': f'{np.mean(loss_list):.3f}', 
                            'mse_loss': f'{np.mean(mse_loss_list):.3f}', 
                            'tri_loss': f'{np.mean(tri_loss_list):.3f}'})

        else:
            if epoch % 100 == 0 or epoch == pretrain_epochs:
                model.eval()
                with torch.no_grad():
                    enc_rep, _ = model(data.x, data.edge_index)
                model.train()
                if verbose:
                    print(f'\nNo. {epoch} epoch update spot triplets')
                mnn_dict, anchor_to_batch = create_dictionary_mnn(enc_rep, batch_idx, knn_neigh, iter_comb, verbose=0)
                anchor_idx, pos_idx, neg_idxs = generate_triplets(mnn_dict, anchor_to_batch, batch_idx, neg_k=10)

            optimizer.zero_grad()
            mse_loss, enc_rep = model.train_step(data.x, data.edge_index)
            anchor_arr = enc_rep[anchor_idx,]      # [batch_size, feature_dim]
            pos_arr = enc_rep[pos_idx,]            # [batch_size, feature_dim]
            neg_arr = enc_rep[neg_idxs,]           # [batch_size, neg_k, feature_dim]
            tri_loss = info_nce_loss(anchor_arr, pos_arr, neg_arr)
            loss = mse_loss + alpha * tri_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                            'mse_loss': f'{mse_loss.item():.3f}', 
                            'tri_loss': f'{tri_loss.item():.3f}'})

    print('Save the embedding for downstream analysis...')
    model.eval()
    with torch.no_grad():
         # As numpy array saved in adata.obsm[key_added]
        if batch_data:
            model.to('cpu')
        enc_rep, recon_rep = model(data.x, data.edge_index)
        adata.obsm[key_added] = enc_rep.cpu().detach().numpy()
        if save_recon:
            adata.obsm['recon'] = recon_rep.cpu().detach().numpy()
    return adata