#%%
import sys
sys.path.append("/workspace/mta_vision_transformers/")
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from ncut_pytorch import NCUT
from nystrom_ncut import KernelNCut, NystromNCut, affinity_from_features, SampleConfig, AxisAlign

from infrastructure.settings import DEVICE
from infrastructure import utils


if __name__ == "__main__":
    torch.manual_seed(1212)
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(linewidth=400, sci_mode=False, precision=6)



    n, d = 300000, 1024
    n_components = 100
    sample_method = "fps"
    affinity_type = "rbf"
    shape = ()
    X = torch.randn((*shape, n, d))
    # X = torch.where((torch.rand((*shape, n, 1)) < 0.8).expand((*shape, n, d)), X, torch.nan)

    num_sample = 10000
    
    nc = NCUT(
        num_eig=n_components,
        distance=affinity_type,
        num_sample=num_sample,
        sample_method="farthest",
        eig_solver="svd_lowrank", 
    )  
    nnc = NystromNCut(
        n_components=n_components,
        affinity_type=affinity_type,
        adaptive_scaling=False,
        sample_config=SampleConfig(method=sample_method, num_sample=num_sample, fps_dim=8),
        # sample_config=SampleConfig(method="random", num_sample=num_sample),
        # sample_config=SampleConfig(method="fps_recursive", num_sample=num_sample, n_iter=10),
        eig_solver="svd_lowrank",
    )
    knc = KernelNCut(
        n_components=n_components,
        kernel_dim=1000,
        affinity_type=affinity_type,
        sample_config=SampleConfig(method=sample_method, num_sample=num_sample, fps_dim=8),
    )

    # precomputed_sampled_indices = torch.arange(num_sample).expand((*shape, num_sample))

    n_trials = 1
    start_t = time.perf_counter()
    
    P = nn.Parameter(X)
    optimizer = optim.AdamW((P,))
    
    # for _ in tqdm(range(n_trials)):
    #     V = nc.fit_transform(P)
        
    #     optimizer.zero_grad()
    #     (torch.norm(V) ** 2).backward()
        
    #     utils.empty_cache()
    # end_t = time.perf_counter()
    # print(f"NCUT: {(end_t - start_t) / n_trials}s")
    
    # start_t = time.perf_counter()
    # for _ in tqdm(range(n_trials)):
    #     Vn = nnc.fit_transform(X)
    #     utils.empty_cache()
    # end_t = time.perf_counter()
    # print(f"NystromNCut: {(end_t - start_t) / n_trials}s")

    start_t = time.perf_counter()
    for _ in tqdm(range(n_trials)):
        Vk = knc.fit_transform(P)
        
        optimizer.zero_grad()
        (torch.norm(Vk) ** 2).backward()
        
        utils.empty_cache()
    end_t = time.perf_counter()
    print(f"KernelNCut: {(end_t - start_t) / n_trials}s")


    # print(V)

    # V_ = torch.stack([
    #     nc.fit_transform(X[idx], precomputed_sampled_indices=precomputed_sampled_indices[idx])
    #     for idx in range(shape[0])
    # ], dim=0)
    #
    # aa = AxisAlign(sort_method="count")
    # print(aa.fit_transform(V))





    # X = torch.randn((10, 5))
    # X[1] = torch.nan
    # Y = X[~torch.any(torch.isnan(X), dim=1)]
    #
    #
    # Z = torch.randn((7, 5))
    # Z[torch.randn((len(Z),)) < 0] = torch.nan
    #
    # torch.manual_seed(2002)
    # print(nc.fit_transform(torch.cat((X, Z), dim=0)))   # , precomputed_sampled_indices=torch.arange(len(X))))
    # torch.manual_seed(2002)
    # print(nc.fit_transform(torch.cat((Y, Z), dim=0)))   # , precomputed_sampled_indices=torch.arange(len(Y))))


# %%
