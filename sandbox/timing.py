import time

import torch
import torch.nn as nn






def new_attention_forward(
            _self: nn.MultiheadAttention,
            query: torch.Tensor,
            key: torch.Tensor, 
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            assert key is query and value is query, "Only implemented for k_x and v_x as None"
            mask_dict: Dict[str, torch.Tensor] = self._cache.get("mask_dict", {})
            
            sample_input = query
            query, key, value = einops.rearrange(
                Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias),
                "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads,
            )
            
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix            
            if self.use_layer_input:
                sample_input: torch.Tensor = self._cache.pop("layer_input")
               
            bsz = query.shape[0]
            head_dim = query.shape[-1]
            invsqrt_d = head_dim ** -0.5
            
            bsz_index = torch.arange(bsz)[:, None]
            def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
                return t[bsz_index, :, sample_indices, :].transpose(dim0=1, dim1=2)

            def mean(t: torch.Tensor, cluster_indices: torch.Tensor, num_centers: int) -> torch.Tensor:
                cluster_mask = (cluster_indices[..., None] == torch.arange(num_centers))        # bool: [bsz x n x num_centers]
                cluster_sums = torch.sum(einops.rearrange(
                    t, "bsz ... n d -> ... bsz n 1 d"
                ) * cluster_mask[..., None], dim=-3)                                            # float: [... x bsz x num_centers x d]
                cluster_counts = torch.sum(cluster_mask, dim=1)                                 # int: [bsz x num_centers]
                return einops.rearrange(cluster_sums / cluster_counts[..., None], "... bsz s d -> bsz ... s d")
                
            restricted_samples: int = self.num_sample - 1
            if self.mode in ["fps", "uniform", "multiclass_spectral_clustering"]:
                
                sample_indices = self._cache.get("sample_indices", None)
                if self.resample or sample_indices is None:
                    if self.mode == "fps": 
                        sample_indices = sample_farthest_points(sample_input[:, 1:], K=restricted_samples)[1] + 1                   # int: [bsz x max_restricted_samples]
     
                    elif self.mode == "uniform":
                        sample_indices = torch.topk(torch.rand((bsz, ImageFeatures.N)), k=restricted_samples, dim=1).indices + 1    # int: [bsz x max_restricted_samples]

                    elif self.mode == "multiclass_spectral_clustering":
                        NC = OpenCLIPNystromCompressionViT.supply_ncut(restricted_samples)
                        AA = AxisAlign(sort_method="marginal_norm")

                        ncut_features = NC.fit_transform(sample_input[:, 1:, :])                            # float: [bsz x N x num_sample]
                        axis_aligned_features = AA.fit_transform(ncut_features, normalize=True, hard=False) # float: [bsz x N x num_sample]
                        sample_indices = torch.argmax(axis_aligned_features, dim=1) + 1
            
                    else:
                        raise ValueError(self.mode)

                    sample_indices = torch.cat((torch.full((bsz, 1), 0), sample_indices), dim=1)                    
                    self.update_cache({"sample_indices": sample_indices})
                    
                qp, kp = index(query, sample_indices), index(key, sample_indices)

            elif self.mode in ["kmeans", "segment_means", "spectral_clustering"]:
                
                cluster_indices = torch.full((bsz, ImageFeatures.N + 1), -1)                        # int: [bsz x N]
                if self.mode in ["kmeans", "spectral_clustering"]:
                    if self.mode == "spectral_clustering":
                        NC = OpenCLIPNystromCompressionViT.supply_ncut(self.num_sample)
                        restricted_sample_input = NC.fit_transform(sample_input[:, 1:, :])
                    else:
                        restricted_sample_input = sample_input[:, 1:, :]
                
                    # OPTION: Using cuml
                    from cuml import KMeans
                    KM = KMeans(n_clusters=restricted_samples)
                    for image_idx in range(bsz):
                        cluster_indices[image_idx, 1:] = torch.tensor(KM.fit_predict(restricted_sample_input[image_idx]), dtype=torch.int64)
                         
                    qp = torch.cat((mean(query, cluster_indices, restricted_samples), query[:, :, 0:1, :]), dim=2)  # float: [bsz x h x num_sample x d]
                    kp = torch.cat((mean(key, cluster_indices, restricted_samples), key[:, :, 0:1, :]), dim=2)      # float: [bsz x h x num_sample x d]
                
                elif self.mode == "segment_means":
                    cluster_indices[:, 1:] = torch.arange(ImageFeatures.N, dtype=torch.int64) // (ImageFeatures.N // self.num_sample)
                    qp, kp = mean(query, cluster_indices, self.num_sample), mean(key, cluster_indices, self.num_sample)
                
                else:
                    raise ValueError(self.mode)
                
            else:
                raise ValueError(self.mode)
                
            A = torch.softmax(invsqrt_d * (qp @ kp.mT), dim=-1)                                     # float: [bsz x h x num_sample x num_sample]
            BT = torch.softmax(query @ (invsqrt_d * kp.mT), dim=-1)                                 # float: [bsz x h x N x num_sample]
            BV = Fn.scaled_dot_product_attention(qp, key, value)                                    # float: [bsz x h x num_sample x d]
            
            # AinvBV = OpenCLIPNystromCompressionViT.invert(A) @ BV                          # float: [bsz x h x num_sample x d]
            # out_proj_weight = einops.rearrange(_self.out_proj.weight, "D (h d) -> h d D", h=_self.num_heads)    # float: [h x d x D]
            # AinvBVout = AinvBV @ out_proj_weight[None]                                              # float: [bsz x h x num_sample x D]
            # x = torch.sum(BT @ AinvBVout, dim=1) + _self.out_proj.bias                              # float: [bsz x N x D]
            
            x = BT @ (OpenCLIPNystromCompressionViT.invert(A) @ BV)                        # float: [bsz x h x N x d]
            x = einops.rearrange(x, "b h n d -> b n (h d)")
            x = _self.out_proj(x)
            
            for k in self.attention_returns:
                _self.get_submodule(OpenCLIPViT.return_module_name(k))(locals()[k])
            return x,







if __name__ == "__main__":
    n_trials = 1000
    
    start_t = time.perf_counter_ns()
    for _ in range(n_trials):
        p = nn.Parameter(torch.randn((100000, 100)))
        t = torch.sum(torch.sum(p, dim=1))
        torch.autograd.grad(t, p)
        
        torch.cuda.empty_cache()
    end_t = time.perf_counter_ns()
    print(f"sum: {1e-6 * (end_t - start_t) / n_trials}ms")
    
    start_t = time.perf_counter_ns()
    for _ in range(n_trials):
        p = nn.Parameter(torch.randn((100000, 100)))
        t = torch.sum(torch.logsumexp(p, dim=1))
        torch.autograd.grad(t, p)
        
        torch.cuda.empty_cache()
    end_t = time.perf_counter_ns()
    print(f"logsumexp: {1e-6 * (end_t - start_t) / n_trials}ms")
    
    
    




