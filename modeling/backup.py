# SECTION: Replace resblock.attn.forward
        def get_attention_forward_func_for_layer(idx: int):
            def attention(
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
                compression_condition = idx in self.mask_layers
                
                x = query
                qkv = F.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                query, key, value = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads)
                
                bsz = x.shape[0]
                bsz_index = torch.arange(bsz)[:, None]
                head_dim = query.shape[-1]
                subsample_matrix: torch.Tensor = torch.full((bsz, self.num_sample), -1)
                
                attn_weights = torch.matmul(query, key.mT) / (head_dim ** 0.5)
                attn_matrix = F.softmax(attn_weights, dim=-1)
                
                if compression_condition:
                    empty_mask = torch.full((bsz, ImageFeatures.N + 1,), False)
                    
                    def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
                        return einops.rearrange(t[bsz_index, :, sample_indices, :], "bsz s h d -> bsz h s d")
                    
                    def mean(t: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
                        cluster_mask = (cluster_indices[..., None] == torch.arange(self.num_sample))    # bool: [bsz x n x num_sample]
                        cluster_sums = torch.sum(t[:, None, :, None, :] * cluster_mask, dim=2)          # float: [bsz x h x d x num_sample]
                        cluster_counts = torch.sum(cluster_mask, dim=1)[:, None, :, None]               # int: [bsz x 1 x num_sample x 1]
                        return einops.rearrange(cluster_sums, "bsz h d s -> bsz h s d") / cluster_counts
                        
                    def get_propagation_query_and_key(
                        implicit_mask: torch.Tensor,            # [bsz x N]
                    ) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
                        
                        nonlocal subsample_matrix
                        
                        if self.mode in ["fps", "uniform"]:
                            guarantee_mask = mask_dict["guarantee"]                                 # bool: [bsz x N]
                            exclude_mask = mask_dict["exclude"]                                     # bool: [bsz x N]
                            
                            restricted_samples = self.num_sample - torch.sum(guarantee_mask, dim=1) # int: [bsz]
                            restricted_mask = implicit_mask * ~guarantee_mask * ~exclude_mask       # bool: [bsz x N]
                            
                            counts = torch.sum(restricted_mask, dim=1)                              # [bsz]
                            max_count: int = torch.max(counts).item()                               # int
                            if self.mode == "fps":
                                max_restricted_samples: int = torch.max(restricted_samples).item()  # int

                                topk_indices = torch.topk(restricted_mask.to(torch.int), k=max_count, dim=1).indices    # int: [bsz x max_count]
                                fps_indices = sample_farthest_points(
                                    x[bsz_index, topk_indices].cpu(),
                                    lengths=counts.cpu(), K=restricted_samples.cpu(),
                                )[1].to(DEVICE)                                                                         # int: [bsz x max_restricted_samples]               
                                
                                sample_indices = torch.cat((
                                    torch.cat((
                                        topk_indices, torch.full((bsz, 1), -1)
                                    ), dim=1)[bsz_index, fps_indices],
                                    torch.full((bsz, self.num_sample - max_restricted_samples), -1)
                                ), dim=1)                                                                               # int: [bsz x num_sample]
                            else:
                                sort_weights = torch.rand(restricted_mask.shape) + (~restricted_mask).to(torch.float)   # float: [bsz x N]
                                sample_indices = torch.argsort(sort_weights, dim=1)[:self.num_sample]                   # int: [bsz x num_sample]
                                sample_indices[torch.arange(self.num_sample) >= restricted_samples] = -1
                            sample_indices[sample_indices == -1] = torch.where(guarantee_mask)[1]
                            
                            subsample_matrix = sample_indices
                            return index(query, sample_indices), index(key, sample_indices)         # float: [bsz x h x num_sample x d]
                        
                        elif self.mode == "manual":
                            manual_mask = mask_dict["manual"]                                       # bool: [bsz x N]
                            return get_implicit_query_and_key(manual_mask)
                        
                        elif self.mode in ["kmeans", "random_mean"]:
                            if self.mode == "kmeans":
                                counts = torch.sum(implicit_mask, dim=1)                            # [bsz]
                                max_count: int = torch.max(counts).item()                           # int
                                                                
                                topk_indices = torch.topk(implicit_mask.to(torch.int), k=max_count, dim=1).indices  # int: [bsz x max_count]
                                KM = KMeans(n_clusters=self.num_sample, verbose=False)
                                km_indices = KM.fit_predict(x[bsz_index, topk_indices], k=counts)                   # int: [bsz x max_count]
                                
                                cluster_indices = torch.zeros((bsz, ImageFeatures.N + 1), dtype=torch.int64)        # int: [bsz x N]
                                cluster_indices[bsz_index, topk_indices] = km_indices                                
                            else:
                                cluster_indices = torch.randint(0, self.num_sample, (bsz, ImageFeatures.N + 1))     # int: [bsz x N]
                            cluster_indices[~implicit_mask] = -1
                            
                            return mean(query, cluster_indices), mean(key, cluster_indices)         # float: [bsz x h x num_sample x d]

                        else:
                            raise ValueError(self.mode)
                    
                    def get_implicit_query_and_key(
                        implicit_mask: torch.Tensor,            # [bsz x N]
                    ) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
                        counts = torch.sum(implicit_mask, dim=1)                                # int: [bsz]
                        if torch.all(counts == counts[0]):
                            sample_indices = torch.where(implicit_mask)[1].view((bsz, -1))      # int: [bsz x num_sample]
                            return index(query, sample_indices), index(key, sample_indices)     # float: [bsz x h x num_sample x d]
                        else:
                            return [
                                (query[image_idx, :, mask, :], key[image_idx, :, mask, :])      # float: bsz x [h x num_sample x d]
                                for image_idx, mask in enumerate(implicit_mask)
                            ]       
                    
                    explicit_mask = mask_dict["explicit"]                   # bool: [bsz x N]
                    implicit_mask = ~explicit_mask                          # bool: [bsz x N]
                
                    masked_attn_weights = torch.where(explicit_mask[:, None, None, :], -torch.inf, attn_weights)                # [bsz x h x N x N]
                    downscale = torch.exp(torch.logsumexp(masked_attn_weights, dim=-1) - torch.logsumexp(attn_weights, dim=-1)) # [bsz x h x N]
                
                    qkp = get_propagation_query_and_key(implicit_mask)
                    qki = get_implicit_query_and_key(implicit_mask)
                    
                    def compute_extension(qp: torch.Tensor, kp: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
                        s = head_dim ** -0.5
                        A = torch.softmax(s * (qp @ kp.mT), dim=-1)         # float: [bsz x h x num_sample x num_sample]
                        B = torch.softmax(s * (qp @ ki.mT), dim=-1)         # float: [bsz x h x num_sample x num_implicit]
                        BT = torch.softmax(s * (qi @ kp.mT), dim=-1)        # float: [bsz x h x num_implicit x num_sample]
                        return BT @ self.invert(A) @ B                      # float: [bsz x h x num_implicit x num_implicit]

                    if isinstance(qkp, tuple) and isinstance(qki, tuple):
                        qp, kp = qkp                                        # float: [bsz x h x num_sample x d]
                        qi, ki = qki                                        # float: [bsz x h x num_implicit x d]
                        
                        implicit_indices = torch.where(implicit_mask)[1].view((bsz, -1))    # int: [bsz x num_implicit]
                        implicit_downscale = einops.rearrange(downscale[bsz_index, :, implicit_indices], "bsz i h -> bsz h i")  # float: [bsz x h x num_implicit]
                        
                        C = compute_extension(qp, kp, qi, ki)               # float: [bsz x h x num_implicit x num_implicit]
                        C = C * implicit_downscale[..., None]
                        attn_matrix[torch.arange(bsz)[:, None, None], :, implicit_indices[:, :, None], implicit_indices[:, None, :]] = einops.rearrange(C, "bsz h ni1 ni2 -> bsz ni1 ni2 h")
                    
                    else:
                        if isinstance(qkp, tuple):
                            qkp = [*zip(*qkp)]
                        if isinstance(qki, tuple):
                            qki = [*zip(*qki)]

                        for image_idx, ((qp, kp), (qi, ki)) in enumerate(zip(qkp, qki)):
                            implicit_indices = torch.where(implicit_mask[image_idx])[0]     # int: [num_implicit]
                            implicit_downscale = downscale[image_idx, :, implicit_indices]  # float: [h x num_implicit]
                            
                            C = compute_extension(qp, kp, qi, ki)               # float: [h x num_implicit x num_implicit]
                            C = C * implicit_downscale[..., None]
                            attn_matrix[image_idx, :, implicit_indices[:, None], implicit_indices[None, :]] = C
                
                x = einops.rearrange(attn_matrix @ value, "b h n d -> b n (h d)")                
                x = F.linear(x, _self.out_proj.weight, _self.out_proj.bias)
                    
                for k in self.attention_returns:
                    _self.get_submodule(OpenCLIPNystromCompressionViT.return_module_name(k))(locals()[k])
                return x,
            return attention