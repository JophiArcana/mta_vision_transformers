import torch


def visualize_feature_values_by_token(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    layer_indices: List[int] = None,
) -> None:
    bsz = t.shape[1]
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(7 * len(layer_indices), 5), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = t[layer_idx]
        
        m, s = feature_values.mean().item(), feature_values.std().item()
        left, right = m - 3 * s, m + 3 * s
        bins = torch.arange(left, right, s / 20).tolist()
        
        for image_idx, token_idx in itertools.product(range(bsz), range(-1, H * W)):
            h_idx, w_idx = token_idx // W, token_idx % W
            token_feature_values = feature_values[image_idx, token_idx + 1]
            
            p = torch.rand(()) < 0.3
            if token_idx != -1 and (mta_mask[image_idx, h_idx, w_idx] or p):
                axs[i].hist(
                    token_feature_values.numpy(force=True), bins=bins, histtype="step", density=True,
                    color=rgb_assignment[image_idx, h_idx, w_idx].numpy(force=True), zorder=20
                )
            elif token_idx == -1 and p:
                axs[i].hist(
                    token_feature_values.numpy(force=True), bins=bins, histtype="step", density=True,
                    color="black", zorder=12, label="cls_token" if image_idx == 0 else None
                )
        
        # axs[i].autoscale(False)
        # axs[i].plot([0, 0], [0, top := 1.5], color="black", linestyle="--")
        
        axs[i].set_title(f"Layer {layer_idx}: {metric_name}_values")
        axs[i].set_xlabel("value")
        axs[i].set_xlim(left=left, right=right)
        axs[i].set_ylabel("count")
        # axs[i].set_ylim(top=top)
    
    plt.show()


def visualize_feature_values_by_channel(
    metric_name: str,
    t: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor] = None,
    align_layers: bool = False,
    layer_indices: List[int] = None,
) -> None:
    if fn is None:
        fn = lambda t: torch.norm(t, dim=0)
    if align_layers:
        channel_feature_norms = fn(torch.flatten(t[layer_indices], 0, -2))
        channel_colors = plt.cm.get_cmap("plasma")((channel_feature_norms - channel_feature_norms.min()) / (channel_feature_norms.max() - channel_feature_norms.min()))
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(1 * len(layer_indices), 5 / 7), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = t[layer_idx]
        
        if not align_layers:
            channel_feature_norms = fn(torch.flatten(feature_values, 0, -2))
            channel_colors = plt.cm.get_cmap("plasma")((channel_feature_norms - channel_feature_norms.min()) / (channel_feature_norms.max() - channel_feature_norms.min()))
        
        k = 300
        m, s = feature_values.mean().item(), feature_values.std().item()
        left, right = feature_values.flatten().topk(k=k, largest=False).values[-1], feature_values.flatten().topk(k=k, largest=True).values[-1]
        bins = torch.arange(left, right, s / 20).tolist()
        
        for channel_idx in range(feature_values.shape[2]):
            channel_feature_values = feature_values[:, :, channel_idx]
            
            axs[i].hist(
                channel_feature_values.flatten().numpy(force=True), bins=bins, histtype="step", density=True,
                color=channel_colors[channel_idx], zorder=channel_feature_norms[channel_idx]
            )
        
        axs[i].set_title(f"Layer {layer_idx}: {metric_name}_values")
        axs[i].set_xlabel("value")
        axs[i].set_xlim(left=left, right=right)
        axs[i].set_ylabel("count")
        # axs[i].set_ylim(top=top)
    
    plt.show()


def visualize_feature_norms_by_channel(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    layer_indices: List[int] = None,
) -> None:
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=2, ncols=len(layer_indices), figsize=(2 * len(layer_indices), 10 / 7), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = einops.rearrange(t[layer_idx, :, -H * W:], "bsz (h w) c -> bsz h w c", h=H, w=W)
        demeaned_feature_values = feature_values - torch.mean(feature_values.flatten(0, -2), dim=0)
        
        def plot_channel_norm(ax, values: torch.Tensor, prefix: str):
            ax.plot((values[mta_mask] ** 2).mean(dim=0) ** 0.5, zorder=1, label=f"{prefix}mt_channel_norm")
            ax.plot((values[~mta_mask] ** 2).mean(dim=0) ** 0.5, zorder=0, label=f"{prefix}non_mt_channel_norm")
            
            ax.set_title(f"Layer {layer_idx}: {metric_name}_{prefix}channel_norm")
            ax.legend()
            
        plot_channel_norm(axs[0, i], feature_values, "")
        plot_channel_norm(axs[1, i], demeaned_feature_values, "demeaned_")
    
    plt.show()


def visualize_layer_norm(
    output_dict: Dict[str, torch.Tensor],
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    mean: Literal[None, "local", "global"],
    projection_mode: Literal["pca", "tsne"],
    with_ma: bool = True,
    highlight: torch.Tensor = None,
    subsample: float = 0.5,
    include: Set[str] = None,
    **kwargs: Any,
) -> None:
    if include is None:
        include = output_dict.keys()
        
    fig, axs = plt.subplots(nrows=1, ncols=len(include), figsize=(2 * len(include), 2), subplot_kw={"projection": "3d"})
    for i, metric_name in enumerate(include):
        ax = axs if len(include) == 1 else axs[i]
        
        feature_values = output_dict[metric_name]
        bsz, _, D = feature_values.shape

        if feature_values.shape[-2] == H * W + 1 or torch.any(torch.isnan(feature_values)):
            mask = torch.cat((
                torch.full((bsz, feature_values.shape[-2] - H * W), False),
                torch.flatten(mta_mask, start_dim=1, end_dim=2),
            ), dim=-1)
        else:
            mask = torch.full(feature_values.shape[:-1], False)
            mask[:, 1:-H * W] = True
        inverse_mask = ~torch.any(torch.isnan(feature_values), dim=-1) & ~mask
                
        if projection_mode == "pca":
            # global_feature_mean = 0 if mean is None else torch.mean(feature_values[inverse_mask], dim=0)
            global_feature_mean = 0 if mean is None else torch.mean(feature_values[~torch.any(torch.isnan(feature_values), dim=-1)], dim=0)
            demeaned_feature_values = feature_values - global_feature_mean

            if with_ma:
                V_mta = torch.pca_lowrank(demeaned_feature_values[mask], q=1, center=(mean == "local"))[-1]
                proj = torch.eye(D) - V_mta @ torch.linalg.pinv(V_mta)
                V_nonmta = torch.pca_lowrank(demeaned_feature_values[inverse_mask] @ proj, q=2, center=(mean == "local"))[-1]
                V = torch.cat((V_mta, V_nonmta), dim=-1)
                
                ax.set_xlabel("ma_direction")
            else:
                V = torch.pca_lowrank(demeaned_feature_values[inverse_mask], q=3, center=(mean == "local"))[-1]
            
            compressed_features = demeaned_feature_values @ V
            
        elif projection_mode == "tsne":
            input_mask = inverse_mask | (mask if with_ma else False)
            compressed_features = torch.full((*feature_values.shape[:-1], 3), torch.nan)
            compressed_features[input_mask] = torch.tensor(TSNE(n_components=3).fit_transform(feature_values[input_mask].numpy(force=True)))
        
        def to_rgb_mask(m: torch.Tensor) -> torch.Tensor:
            return m[:, -H * W:].view(bsz, H, W)
        
        subsample_mask = inverse_mask & (torch.arange(-feature_values.shape[-2], 0) >= -H * W) & (torch.rand(mask.shape) < subsample)
        if with_ma:
            ax.scatter(*compressed_features[mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(mask)].numpy(force=True), s=10, **kwargs)
        ax.scatter(*compressed_features[subsample_mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(subsample_mask)].numpy(force=True), s=1, **kwargs)
        ax.scatter(*compressed_features[:, 0].mT.numpy(force=True), s=30, color="black", label="cls_token")
        
        if highlight is not None:
            image_idx, h_idx, w_idx = torch.unbind(highlight, dim=-1)
            highlight_mask = torch.full_like(inverse_mask, False)
            highlight_mask[image_idx, (h_idx * W + w_idx - H * W)] = True

            ax.scatter(*compressed_features[highlight_mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(highlight_mask)].numpy(force=True), s=400, marker="*")
        
        ax.set_title(f"{metric_name}_pca_values")
        ax.legend()

    plt.show()