from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import torch
from nystrom_ncut import NystromNCut, KernelNCut, SampleConfig, AxisAlign
from sklearn.base import TransformerMixin
from torch_pca import PCA

from infrastructure import utils

N_COMPONENTS = 100
NUM_SAMPLE = 20000


class ComposeDecomposition(TransformerMixin):
    def __init__(self, decompositions: List[TransformerMixin]):
        self.decompositions = decompositions
        self.is_fitted: bool = False

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            utils.reset_seed()
            if getattr(decomposition, "is_fitted", False):
                X = decomposition.transform(X)
            else:
                X = decomposition.fit_transform(X)
        self.is_fitted = True
        return X

    def fit(self, X: torch.Tensor) -> Any:
        self.fit_transform(X)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            X = decomposition.transform(X)
        return X


def generate_NCUT(distance: str = "rbf"):
    if True:
        return KernelNCut(
            n_components=N_COMPONENTS,
            kernel_dim=10000,
            affinity_type=distance,
            sample_config=SampleConfig(
                method="fps",
                num_sample=NUM_SAMPLE,
                fps_dim=12,
            ),
        )
        # return NystromNCut(
        #     n_components=N_COMPONENTS,
        #     affinity_type=distance,
        #     adaptive_scaling=True,
        #     sample_config=SampleConfig(
        #         method="fps",
        #         # method="fps_recursive",
        #         num_sample=NUM_SAMPLE,
        #         fps_dim=12,
        #         # n_iter=1,
        #     ),
        #     eig_solver="svd_lowrank"
        # )
    else:
        from ncut_pytorch import NCUT
        return NCUT(num_eig=N_COMPONENTS, num_sample=NUM_SAMPLE, distance=distance)


DecompositionOptions = Literal["linear", "ncut", "recursive_ncut", "count", "norm", "marginal_norm"]
def supply_decompositions(modes: Set[DecompositionOptions]) -> Dict[str, TransformerMixin]:
    base_nc = generate_NCUT(distance="rbf")
    recursive_nc = generate_NCUT(distance="cosine")
    result = {
        "linear": PCA(n_components=N_COMPONENTS),
        "ncut": base_nc,
        "recursive_ncut": ComposeDecomposition([base_nc, recursive_nc]),
        **{
            k: ComposeDecomposition([
                base_nc, recursive_nc,
                AxisAlign(sort_method=k),
            ]) for k in ("count", "norm", "marginal_norm",)
        },
        "ncut_pca": ComposeDecomposition([
            generate_NCUT(),
            PCA(n_components=N_COMPONENTS),
        ]),
    }
    return {k: v for k, v in result.items() if k in modes}

