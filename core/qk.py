import einops
import torch


def qk_intersection(
    Qw: torch.Tensor,   # [n_heads x head_dim x D]
    Kw: torch.Tensor,   # [n_heads x head_dim x D]
    Qb: torch.Tensor,   # [n_heads x head_dim]
    Kb: torch.Tensor,   # [n_heads x head_dim]
    eps: float = 1e-6,
) -> torch.Tensor:                              # [n_heads x D x d?]
    N, d, D = Qw.shape
    assert d < D, f"Head dimension should be smaller than embedding dimension but got {d} >= {D}."
    
    rank = d
    Q = torch.cat((Qw, Qb[..., None]), dim=-1)  # [n_heads x head_dim x (D + 1)]
    K = torch.cat((Kw, Kb[..., None]), dim=-1)  # [n_heads x head_dim x (D + 1)]

    M = torch.linalg.pinv(K) @ Q                # [n_heads x D? x D?]
    L, V = torch.linalg.eig(M)                  # [n_heads x D?], [n_heads x D? x D?]
    
    indices = torch.topk(torch.abs(L), k=rank, dim=1).indices
    L = torch.take_along_dim(L, indices, dim=-1)                # [n_heads x d]
    V = torch.take_along_dim(V, indices[:, None, :], dim=-1)    # [n_heads x D? x d]
    
    mask = torch.abs(L.imag) < eps
    L, V = L.real * mask, V.real
    
    V = V[..., :D, :] / V[..., D:, :]           # [n_heads x D x d?]
    return L[..., None, :] * V                  # [n_heads x D x d?]


def qk_projection_variance(
    X: torch.Tensor,            # [... x N x embed_dim]
    qk: torch.Tensor,           # [n_heads x embed_dim x d?]
    p: float,
    joint: bool,
) -> torch.Tensor:              # [... x n_heads x N] or [... x N]
    if joint:
        qk = einops.rearrange(qk, "h d r -> d (h r)")       # [embed_dim x (n_heads * d?)]
        proj = qk @ torch.linalg.pinv(qk)                   # [embed_dim x embed_dim]
    else:
        X = X[..., None, :, :]                              # [... x 1 x N x embed_dim]
        proj = qk @ torch.linalg.pinv(qk)                   # [n_heads x embed_dim x embed_dim]
    X_proj = X @ proj                                       # [... x n_heads x N x embed_dim] or [... x N x embed_dim]
    return (torch.norm(X_proj, p=2, dim=-1) ** 2) / (torch.norm(X, p=2, dim=-1) ** p)   # [... x n_heads x N] or [... x N]
    
    
if __name__ == "__main__":
    Q = torch.randn((1, 3, 5))
    K = torch.randn((1, 3, 5))
    print(qk_intersection(Q, K))

    



