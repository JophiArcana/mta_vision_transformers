
import torch



def qk_intersection(
    Qw: torch.Tensor,           # [n_heads x head_dim x embed_dim]
    Kw: torch.Tensor,           # [n_heads x head_dim x embed_dim]
    Qb: torch.Tensor = None,    # [n_heads x head_dim]
    Kb: torch.Tensor = None,    # [n_heads x head_dim]
    eps: float = 1e-9,
    inf: float = 1e+9,
) -> torch.Tensor:                                      # [n_heads x embed_dim x d?]
    d, D = Qw.shape[-2:]
    assert d < D, f"Head dimension should be smaller than embedding dimension but got {d} >= {D}."
    
    rank = d
    if Qb is not None and Kb is not None:
        Q = torch.cat((Qw, Qb[..., None]), dim=-1)      # [n_heads x head_dim x (D + 1)]
        K = torch.cat((Kw, Kb[..., None]), dim=-1)      # [n_heads x head_dim x (D + 1)]
    else:
        Q, K = Qw, Kw                                   # [n_heads x head_dim x D]

    M = torch.linalg.pinv(K) @ Q                        # [n_heads x D? x D?]
    L0, V0 = torch.linalg.eig(M)                        # [n_heads x D?], [n_heads x D? x D?]
    L, V = L0[..., :rank], V0[..., :rank]               # [n_heads x d], [n_heads x D? x d]
    
    mask = torch.abs(L.imag) < eps
    L, V = L.real * mask, V.real
    
    if Qb is not None and Kb is not None:
        V = V[..., :-1, :] / V[..., -1:, :]             # [n_heads x D x d?]
    return L[..., None, :] * V                          # [n_heads x D x d?]


def qk_projection_variance(
    X: torch.Tensor,            # [... x N x embed_dim]
    qk: torch.Tensor,           # [n_heads x embed_dim x d?]
    p: float,
):
    X = X[..., None, :, :]                              # [... x 1 x N x embed_dim]
    proj = qk @ torch.linalg.pinv(qk)                   # [n_heads x embed_dim x embed_dim]
    X_proj = X @ proj                                   # [... x n_heads x N x embed_dim]
    return (torch.norm(X_proj, p=2, dim=-1) ** 2) / (torch.norm(X, p=2, dim=-1) ** p)   # [... x n_heads x N]
    
    
if __name__ == "__main__":
    Q = torch.randn((1, 3, 5))
    K = torch.randn((1, 3, 5))
    print(qk_intersection(Q, K))

    



