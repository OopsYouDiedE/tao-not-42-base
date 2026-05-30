import torch

def compute_sinkhorn_matching(cost, epsilon=0.1, iters=10):
    """
    GPU 原生 Sinkhorn 算法求软分配概率矩阵 (Zero CPU-GPU Sync)
    
    Args:
        cost: 代价矩阵 [N, M]
        epsilon: 正则化参数
        iters: 迭代次数
        
    Returns:
        P_opt: 软分配概率矩阵 [N, M]
    """
    P = torch.exp(-cost / epsilon)
    u = torch.ones_like(P[:, 0])
    v = torch.ones_like(P[0, :])
    for _ in range(iters):
        u = 1.0 / (torch.matmul(P, v) + 1e-8)
        v = 1.0 / (torch.matmul(P.t(), u) + 1e-8)
    P_opt = u.unsqueeze(1) * P * v.unsqueeze(0)
    return P_opt
