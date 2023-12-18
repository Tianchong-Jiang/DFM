import torch
# from torch_efficient_distloss import (
#     eff_distloss,
#     eff_distloss_native,
#     flatten_eff_distloss,
# )
from einops import rearrange, repeat


def mse_loss(img, gt):
    img = img.view(gt.shape)
    return ((img - gt) ** 2).mean()

def eff_distloss(w, m, interval):
    '''
    Efficient O(N) realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    '''
    loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()
    wm = (w * m)
    w_cumsum = w.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni

# def distortion_loss(weights, z_vals, near, far):
#     # loss from mip-nerf 360; efficient implementation from DVGOv2 (https://github.com/sunset1995/torch_efficient_distloss) with some modifications

#     # weights: [B, N, n_samples, 1]
#     # z_vals: [B, N, n_samples, 1]

#     assert weights.shape == z_vals.shape
#     assert len(weights.shape) == 4
#     weights = rearrange(weights, "b n s 1 -> (b n) s")
#     z_vals = rearrange(z_vals, "b n s 1 -> (b n) s")

#     # go from z space to s space (for linear sampling; INVERSE SAMPLING NOT IMPLEMENTED)
#     s = (z_vals - near) / (far - near)

#     # distance between samples
#     interval = s[:, 1:] - s[:, :-1]

#     loss = eff_distloss(weights[:, :-1], s[:, :-1], interval)
#     return loss


def occupancy_loss(weights):
    # loss from lolnerf (prior on weights to be distributed as a mixture of Laplacian distributions around mode 0 or 1)
    # weights: [B, N, n_samples, 1]
    assert len(weights.shape) == 4

    pw = torch.exp(-torch.abs(weights)) + torch.exp(
        -torch.abs(torch.ones_like(weights) - weights)
    )
    loss = -1.0 * torch.log(pw).mean()
    return loss
