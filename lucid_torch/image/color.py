import torch

color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02],
                                           [0.27, 0.00, -0.05],
                                           [0.27, -0.09, 0.03]])

C = color_correlation_svd_sqrt
max_norm_svd_sqrt = torch.norm(C, dim=0).max()
cc_norm = C / max_norm_svd_sqrt


def linear_decorrelate(img):
    '''multiply input by sqrt of empirical ImageNet color correlation matrix'''
    if img.shape[1] == 1:
        decorrelated = torch.stack([img] * 3, 1)
    elif img.shape[1] == 3:
        decorrelated = img
    elif img.shape[1] == 4:
        decorrelated = img[:, :-1]
    else:
        raise NotImplementedError()

    decorrelated = decorrelated.permute(0, 2, 3, 1)
    decorrelated = (decorrelated.reshape(-1, 3) @ cc_norm.t().to(img.device)).view(*decorrelated.shape)
    decorrelated = decorrelated.permute(0, 3, 1, 2)

    if img.shape[1] == 1:
        return decorrelated[:, :1]
    elif img.shape[1] == 3:
        return decorrelated
    elif img.shape[1] == 4:
        img = img.clone()
        img[:, :-1] = decorrelated
        return img
    else:
        raise NotImplementedError()
