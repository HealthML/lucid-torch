import kornia
import torch


class TFMSRotate(torch.nn.Module):
    def __init__(self, angle=0, padding_mode='border', interpolation='nearest'):
        super(TFMSRotate, self).__init__()
        self.angle = torch.tensor([angle])
        self.padding_mode = padding_mode
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor):
        b, c, h, w = img.shape
        center = torch.tensor([[w, h]], dtype=torch.float) / 2
        transformation_matrix = kornia.get_rotation_matrix2d(
            center,
            self.angle,
            torch.ones(1))
        transformation_matrix = transformation_matrix.expand(
            b, -1, -1)
        transformation_matrix = transformation_matrix.to(img.device)
        return kornia.warp_affine(
            img.float(),
            transformation_matrix,
            dsize=(h, w),
            flags=self.interpolation,
            padding_mode=self.padding_mode)
