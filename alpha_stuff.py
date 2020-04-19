#############################################################
######## START ALPHA ########################################
#############################################################


def to_valid_rgb(img, pre_correlation, post_correlation, decorrelate=False, train=True):
    if len(img) == 2:
        img, alpha = img
        is_alpha = True
    else:
        img = img[0]
        is_alpha = False
    img = pre_correlation(img)
    if decorrelate:
        img = linear_decorrelate(img)
    img = post_correlation(img)

    if is_alpha:
        alpha = post_correlation(alpha)
        if train:
            #background = torch.rand_like(img)
            b, ch, h, w = img.shape
            background = get_bg_img(
                (b, ch, h, w), sd=0.2, decay_power=1.5).to(img.device)
        else:
            background = torch.zeros_like(img)
            #background = torch.ones_like(img)
        img = img * alpha + background * (1 - alpha)
    return img


def get_bg_img(shape, sd=0.2, decay_power=1.5, decorrelate=True):
    b, ch, h, w = shape
    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = torch.normal(0, sd, (3, fh, fw, 2))
        scale = np.sqrt(h*w) / np.maximum(freqs, 1./max(h, w))**decay_power

        scaled_spectrum = spectrum_var * \
            torch.from_numpy(scale.reshape(1, *scale.shape, 1)).float()
        img = torch.irfft(scaled_spectrum, signal_ndim=2)
        img = img[:ch, :h, :w]
        imgs.append(img)

    # 4 for desaturation / better scale...
    imgs = torch.stack(imgs) / 4
    if decorrelate:
        imgs = linear_decorrelate(imgs)
    return imgs.sigmoid()


def get_image(size, std, fft=False, dev='cuda:0', seed=124, decay_power=1., alpha=False):

    if fft:
        b, ch, h, w = size
        freqs = rfft2d_freqs(h, w)
        freq_size = (b, ch, *freqs.shape, 2)
        img = torch.normal(0, std, freq_size).to(dev).requires_grad_()

        scale = (torch.tensor(np.sqrt(w*h) /
                              np.maximum(freqs, 1./max(w, h)) ** decay_power))
        scale = scale.view(1, 1, *scale.shape, 1).float().to(dev)

        def pre_correlation(img):
            scaled_img = scale * img
            rgb_img = torch.irfft(scaled_img, signal_ndim=2)
            rgb_img = rgb_img[:b, :ch, :h, :w]
            # TODO what? why? maybe approx the same scale as for pixel images?
            rgb_img.div_(4)
            return rgb_img

        def post_correlation(img):
            return torch.sigmoid(img)

    else:
        img = torch.normal(0, std, size).to(dev).requires_grad_()
        def pre_correlation(x): return x

        def post_correlation(img):
            return torch.sigmoid(img)
    if alpha:
        alpha_shape = (size[0], 1, *size[2:])
        img_alpha = torch.normal(0, std, size).to(dev).requires_grad_()
        return [img, img_alpha], pre_correlation, post_correlation
    else:
        return [img], pre_correlation, post_correlation


class ImageObjective(Objective):
    def __init__(self):
        super().__init__(None)

    def register(self, model):
        pass

    def remove_hook(self):
        pass


class AlphaObjective(ImageObjective):
    def __init__(self, alpha_channels):
        super().__init__()
        self.alpha_channels = alpha_channels

    def _compute_loss(self):
        return 1-self.alpha_channels.sigmoid().mean([-1, -2, -3])
#############################################################
######## END ALPHA ##########################################
#############################################################
