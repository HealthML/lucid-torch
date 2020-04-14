# TODO all imports

def channel_attr_binary(img, model, layer_func, target_class=torch.ones((1,1)), dev='cuda:0'):
    # TODO --> this dict thing is really weird... -> need it for sorted in vis1?
    act, grad = compute_layer(
            img,
            model,
            layer_func,
            dev,
            include_grad=True,
            targets=torch.tensor(target_class)
            )
    attribution = act * grad[0]
    channel_attr = attribution.sum([-1,-2])
    argsort = channel_attr.argsort(-1, descending=True)
    return dict( (int(s), float(channel_attr[0, s])) for s in argsort[0] )

def channel_cam(img, model, layer_func, channel, target_class=torch.ones((1,1)), dev='cuda:0'):
    act, grad = compute_layer(
            img,
            model,
            layer_func,
            dev,
            include_grad=True,
            targets=torch.tensor(target_class),
            )
    attribution = act * grad[0]
    channel_attr = attribution[0, channel, :, :].detach().cpu().numpy()
    channel_attr = sk_resize(channel_attr, img.shape[-2:], order=0)
    return channel_attr

# been used
def prepare_layer_cams(img, model, layer_func, target_class=torch.ones((1, 1)), dev='cuda:0'):
    normalize = 2

    b, ch, h, w = compute_layer(img, model, layer_func, dev)[0].shape
    cams = []
    for tc in range(ch):
        cam = channel_cam(img, model, layer_func, tc, target_class, dev)
        if (normalize > 1) and (cam.max() - cam.min() > 1e-12):
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cams.append(cam)
    cams = np.array(cams)
    if normalize == 1:
        cams = (cams - cams.min()) / (cams.max() - cams.min())

    return cams

    



