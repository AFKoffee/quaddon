import torch

# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/quant_utils.py

def get_minq_maxq(bits: int, sym: bool):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = torch.tensor(-maxq -1)
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)

    return minq, maxq

def get_asym_quant_params(x, maxq):
    shape = x.shape
    # Per channel quantization
    x = x.flatten(1)

    tmp = torch.zeros(x.shape[0])
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin).clamp(min=1e-5) / maxq
    zero = torch.round(-xmin / scale)

    shape = [-1] + [1] * (len(shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    return scale, zero

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def flexible_quant(x, bits=4):
    _, maxq = get_minq_maxq(bits, False)
    s, z = get_asym_quant_params(x, maxq)
    return asym_quant(x, s, z, maxq)

def flexible_dequant(q, scale, zero):
    return asym_dequant(q, scale, zero)

def flexible_quant_dequant(x, bits=4):
    return flexible_dequant(*flexible_quant(x, bits))
