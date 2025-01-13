import torch

def get_minq_maxq(bits: int, sym: bool):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = torch.tensor(-maxq -1)
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)

    return minq, maxq

def get_asym_quant_params(x, maxq):
    zero = torch.mean(x, dim=-1)
    scale = torch.max(torch.abs(x - zero), dim=-1)[0].unsqueeze(1)/maxq
    return scale, zero

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def four_bit_quant(x):
    _, maxq = get_minq_maxq(4, False)
    s, z = get_asym_quant_params(x, maxq)
    return asym_quant(x, s, z, maxq)

def four_bit_dequant(q, scale, zero):
    return asym_dequant(q, scale, zero)

def four_bit_quant_dequant(x):
    return four_bit_dequant(*four_bit_quant(x))
