import torch
from skimage.metrics import structural_similarity as ssim_id


def PSNR(op, t, batch_size):
    """Peak signal-to-noise ratio between prediction `op` and target `t`.

    The spatial normaliser used to be hardcoded as 80*80, which is correct only for the SS316L /
    Ti-6Al-4V cross-sections. It is now read from the target tensor so the metric stays valid at
    other resolutions (the zero-shot resolution probes and any non-80x80 dataset). For 80x80 inputs
    this is numerically identical to the previous implementation.

    Note (unchanged on purpose): the sum runs over channels but the divisor does not include them,
    so multi-field inputs yield a channel-count-inflated MSE and correspondingly lower PSNR. That
    behaviour is preserved so previously reported multifield numbers remain comparable; divide by
    `t.shape[1]` as well if you want a per-channel value.
    """
    mse = torch.sum((t - op) ** 2)  # Total squared error summed over batch, channels and both spatial axes
    # t.shape[-2] and t.shape[-1] are the height and width of the target, so this reproduces the old
    # 80*80 divisor on the original data while adapting automatically to any other grid size.
    mse = mse / (batch_size * t.shape[-2] * t.shape[-1])
    max_pixel = torch.max(t)  # Peak signal level, taken from the data itself rather than assumed to be 1.0 or 255
    # Standard PSNR definition in decibels: 20*log10(peak / RMSE). Higher is better; each +6 dB is
    # roughly a halving of the root-mean-square error.
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def SSIM(op, t, batch_size):
    ssim = 0 
    # print(op.shape, t.shape)
    
    for i in range(op.shape[0]):

        # print(out[0,0].size())
        # print(op.shape, t.shape)
        if isinstance(op, torch.Tensor):
            score = ssim_id(op[i][0].detach().cpu().numpy(), t[i][0].detach().cpu().numpy())#, full=True)
        else:
            score = ssim_id(op[i][0], t[i][0].detach().cpu().numpy())#, full=True)
        ssim+=score/batch_size
    
        #print("SSIM: {}".format(score))
    return ssim
