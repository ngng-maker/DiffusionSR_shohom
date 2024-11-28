import torch
from skimage.metrics import structural_similarity as ssim_id


def PSNR(op, t, batch_size): 
    mse = torch.sum((t - op) ** 2) 
    mse /= (batch_size*80*80)
    max_pixel = torch.max(t)
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
