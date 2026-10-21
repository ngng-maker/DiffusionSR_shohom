"""
Evaluate the frozen encoder ON ITS OWN, as a reference arm with no generative model.

The interpolant and DDPM arms all condition on (or, for si_base='enc', start from) this encoder, so
its plain HR prediction is the bar a generative stage has to clear. Scored with exactly the metric
code every eval uses (evaluate_coupled.compute_metrics), on the same test split and dataset settings
(FlippedDataset, p_flip=0), and saved in the same predictions.npz layout so comparison notebooks can
load it like any other arm. The encoder is deterministic, so K = 1: ensemble spread is zero and
calibration_2sigma is not meaningful for this arm.

    python -m diffusionsr.analysis.evaluate_encoder_only \
        --data_root <data_fields> --enc_dir <runs/fno_ablation/enc_rrdb> \
        --out_dir eval_results/interpolant/encoder_only
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusionsr.analysis.evaluate_coupled import _px_um, compute_metrics
from diffusionsr.datasets.flipped_dataset import FlippedDataset
from diffusionsr.models.encoder_factory import build_encoder
from diffusionsr.runners.train_diffusion import forwardpass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--enc_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--field_names', nargs='+', default=['temperature'])
    ap.add_argument('--encoder_type', default='rrdb')
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kw = dict(downscale_method='direct', root_folder=args.data_root, normalize='standardize',
              n_steps=1, field_names=args.field_names, p_flip=0.0)
    train_ds = FlippedDataset(split='train', **kw)   # forwardpass reads the factor/stats from it
    test_ds = FlippedDataset(split='test', **kw)
    C = test_ds.n_steps * test_ds.num_fields

    # Same factory and checkpoint the training runs load, so this is byte-identical to their encoder.
    enc = build_encoder(encoder_type=args.encoder_type, upscale_factor=test_ds.factor,
                        in_channels=C, out_channels=C, encoder_kwargs=None)
    ckpt = torch.load(os.path.join(args.enc_dir, 'bestmodel_saved.pth'), map_location=device,
                      weights_only=False)
    enc.load_state_dict(ckpt['model_state_dict'])
    enc.to(device).eval()

    preds, gts = [], []
    with torch.no_grad():
        for res, hr, lr, ulr in DataLoader(test_ds, batch_size=args.batch_size, shuffle=False):
            out = forwardpass(enc, lr.to(device).float(), factor=test_ds.factor, output=True,
                              transform_rescale=False, dataset=train_ds).cpu().numpy()
            for s in range(out.shape[0]):
                preds.append(test_ds.unscale_data(out[s], input_type='hr'))
                gts.append(test_ds.unscale_data(hr[s].numpy(), input_type='hr'))
    pred = np.stack(preds)[:, None]        # (N, K=1, C, H, W)
    gt = np.stack(gts)                     # (N, C, H, W)

    results = compute_metrics(pred, gt, _px_um(test_ds.factor))
    results['note'] = 'frozen encoder alone, K=1: no ensemble, so calibration_2sigma is not meaningful'
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.out_dir))
    with open(os.path.join(args.out_dir, f'{name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.out_dir, 'predictions.npz'),
             pred=pred.astype(np.float32), gt=gt.astype(np.float32),
             ens_std=np.zeros_like(gt, dtype=np.float32),
             field_names=np.array(args.field_names, dtype=object))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
