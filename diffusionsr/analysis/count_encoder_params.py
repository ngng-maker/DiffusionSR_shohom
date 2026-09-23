"""Count conditioning-encoder parameters, and search for a parameter-matched configuration.

Two uses:

1. Report the parameter count of every arm in the study, so the matched-parameter claim can be
   verified rather than asserted:

       python -m diffusionsr.analysis.count_encoder_params --configs diffusionsr/configs/fno_ablation/enc_*.yml

2. Find FNO hyperparameters whose parameter count matches a target (normally RRDB's). This is
   required for the PhysicsNeMo calibration arm: PhysicsNeMo's FNO has a different decoder than the
   builtin body, so identical kwargs give a different size, and an unmatched calibration arm would
   be confounded by capacity as well as backend:

       python -m diffusionsr.analysis.count_encoder_params --match 5904321 --backend physicsnemo

Comparing models of different sizes tells you about capacity, not architecture. That is the whole
reason this script exists.
"""

import argparse  # Command-line parsing
import glob  # Expands the --configs shell pattern when the shell has not already done so
import itertools  # product() builds the hyperparameter grid for --match
import os  # Basename formatting in the report

import yaml  # Configs are plain YAML

from diffusionsr.models.encoder_factory import build_encoder  # Single construction point for both architectures


def count_parameters(model) -> int:
    """Total number of learnable scalars in a module."""
    # .parameters() yields every registered Parameter tensor (recursively through submodules);
    # .numel() is the element count of one tensor, so the sum is the model's total parameter count.
    # Complex parameters (the FNO spectral weights) count their real and imaginary parts separately,
    # which is the correct comparison against a real-valued CNN.
    return sum(p.numel() for p in model.parameters())


def encoder_from_config(path: str, upscale_factor: int, in_channels: int, out_channels: int):
    """Build the encoder a config file describes, exactly as train_srdiff would."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}  # `or {}` guards against an empty file parsing to None
    encoder_type = cfg.get("encoder_type", "rrdb")  # Absent key means the RRDB baseline
    encoder_kwargs = cfg.get("encoder_kwargs", None)  # Architecture-specific extras, may be absent
    model = build_encoder(encoder_type, upscale_factor=upscale_factor,
                          in_channels=in_channels, out_channels=out_channels,
                          encoder_kwargs=encoder_kwargs)
    return model, encoder_type, encoder_kwargs


def report_configs(paths, upscale_factor, in_channels, out_channels, target):
    """Print one row per config: architecture, backend, parameter count, ratio to target."""
    print(f"{'config':<30} {'type':<6} {'backend':<12} {'params':>12} {'vs target':>10}")
    print("-" * 74)
    for path in paths:
        try:
            model, etype, _ = encoder_from_config(path, upscale_factor, in_channels, out_channels)
        except Exception as exc:  # A config may name a backend this environment cannot build
            print(f"{os.path.basename(path):<30} {'-':<6} {'-':<12} {'ERROR':>12}  {exc}")
            continue
        n = count_parameters(model)
        # getattr with a default because RRDBNet has no `backend` attribute at all.
        backend = getattr(model, "backend", "-")
        ratio = f"{n / target:.3f}x" if target else "-"
        print(f"{os.path.basename(path):<30} {etype:<6} {backend:<12} {n:>12,} {ratio:>10}")


def search_match(target, backend, upscale_factor, in_channels, out_channels,
                 feature_channels, upsample_mode, padding, top_n):
    """Grid-search FNO hyperparameters for the closest parameter count to `target`."""
    # Search space. Latent width and mode count trade off against each other: spectral weights scale
    # as latent^2 * modes_x * modes_y per layer, so many combinations hit a similar total by
    # different routes. Reporting several lets you pick on grounds other than size alone.
    latents = [32, 48, 56, 64, 80, 96, 112, 128]
    modes = [(8, 8), (10, 10), (12, 12), (13, 14), (14, 13), (14, 14), (16, 11), (16, 12), (16, 16), (20, 20)]
    layers = [4, 5]

    results = []
    for latent, mode, n_layers in itertools.product(latents, modes, layers):
        kwargs = dict(feature_channels=feature_channels, latent_channels=latent,
                      num_fno_layers=n_layers, num_fno_modes=list(mode),
                      padding=padding, upsample_mode=upsample_mode, backend=backend)
        try:
            model = build_encoder("fno", upscale_factor=upscale_factor, in_channels=in_channels,
                                  out_channels=out_channels, encoder_kwargs=kwargs)
        except Exception:
            continue  # Skip combinations this backend rejects (e.g. spectral mode on physicsnemo)
        n = count_parameters(model)
        # Rank by relative distance from the target, so the result is scale-independent.
        results.append((abs(n / target - 1.0), latent, mode, n_layers, n))

    if not results:
        print(f"No buildable configuration found for backend={backend!r}.")
        return

    results.sort()
    print(f"Target: {target:,} parameters   (backend={backend}, upsample_mode={upsample_mode})")
    print()
    print(f"{'latent':>7} {'modes':>10} {'layers':>7} {'params':>12} {'ratio':>8}")
    print("-" * 50)
    for _, latent, mode, n_layers, n in results[:top_n]:
        print(f"{latent:>7} {str(list(mode)):>10} {n_layers:>7} {n:>12,} {n / target:>7.3f}x")

    _, latent, mode, n_layers, n = results[0]
    print()
    print("Paste into the config's encoder_kwargs:")
    print(f"  latent_channels: {latent}")
    print(f"  num_fno_layers: {n_layers}")
    print(f"  num_fno_modes: {list(mode)}")
    print(f"  backend: \"{backend}\"")
    print(f"# -> {n:,} params ({n / target:.3f}x target)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="config yaml paths (globs accepted) to report parameter counts for")
    parser.add_argument("--match", type=int, default=None,
                        help="target parameter count to search for; 5904321 is RRDB at the study's settings")
    parser.add_argument("--backend", default="builtin", choices=["builtin", "physicsnemo", "auto"],
                        help="FNO backend to search over (--match only)")
    parser.add_argument("--upsample_mode", default="pre", choices=["pre", "spectral", "conv"],
                        help="FNO variant to search over (--match only)")
    parser.add_argument("--feature_channels", type=int, default=64,
                        help="conditioning width; must equal the U-Net's init_dim")
    parser.add_argument("--padding", type=int, default=8, help="non-periodic FFT padding margin")
    parser.add_argument("--upscale_factor", type=int, default=4, help="HR/LR ratio, 4 for the SS316L 4x task")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=8, help="how many candidates to print (--match only)")
    args = parser.parse_args()

    # An RRDB built at the study's settings is the natural reference point, so compute it either way
    # rather than making the caller supply a magic number.
    rrdb = build_encoder("rrdb", upscale_factor=args.upscale_factor,
                         in_channels=args.in_channels, out_channels=args.out_channels)
    rrdb_params = count_parameters(rrdb)
    print(f"RRDB reference (num_blocks=8, channels=64, upscale={args.upscale_factor}): {rrdb_params:,} params")
    print()

    if args.match is not None:
        search_match(args.match, args.backend, args.upscale_factor, args.in_channels,
                     args.out_channels, args.feature_channels, args.upsample_mode,
                     args.padding, args.top_n)
        return

    # Default to reporting every arm in the study when no explicit list is given.
    paths = args.configs or sorted(glob.glob("diffusionsr/configs/fno_ablation/enc_*.yml"))
    # Re-glob each entry: a quoted pattern reaches us unexpanded when the shell did not expand it.
    expanded = []
    for p in paths:
        expanded.extend(sorted(glob.glob(p)) or [p])
    report_configs(expanded, args.upscale_factor, args.in_channels, args.out_channels, rrdb_params)


if __name__ == "__main__":
    main()
