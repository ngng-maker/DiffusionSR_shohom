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
    """Total number of learnable REAL scalars (degrees of freedom) in a module.

    Not simply `sum(p.numel())`. torch's `.numel()` on a complex tensor returns the number of
    COMPLEX elements, and an FNO's spectral weights are complex64 - each carries an independent
    real and imaginary part, i.e. two learnable scalars. Summing raw numel() therefore undercounts
    an FNO by ~2x on exactly the weights that dominate its size, and comparing that against a
    real-valued CNN like RRDB is not a like-for-like comparison: it silently gives the FNO twice
    the capacity at nominally "equal" parameter count.

    Real DOF is the right measure here because it is what determines capacity, optimizer state
    size and memory. Use `count_parameters_naive` if you specifically need the element count.
    """
    # p.is_complex() is True for complex64/complex128 parameters; those contribute two real scalars
    # per element, real parameters contribute one.
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def count_parameters_naive(model) -> int:
    """Raw element count, counting each complex parameter as one. Reported for transparency only."""
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
    print(f"{'config':<30} {'type':<6} {'backend':<12} {'real DOF':>12} {'vs target':>10} {'elements':>12}")
    print("-" * 88)
    for path in paths:
        try:
            model, etype, _ = encoder_from_config(path, upscale_factor, in_channels, out_channels)
        except Exception as exc:  # A config may name a backend this environment cannot build
            print(f"{os.path.basename(path):<30} {'-':<6} {'-':<12} {'ERROR':>12}  {exc}")
            continue
        n = count_parameters(model)              # real DOF - the comparable number
        n_naive = count_parameters_naive(model)  # element count, for transparency
        # getattr with a default because RRDBNet has no `backend` attribute at all.
        backend = getattr(model, "backend", "-")
        ratio = f"{n / target:.3f}x" if target else "-"
        print(f"{os.path.basename(path):<30} {etype:<6} {backend:<12} {n:>12,} {ratio:>10} {n_naive:>12,}")


def search_match(target, backend, upscale_factor, in_channels, out_channels,
                 feature_channels, upsample_mode, padding, top_n, hr_side):
    """Find FNO hyperparameters whose parameter count is closest to `target`.

    A naive grid over a hand-picked list of mode pairs misses good matches, because parameter count
    depends on the PRODUCT m1*m2 and the useful products are spread unevenly across that list. So
    instead of guessing, this calibrates the cost model per (latent, layers) from two measurements
    and then tests only the mode pairs predicted to land near the target.

    Spectral weights dominate and scale as  k * latent^2 * m1 * m2 * layers + overhead,  linear in
    the product m1*m2 for fixed latent and layers. Two probes therefore pin down the line exactly:

        slope    = (n(256) - n(64)) / (256 - 64)      cost per unit of m1*m2
        overhead = n(64) - slope * 64                 everything not in the spectral weights
        wanted   = (target - overhead) / slope        the product that hits the target

    which is inverted to give the wanted product, and the nearby integer factorisations are then
    built and measured exactly rather than trusted.
    """
    # Nyquist caps the useful mode count at half the grid the operator actually runs on. 'pre'
    # upsamples to HR first, so it works on the HR grid; 'spectral' and 'conv' stay at LR. Exceeding
    # the cap is silently a no-op (SpectralConv2d clamps), which would make a "match" fictitious -
    # the extra weights would exist but never be read.
    grid_side = hr_side if upsample_mode == "pre" else hr_side // upscale_factor
    mode_cap = grid_side // 2
    print(f"Nyquist cap for upsample_mode={upsample_mode!r}: {mode_cap} modes "
          f"(operator runs on a {grid_side}x{grid_side} grid)")
    print()

    latents = [24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128]
    layer_counts = [4, 5]

    def build(latent, m1, m2, n_layers):
        """Construct one candidate, returning None if this backend cannot build it."""
        kwargs = dict(feature_channels=feature_channels, latent_channels=latent,
                      num_fno_layers=n_layers, num_fno_modes=[m1, m2],
                      padding=padding, upsample_mode=upsample_mode, backend=backend)
        try:
            return build_encoder("fno", upscale_factor=upscale_factor, in_channels=in_channels,
                                 out_channels=out_channels, encoder_kwargs=kwargs)
        except Exception:
            return None

    results = []
    for latent, n_layers in itertools.product(latents, layer_counts):
        # Two probes at known products (8*8=64 and 16*16=256) to calibrate the line.
        lo, hi = build(latent, 8, 8, n_layers), build(latent, 16, 16, n_layers)
        if lo is None or hi is None:
            continue
        n_lo, n_hi = count_parameters(lo), count_parameters(hi)
        slope = (n_hi - n_lo) / (256 - 64)
        if slope <= 0:
            continue  # Degenerate; nothing to solve for
        overhead = n_lo - slope * 64
        wanted = (target - overhead) / slope
        if wanted < 1:
            continue  # Even the smallest operator overshoots at this latent/layer combination

        # Enumerate integer factorisations near the wanted product. Both factors are capped at the
        # Nyquist limit, and near-square pairs are preferred for a square grid, so the truncation is
        # roughly isotropic rather than arbitrarily favouring one axis.
        candidates = set()
        for m1 in range(4, mode_cap + 1):
            m2 = max(4, min(mode_cap, round(wanted / m1)))
            candidates.add((m1, m2))
        # Score by predicted distance first, and only build the most promising handful exactly.
        ranked = sorted(candidates, key=lambda mm: (abs(mm[0] * mm[1] - wanted), abs(mm[0] - mm[1])))
        for m1, m2 in ranked[:4]:
            model = build(latent, m1, m2, n_layers)
            if model is None:
                continue
            n = count_parameters(model)
            results.append((abs(n / target - 1.0), latent, (m1, m2), n_layers, n))

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
    print(f"# -> {n:,} real DOF ({n / target:.3f}x target)")


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
    parser.add_argument("--upscale_factor", type=int, default=2,
                        help="HR/LR ratio. The data_fields dataset reports 2; the paper's Ti64 task is 4.")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=8, help="how many candidates to print (--match only)")
    # The Nyquist cap depends on the grid the operator actually runs on, which depends on the HR
    # size. Previously hardcoded to 80; wrong for any other dataset, and the cap being wrong makes a
    # reported "match" fictitious because modes above it are silently clamped away unused.
    parser.add_argument("--hr_side", type=int, default=80,
                        help="HR grid side length (dataset.img_shape). Determines the Nyquist mode cap.")
    args = parser.parse_args()

    # An RRDB built at the study's settings is the natural reference point, so compute it either way
    # rather than making the caller supply a magic number.
    rrdb = build_encoder("rrdb", upscale_factor=args.upscale_factor,
                         in_channels=args.in_channels, out_channels=args.out_channels)
    rrdb_params = count_parameters(rrdb)
    print(f"RRDB reference (num_blocks=8, channels=64, upscale={args.upscale_factor}): {rrdb_params:,} real DOF")
    print("NOTE: 'real DOF' counts each complex FNO weight as 2 scalars (re + im). Raw torch numel()")
    print("      counts it as 1, which undercounts an FNO by ~2x against a real-valued CNN.")
    print()

    if args.match is not None:
        search_match(args.match, args.backend, args.upscale_factor, args.in_channels,
                     args.out_channels, args.feature_channels, args.upsample_mode,
                     args.padding, args.top_n, args.hr_side)
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
