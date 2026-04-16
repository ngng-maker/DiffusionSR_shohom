from pathlib import Path

import diffusionsr


PACKAGE_DIR = Path(diffusionsr.__file__).resolve().parent        # <repo>/diffusionsr
REPO_ROOT = PACKAGE_DIR.parent                                    # <repo>


def relocate_config_paths(config):
    """Rewrite path-like fields pulled from a W&B config to point at the current repo.

    W&B runs store absolute paths from the training machine. This rewrites them
    to their equivalents under the current checkout:
      - root_folder: rerooted to <REPO_ROOT>/data/<basename> (matches download_data.sh)
      - encoder_results_dir, restart_dir: anchored on 'runs/' and re-rooted under
        <PACKAGE_DIR>, since training is launched from inside diffusionsr/

    Fields without a recognized structure are left untouched.
    """
    if "root_folder" in config:
        basename = Path(str(config["root_folder"]).rstrip("/")).name
        config["root_folder"] = str(REPO_ROOT / "data" / basename)

    for key in ("encoder_results_dir", "restart_dir"):
        if key not in config:
            continue
        s = str(config[key])
        i = s.find("runs/")
        if i != -1:
            config[key] = str(PACKAGE_DIR / s[i:])

    return config
