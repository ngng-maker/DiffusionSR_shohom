"""
Keep only the latest version of every W&B artifact collection; list (default) or delete the rest.

Each training run writes its checkpoints to one collection (`<run>_ckpt`), and each eval to one
(`<run>_eval_predictions`), so "one collection" = "one experiment/task". Older versions pile up because
upload_checkpoint_artifact() logs a new ~70-160 MB version on every improvement, and the end-of-training
cleanup never deleted anything: every version carries an `epoch_N` alias, and W&B refuses to delete an
aliased version unless delete_aliases=True.

The version kept per collection is the one aliased `latest`, or the highest version number if none is.
--also-keep-best additionally keeps the version aliased `best` where that differs from `latest`.

DRY RUN BY DEFAULT: nothing is deleted unless --apply is given. Deletion is permanent.

    python scripts/wandb_prune_artifact_versions.py --entity ngng- \
        --projects Flow3D_SuperResolution RRDN_Encoder --types checkpoint eval_predictions \
        [--also-keep-best] [--plan prune_plan.csv] [--apply]
"""
import argparse
import csv
import sys

import wandb


def version_number(v):
    return int(v.version.lstrip('v'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--entity', required=True)
    ap.add_argument('--projects', nargs='+', required=True)
    ap.add_argument('--types', nargs='+', default=['checkpoint', 'eval_predictions'])
    ap.add_argument('--also-keep-best', action='store_true')
    ap.add_argument('--plan', default='wandb_prune_plan.csv', help='CSV of every version and its decision')
    ap.add_argument('--apply', action='store_true', help='actually delete (permanent)')
    args = ap.parse_args()

    api = wandb.Api(timeout=120)
    rows, del_bytes, del_count, keep_count, best_differs = [], 0, 0, 0, 0
    for project in args.projects:
        for atype in args.types:
            try:
                colls = list(api.artifact_type(atype, f'{args.entity}/{project}').collections())
            except Exception as e:
                print(f'{project}/{atype}: skipped ({e})')
                continue
            for coll in colls:
                versions = list(coll.artifacts())
                if not versions:
                    continue
                latest = [v for v in versions if 'latest' in v.aliases]
                keep = {(latest[0] if latest else max(versions, key=version_number)).version}
                best = [v for v in versions if 'best' in v.aliases]
                if best and best[0].version not in keep:
                    best_differs += 1
                    if args.also_keep_best:
                        keep.add(best[0].version)
                for v in versions:
                    decision = 'keep' if v.version in keep else 'delete'
                    rows.append({'project': project, 'type': atype, 'collection': coll.name,
                                 'version': v.version, 'aliases': ' '.join(v.aliases),
                                 'size_mb': round((v.size or 0) / 1e6, 1), 'decision': decision})
                    if decision == 'delete':
                        del_count += 1
                        del_bytes += v.size or 0
                        if args.apply:
                            v.delete(delete_aliases=True)
                    else:
                        keep_count += 1
            print(f'{project}/{atype}: {len(colls)} collections processed', flush=True)

    with open(args.plan, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['project'])
        w.writeheader()
        w.writerows(rows)

    verb = 'DELETED' if args.apply else 'would delete'
    print(f'\n{verb} {del_count} versions ({del_bytes / 1e9:.1f} GB); keeping {keep_count}.')
    print(f'collections where `best` is not the latest version: {best_differs} '
          f'({"kept" if args.also_keep_best else "best NOT kept; pass --also-keep-best to keep it"})')
    print(f'full per-version plan: {args.plan}')
    if not args.apply:
        print('Dry run only. Re-run with --apply to delete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
