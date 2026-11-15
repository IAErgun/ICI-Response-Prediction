"""Command-line entry point: ``python -m brainmets.training <command> [options]``.

    python -m brainmets.training presets                      # list the available runs
    python -m brainmets.training run t1post-binary            # all 5 folds, in sequence
    FOLD=2 python -m brainmets.training run multiseq-binary   # just fold 2 (one GPU per fold)
    python -m brainmets.training run t1post-binary --fold 2   # the same, without the env var
    python -m brainmets.training learning-curve

Each preset names one experiment from the paper; see :mod:`training.config`.
"""
import argparse

from . import config as cfg_mod


def _add_overrides(p):
    """Options that override the chosen preset."""
    p.add_argument("--fold", type=int, default=None,
                   help="run only this fold (same as FOLD=k in the environment)")
    p.add_argument("--seqs", default=None,
                   help="comma list of sequences, e.g. 't1post,t2,flair'")
    p.add_argument("--task", choices=["binary", "multiclass"], default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help=f"default: $MAX_EPOCHS, else {cfg_mod.default_max_epochs()}")
    p.add_argument("--rebuild-data-lists", action="store_true",
                   help="regenerate the cached .npy data lists before training")
    return p


def resolve(args, base):
    """Apply CLI overrides to a preset.

    Changing the sequence set invalidates the preset's cache and output directories, so
    those are re-derived from the new set rather than silently reused.
    """
    cfg = base
    seqs = tuple(s.strip() for s in args.seqs.split(",")) if args.seqs else None
    if seqs and seqs != base.seqs:
        cfg = cfg.with_overrides(seqs=seqs,
                                 cache_subdir=f"persistent_cache_{'_'.join(seqs)}",
                                 out_subdir=None)
        print(f"note: --seqs changed the sequence set, so cache and output directories "
              f"are derived as persistent_cache_{'_'.join(seqs)} / "
              f"image_only_models_{'_'.join(seqs)}")
    cfg = cfg.with_overrides(task=args.task, max_epochs=args.max_epochs)
    if args.rebuild_data_lists:
        cfg = cfg.with_overrides(rebuild_data_lists=True)
    return cfg


def cmd_presets(args):
    print("available runs (python -m brainmets.training run <name>):\n")
    width = max(len(n) for n in cfg_mod.PRESETS)
    for name, cfg in cfg_mod.PRESETS.items():
        print(f"  {name:{width}}  seqs={','.join(cfg.seqs):22} task={cfg.task:11} "
              f"-> {cfg.out_subdir or 'image_only_models_' + cfg.seq_tag}")
    lc = cfg_mod.LEARNING_CURVE
    print(f"\n  {'learning-curve':{width}}  seqs={','.join(lc.seqs):22} "
          f"task={lc.task:11} -> {lc.out_subdir}")
    print(f"\n  fractions {list(cfg_mod.LEARNING_CURVE_FRACTIONS)}, "
          f"{cfg_mod.LEARNING_CURVE_REPEATS} repeat(s) per fraction")
    return 0


def cmd_run(args):
    from . import runner          # imported here so `presets` needs no torch/monai
    cfg = resolve(args, cfg_mod.PRESETS[args.preset])
    folds = [args.fold] if args.fold is not None else None
    results = runner.run(cfg, folds=folds)
    if results:
        mean = sum(results.values()) / len(results)
        print(f"\n{cfg.metric_name}: " +
              "  ".join(f"fold{k}={v:.4f}" for k, v in sorted(results.items())))
        print(f"mean over {len(results)} fold(s): {mean:.4f}")
    return 0


def cmd_learning_curve(args):
    from . import learning_curve  # imported here so `presets` needs no torch/monai
    cfg = resolve(args, cfg_mod.LEARNING_CURVE)
    fractions = ([float(f) for f in args.fractions.split(",")]
                 if args.fractions else None)
    folds = [args.fold] if args.fold is not None else None
    path = learning_curve.run(cfg, fractions=fractions, repeats=args.repeats,
                              folds=folds)
    print(f"\nresults appended to {path}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m brainmets.training",
        description="Train the 3D CNN experiments.",
        epilog="Site paths come from $BM_DATA_ROOT, $BM_FOLDS_CSV, $BM_WORK_ROOT "
               "and $BM_CACHE_ROOT; see docs/configuration.md.")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("presets", help="list the available runs and where they write")
    p.set_defaults(run=cmd_presets)

    p = sub.add_parser("run", help="train one experiment across the CV folds")
    p.add_argument("preset", choices=sorted(cfg_mod.PRESETS),
                   help="which experiment to run")
    _add_overrides(p)
    p.set_defaults(run=cmd_run)

    p = sub.add_parser("learning-curve", help="sweep training-set size")
    p.add_argument("--fractions", default=None,
                   help="comma list, default "
                        f"{','.join(str(f) for f in cfg_mod.LEARNING_CURVE_FRACTIONS)}")
    p.add_argument("--repeats", type=int, default=None,
                   help="independent subsample draws per fraction "
                        f"(default {cfg_mod.LEARNING_CURVE_REPEATS})")
    _add_overrides(p)
    p.set_defaults(run=cmd_learning_curve)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "run", None):
        ap.print_help()
        return 1
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
