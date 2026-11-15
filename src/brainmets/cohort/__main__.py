"""Command-line entry point: ``python -m brainmets.cohort <command> [options]``.

Run with no arguments, or with ``--help``, to list the available commands. Each
command maps to one function in :mod:`cohort.checks`, :mod:`cohort.demographics`,
:mod:`cohort.table1` or :mod:`cohort.export`; every one of them is also importable
and callable directly.
"""
import argparse

from . import checks, config as cfg, data, demographics, export, make_folds, table1


def _cohort_options(p, seqs=True, folds=True, num_folds=True):
    """Options shared by the commands that re-derive the cohort from the CSV."""
    if folds:
        p.add_argument("--folds-csv", default=None,
                       help="cohort/folds table (default: $BM_FOLDS_CSV)")
    p.add_argument("--lower", type=int, default=None,
                   help=f"lower_size_threshold (default {cfg.LOWER})")
    if num_folds:
        p.add_argument("--num-folds", type=int, default=None,
                       help=f"number of CV folds (default {cfg.NUM_FOLDS})")
    if seqs:
        p.add_argument("--require-seqs", default="",
                       help="comma list, e.g. 't2,flair' for the multisequence cohort")
    return p


def _cohort_kwargs(args, seqs=True, folds=True):
    kw = {"lower": args.lower, "num_folds": args.num_folds}
    if folds:
        kw["folds_csv"] = args.folds_csv
    if seqs:
        kw["require_seqs"] = data.parse_require_seqs(args.require_seqs)
    return kw


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m brainmets.cohort",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All reports are PHI-SAFE: aggregates and column names only.")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    # ---------------------------------------------------------------- reports
    p = sub.add_parser("demographics", help="Table 1 demographics, one row per course")
    p.set_defaults(run=lambda a: demographics.report())

    p = sub.add_parser("table1", help="Table 1 rebuilt with one course per patient")
    p.add_argument("--write-lists", action="store_true",
                   help="write kept/dropped course IDs to CSV (not printed)")
    p.add_argument("--out-dir", default=".")
    p.set_defaults(run=lambda a: table1.rebuild(write_lists=a.write_lists,
                                                out_dir=a.out_dir))

    # ---------------------------------------------------------------- counts
    p = _cohort_options(sub.add_parser(
        "count-patients", help="patients and mets per cohort, with a leakage check"))
    p.set_defaults(run=lambda a: checks.count_patients_per_cohort(**_cohort_kwargs(a)))

    p = _cohort_options(sub.add_parser(
        "count-samples", help="per-fold train/val sample counts"))
    p.add_argument("--npy-dir", default=None,
                   help="dir with *_data_list.npy to cross-check (optional)")
    p.add_argument("--prefix", default="Fold_Binary_", help="data-list prefix")
    p.add_argument("--tag", default="",
                   help="seq_tag in data-list names ('', 't1post', 't1post_t2_flair')")
    p.set_defaults(run=lambda a: checks.count_samples_per_fold(
        npy_dir=a.npy_dir, prefix=a.prefix, tag=a.tag, **_cohort_kwargs(a)))

    p = _cohort_options(sub.add_parser(
        "count-classes", help="class balance and zero-rule baselines"))
    p.set_defaults(run=lambda a: checks.count_train_val_classes(**_cohort_kwargs(a)))

    # ---------------------------------------------------------------- checks
    p = sub.add_parser("verify-courses",
                       help="is a treatment course uniquely identified?")
    p.set_defaults(run=lambda a: checks.verify_course_identity())

    p = sub.add_parser("check-round-age",
                       help="diagnose the negative age delta for repeat-course patients")
    p.set_defaults(run=lambda a: checks.check_round_sort_and_age_delta())

    p = sub.add_parser("sensitivity-course-key",
                       help="counterfactual: age instead of round as the course key")
    p.set_defaults(run=lambda a: checks.sensitivity_earliest_course_key())

    p = sub.add_parser("describe-ici",
                       help="'other or none' regimens and repeat-course patients")
    p.set_defaults(run=lambda a: checks.describe_ici_other_and_repeat_courses())

    p = sub.add_parser("compare-labels",
                       help="do the used and intended label sources agree?")
    p.add_argument("--folds-csv", default=None)
    p.add_argument("--centroids-csv", default=None)
    p.add_argument("--label", default=None, help=f"default {cfg.LABEL_MET}")
    p.set_defaults(run=lambda a: checks.compare_label_sources(
        folds_csv=a.folds_csv, centroids_csv=a.centroids_csv, label=a.label))

    # ------------------------------------------------------------ fold creation
    p = sub.add_parser(
        "make-folds",
        help="generate patient-grouped, label-stratified fold columns for a table")
    p.add_argument("--in", dest="in_csv", default=None,
                   help="cohort table to read (default: $BM_FOLDS_CSV)")
    p.add_argument("--out", dest="out_csv", required=True,
                   help="where to write the table with fold columns added")
    p.add_argument("--num-folds", type=int, default=None,
                   help=f"cross-validation folds (default {cfg.NUM_FOLDS})")
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="share of patients held out from every fold (default 0.15; "
                        "0 uses the whole cohort for cross-validation)")
    p.add_argument("--group-col", default=None,
                   help=f"patient column to group by (default {cfg.PID_COL!r})")
    p.add_argument("--group-from-image-path", action="store_true",
                   help="parse the patient id out of the image path instead, as the "
                        "trainer does")
    p.add_argument("--label-col", default=None,
                   help=f"label to stratify on (default {cfg.LABEL_MET!r})")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--drop-unlabelled", action="store_true",
                   help="exclude rows whose label is not a RANO level instead of stopping")
    p.set_defaults(run=lambda a: make_folds.make_folds(
        in_csv=a.in_csv, out_csv=a.out_csv, num_folds=a.num_folds,
        test_fraction=a.test_fraction, group_col=a.group_col,
        label_col=a.label_col, seed=a.seed,
        from_image_path=a.group_from_image_path,
        drop_unlabelled=a.drop_unlabelled))

    # ---------------------------------------------------------------- exports
    p = _cohort_options(sub.add_parser(
        "common-cohort-csv", help="write the all-sequence common cohort to CSV"),
        seqs=False, num_folds=False)
    p.add_argument("--out", dest="out_csv", default=None, help="output CSV path")
    p.set_defaults(run=lambda a: export.write_common_cohort_csv(
        folds_csv=a.folds_csv, out_csv=a.out_csv, lower=a.lower))

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "run", None):
        ap.print_help()
        return 1
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
