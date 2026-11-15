"""Site configuration and cohort constants.

Every path, input or output, is read from an environment variable here and nowhere
else, so no site-specific location is baked into the code (see docs/configuration.md).
Everything else in this module is a cohort definition shared by the whole package —
change it in one place and every check, table and export follows.
"""
import os

# ---------------------------------------------------------------------- inputs
# Resolved on each call rather than at import, so setting an environment variable always
# takes effect and tests can point the package at a fixture.


def resolve_input(env_var, what):
    """Resolve an input path from the environment, failing with a usable message.

    Without this, an unset variable surfaces much later as a FileNotFoundError from
    somewhere inside pandas, which says nothing about what to do.
    """
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"{env_var} is not set, so {what} cannot be located.\n"
            f"  export {env_var}=/your/path   (see docs/configuration.md)")
    if not os.path.exists(value):
        raise SystemExit(f"{env_var} points at {value!r}, which does not exist ({what}).")
    return value


def folds_csv():
    """The cohort table: one row per metastasis."""
    return resolve_input("BM_FOLDS_CSV", "the cohort table")


def clin_csv():
    """The clinical table: one row per treatment course."""
    return resolve_input("BM_CLIN_CSV", "the clinical table")


def centroids_csv():
    """The lesion centroid / RANO label table."""
    return resolve_input("BM_CENTROIDS_CSV", "the lesion centroid table")


def data_root():
    """Root of the preprocessed MRI tree."""
    return resolve_input("BM_DATA_ROOT", "the preprocessed MRI tree")


# --------------------------------------------------------------------- outputs
def resolve_output(env_var, what):
    """Resolve an output root from the environment.

    The directory need not exist yet (it is created on first use), but the variable has
    to be set. Falling back to a placeholder instead would have a run create a literal
    ``/path/to/...`` wherever it has permission to, and fill it with checkpoints.
    """
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"{env_var} is not set, so {what} cannot be located.\n"
            f"  export {env_var}=/your/path   (see docs/configuration.md)")
    return value


def work_root():
    """Where runs write: checkpoints, data lists, logs, predictions."""
    return resolve_output("BM_WORK_ROOT", "the run output directory")


def cache_root():
    """Where the MONAI dataset caches live (large; safe to delete)."""
    return resolve_output("BM_CACHE_ROOT", "the dataset cache directory")

# ------------------------------------------------------------ cohort definition
# One baseline scan == one treatment course. The clinical table is joined on this.
MERGE_KEY = "PreTreatment_Image"
NUM_FOLDS = 5
# lower_size_threshold in the trainer; a metastasis diameter must EXCEED this.
LOWER = 10
LABEL_MET = "RANOResponse_-40_40_Met"
# The pipeline's own per-patient id. The real patient id is embedded in the image
# path (see data.add_patient_round); Patient_Number is 1:1 with it.
PID_COL = "Patient_Number"

# Sequence name -> presence column, for the multisequence cohorts.
PRESENCE = {"t2": "T2_present", "flair": "FLAIR_present"}
# Sequences available for every metastasis, so there is no presence column to filter on
# and asking for them is a no-op rather than an error.
ALWAYS_PRESENT = frozenset({"t1post"})

# ------------------------------------------------------------ clinical variables
CONT = ["Age_at_Treatment", "Number_of_BM", "BMI_Result",
        "ALB_Latest_PreTreatment", "HGB_Latest_PreTreatment", "PLT_Latest_PreTreatment"]
CAT = ["Gender", "STEROIDS", "Histology_New"]

# All checkpoint-inhibitor flag columns in the clinical dataset. Regimen labels in this
# package are built from all five, which is how Table 1 lists them.
#
# This DIFFERS FROM THE PAPER'S SUBGROUP ANALYSIS, which built its regimen subgroups from
# IPI/NIVO/PEMBRO only (BASE3 below): atezolizumab and durvalumab courses were too sparse
# in that cohort to support subgroups of their own, so they were reported under "Other or
# none". Counting all five keeps the labels meaningful on datasets where those agents are
# common. On the paper's data it moves those courses out of "Other or none", so the
# regimen subgroups will not match the published ones exactly.
ICI = ["IPI", "NIVO", "PEMBRO", "ATEZO", "DURVA"]
# The three agents the paper's subgroup label was built from, kept so its "Other or none"
# bucket can still be reproduced and explained (see checks.describe_ici_other_...).
BASE3 = ["IPI", "NIVO", "PEMBRO"]

# Column-name tokens that identify an agent column when scanning the clinical CSV
# for inhibitors beyond the ICI list above.
AGENT_TOKENS = ["IPI", "IPILIMUMAB", "NIVO", "NIVOLUMAB", "PEMBRO", "PEMBROLIZUMAB",
                "ATEZO", "ATEZOLIZUMAB", "DURVA", "DURVALUMAB", "AVELU", "AVELUMAB",
                "TREMEL", "TREMELIMUMAB", "CEMIP", "CEMIPLIMAB"]

# Truthy encodings used by the clinical flag columns.
POS = frozenset({1, 1.0, "1", "Yes", "yes", "Y", "TRUE", "True", True})

# Cohort keys and the labels they print as.
COHORTS = ("dev", "test")
COHORT_LABELS = {"dev": "DEVELOPMENT (train/val)", "test": "HELD-OUT INTERNAL TEST"}
# Short forms, for one-line progress output.
COHORT_SHORT = {"dev": "DEVELOPMENT", "test": "TEST"}
EXTERNAL_NOTE = "\n[External site: summarize separately from its own clinical file.]"
