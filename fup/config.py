"""Central configuration: paths, seeds, calendar windows and facility catalogue."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

BOOKINGS_CSV = DATA_DIR / "bookings.csv"
FACILITIES_CSV = DATA_DIR / "facilities.csv"
RESIDENTS_CSV = DATA_DIR / "residents.csv"
# Generator ground truth (personas, drift dates). NEVER used by the pipeline for
# fitting/predicting -- only by the error analysis to explain mistakes.
TRUTH_CSV = DATA_DIR / "generator_truth" / "resident_profiles.csv"

SEED = 42

# Simulated observation window: one calendar year of bookings for one community.
DATA_START = pd.Timestamp("2025-01-01")
DATA_END = pd.Timestamp("2026-01-01")  # exclusive; booking_ts < DATA_END

# Chronological split on the *target* booking's creation time (booking_ts).
VAL_START = pd.Timestamp("2025-09-01")   # train:  targets booked before this
TEST_START = pd.Timestamp("2025-11-01")  # val:    [VAL_START, TEST_START)
                                         # test:   [TEST_START, DATA_END)  -- holdout

# Match tolerance for the notification-time output.
NUDGE_TOLERANCE_MIN = 60

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# id, name, short label, opening hour, last start hour, slot capacity, base popularity, slot minutes
FACILITIES = [
    ("F01", "Gym", "Gym", 5, 22, 25, 0.33, 60),
    ("F02", "Swimming Pool", "Pool", 6, 20, 15, 0.17, 60),
    ("F03", "Badminton Court", "Badminton", 6, 22, 2, 0.15, 60),
    ("F04", "Tennis Court", "Tennis", 6, 21, 1, 0.08, 60),
    ("F05", "Squash Court", "Squash", 6, 22, 1, 0.05, 60),
    ("F06", "Table Tennis Room", "Table Tennis", 7, 22, 2, 0.08, 60),
    ("F07", "Multipurpose Hall", "MP Hall", 6, 21, 1, 0.09, 60),
    ("F08", "Clubhouse Party Hall", "Clubhouse", 10, 20, 1, 0.05, 180),
]
FACILITY_IDS = [f[0] for f in FACILITIES]
FACILITY_LABEL = {f[0]: f[2] for f in FACILITIES}
N_FAC = len(FACILITIES)
