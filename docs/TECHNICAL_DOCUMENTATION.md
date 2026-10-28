# Facility Usage Prediction System — Technical Documentation

**Goal:** For one residential community, predict each resident's *next* facility booking:
**(1) facility**, **(2) usage day**, **(3) usage hour**, and **(4) when to send a notification
("nudge")** so that it arrives just as the resident would normally book.

All numbers below come from `python -m fup.evaluate` on the untouched holdout
(see `reports/metrics.json`). Everything runs offline with the pinned versions in `requirements.txt`.

---

## 1. Data

### 1.1 Files

| File | Rows | Purpose |
|---|---|---|
| `data/bookings.csv` | 25,775 | **The dataset.** One row per booking. |
| `data/facilities.csv` | 8 | Facility catalogue: opening hours, slot capacity, slot length. |
| `data/residents.csv` | 320 | Resident → flat number (reference only). |
| `data/generator_truth/resident_profiles.csv` | 320 | Hidden generator parameters (persona, booking style, drift date). **Never read by the pipeline**; used only in error analysis to explain mistakes. |

`bookings.csv` columns (the four core fields required by the brief plus an id):

| Column | Meaning | Example |
|---|---|---|
| `booking_id` | Record reference, in creation order | `B000123` |
| `resident_id` | Resident who booked | `R-104` |
| `facility_id` | Booked facility (`F01`–`F08`, see `facilities.csv`) | `F01` (Gym) |
| `booking_ts` | When the booking was created (minute precision) | `2025-03-02 18:20` |
| `usage_ts` | Scheduled start of the slot (on the hour) | `2025-03-03 07:00` |

Period: bookings created **1 Jan 2025 – 31 Dec 2025**; 319 of the 320 residents book at least once.

### 1.2 How it was generated (`fup/generate_data.py`, seed 42)

Each simulated week, every resident draws a number of sessions from a Poisson distribution, chooses
days, a facility, an hour and a booking moment from their personal profile. Then a
**capacity pass** admits bookings in creation order: when a slot is full, the resident is moved to an
adjacent hour, or the attempt is dropped (1,871 attempts were dropped). The brief asked for specific
realism properties; the table shows how each one is produced:

| Requirement | How it is simulated | Visible effect in the data |
|---|---|---|
| Resident preferences | 7 personas (early gym, evening gym, swimmer, racket, family-weekend, senior wellness, casual). Each resident's facility mix is drawn from a Dirichlet around the persona, so no two residents are identical. Each resident also has preferred weekdays and a preferred hour per facility. | A resident uses the same facility as their previous booking 49% of the time. |
| Facility popularity / imbalance | Global popularity prior; tennis, squash and clubhouse have 1-slot capacity. | Gym 42.6%, Pool 18.4%, Badminton 14.0%, MP Hall 9.1%, Table Tennis 6.8%, Squash 3.7%, Tennis 3.6%, Clubhouse 1.8%. |
| Time patterns | Weekday and weekend persona day profiles, a +0–2 h weekend shift, early-morning vs evening peaks, and opening hours per facility. Clubhouse uses 3-hour party slots at 11:00, 16:00 and 19:00. | Bimodal usage-hour distribution with peaks at 07:00 and 19:00. |
| Booking lead times | Four booking styles: **planner** (books the day before at a habitual evening time), **same-day** (1–4 h ahead, or the night before for early slots), **batch** (books the whole week on Sunday evening), and **ad-hoc** (geometric lead). Clubhouse bookings are made 3–21 days ahead. | Lead: same day 22%, 1 day 48%, 2 days 11%, 3 days 4%, 4+ days 14%. |
| Sparsity | Log-normal activity rates (heavy tail), 29% casual users, 12% mid-year move-ins, 6% move-outs, and 0–2 vacation breaks per resident. | Median 63 bookings per resident, range 2–304; 47 residents have fewer than 20; the top 10% of residents make 28% of bookings. |
| Noise | 6% exploratory bookings at a random facility, ±1 h jitter (individual β-distributed consistency), 3% random hours, 5% of bookings off the resident's usual booking style, and minute jitter on booking times. | |
| Changing behaviour | (a) **Seasonality**: pool peaks in Apr–Jun, gym peaks in January (new-year resolutions), tennis dips in the monsoon, clubhouse peaks in Oct–Dec (festivals), and activity drops in December. (b) **Personal drift**: 28% of residents get one change event (time shift, facility switch, rate change or booking-style change) at a random date. | Pool share rises from 10% in January to 28% in May. |

---

## 2. Problem framing

**When is a prediction made?** Immediately after a resident creates a booking (the *anchor*, time `t0`),
we predict their *next* booking. Every booking except a resident's first is therefore one
labelled example (25,456 in total). This mirrors production: the system re-scores a resident each time
they book, and schedules one nudge.

**Targets for example k** (anchor = booking k‑1):

| Output | Target definition | Type |
|---|---|---|
| Facility | `facility_id` of booking k | 8-class |
| Usage day | calendar days from `t0`'s date to booking k's usage date. Offsets 0–9 are separate classes; 10 or more is one class, decoded to the resident's typical long gap. The day is scored as an **absolute calendar date**. | 11-class |
| Usage hour | start hour of booking k | 18 classes (05–22) |
| Notification time | the moment booking k was **created**. The ideal nudge reaches the resident when they would book anyway. The model learns the booking minute relative to the usage day's midnight. | regression |

---

## 3. Features and leakage controls (`fup/features.py`)

### 3.1 Leakage controls

1. **Point-in-time history.** For example k, only bookings with `booking_ts ≤ t0` are visible. That is the resident's bookings 0…k‑1 and, for community features, all residents' bookings created by `t0`. Bookings created before `t0` whose usage lies in the future *are* visible (as "already scheduled"), because the live system would know about them.
2. **Target excluded by construction.** Booking k is created after `t0`, so it can never appear in its own features.
3. **Chronological split on the target's creation time.** Train: targets before 1 Sep 2025. **Validation** (tuning only): Sep–Oct 2025. **Test / holdout:** 1 Nov – 31 Dec 2025 (3,843 targets). The test set is never used for fitting, tuning, feature selection or model choice.
4. **Validation used only for tuning; refit afterwards.** After the grid search, the chosen configuration is refit on train + validation, which are both strictly before the holdout. No information from November or December reaches the model.
5. **Generator truth quarantined.** Persona, booking style and drift labels are kept in a separate folder. They are read only by the error-analysis code, never by training or prediction.
6. **Automated tests** (`tests/test_pipeline.py`):
   - `test_features_do_not_peek_into_the_future`: rebuilds the features from a table truncated on 1 Jul 2025 and asserts they are *bit-identical* to the full-table features for every target before that date.
   - `test_features_ignore_target_row`: alters a target booking and asserts that its own features do not change.
   - Split disjointness and order checks, plus a capacity and opening-hours sanity check on the data.

### 3.2 Feature groups (≈ 90 shared + stage-specific)

| Group | Examples | Why |
|---|---|---|
| Facility preference | lifetime share, **exponentially decayed share (30-day half-life)**, last-10 share, last facility, previous facility, streak | Captures habits and lets recent behaviour outweigh old behaviour, which handles drift. |
| Community popularity | 30-day rolling facility share across all residents (point-in-time) | Seasonality, and a cold-start fallback. |
| Activity / sparsity | number of past bookings, history span, bookings in the last 28 days, last and median inter-booking gap | Tells the model how much to trust a resident's own history. |
| Timing of `t0` | weekday, time of day, month | Tells the model where in the resident's week the prediction is made. |
| Lead-time habits | median lead and its IQR, fraction same-day and fraction day-before, median booking time of day and its spread, median booking minute relative to usage day | Feeds the nudge prediction and helps the day prediction. |
| Already-scheduled | count of future bookings at each offset 0–10, overall and per facility | Batch bookers have already booked their week. These features stop the model from predicting a slot that is already taken and help find the next free day. |
| Weekday alignment | decayed weekday preference for each candidate offset (`t0`'s day + o), historic next-booking offset distribution | Turns "likes Mon/Wed/Fri" into "the next booking is probably in 2 days". |
| Stage-conditional | for the chosen facility f: weekday profile at each offset; hour histogram (all-time, decayed, weekday/weekend-specific); most frequent hour; last hour at f; booking-time habit at f; predicted hour − typical lead | The day, hour and nudge models depend on which facility (and day) has been chosen. |

---

## 4. Modelling approach (`fup/model.py`)

### 4.1 Chained conditional models

```
P(facility | history)                                → HistGradientBoostingClassifier (8 classes)
P(day offset | history, facility)                    → HistGradientBoostingClassifier (11 classes)
P(hour | history, facility, day)                     → HistGradientBoostingClassifier (18 classes)
booking minute | history, facility, day, hour        → HistGradientBoostingRegressor (absolute-error loss)
```

* **Training uses teacher forcing.** Each downstream stage is trained on the *true* upstream values.
* **Inference uses greedy decoding.** Each stage takes the argmax of the previous stage, so the four outputs are mutually consistent (the hour is the hour *at the predicted facility on the predicted day*).
* The nudge is the predicted booking minute. It is clamped to after `t0` and at least 15 minutes before the predicted usage.
* **Absolute-error loss** makes the regressor predict the conditional *median* booking time. This is robust to ad-hoc bookers, whose lead times are heavy-tailed.
* Facility, weekday and similar features are native categoricals in the gradient-boosting models.

### 4.2 Why gradient boosting

The data is tabular, mixed-scale and contains missing values (for example "never used this facility").
The relationships are non-linear and involve interactions such as "planner AND evening AND
facility = gym". HistGradientBoosting handles all of these natively, is fast (one CPU, about 4 minutes
including tuning), deterministic with a fixed seed, and needs no extra dependencies beyond scikit-learn.

### 4.3 Tuning

A per-stage grid over 3 configurations (learning rate, number of trees, number of leaves, minimum leaf
size), scored on the **validation window** using accuracy, or the share of nudges within ±60 min for the
regressor. The selected configuration is the same for all stages: `lr=0.05, 200 trees, 15 leaves, min_leaf=100`.
The tuning log is in `models/training_metadata.json`.

### 4.4 Alternatives considered

| Alternative | Outcome / reason not chosen |
|---|---|
| **Habit heuristic** (decayed-most-frequent facility, most common offset, modal hour at that facility, median booking time) | Kept as **baseline**. It is surprisingly strong on facility top-1, but much weaker on day and nudge (see §5). |
| **Popularity baseline** (community mode) | Kept as a floor reference. |
| **Joint beam decoding**: search the top-3 facilities × top-3 days and maximise p(f)·p(d\|f)·p(h\|f,d) | Same validation score as greedy decoding (0.459 vs 0.462 mean output accuracy) with up to 9 hour-model calls per record instead of 1. It is available as `ChainModel(decode="joint")`. |
| Independent (non-chained) models per output | Rejected: the outputs could contradict each other, for example a 07:00 hour for a clubhouse that opens at 10:00, or a nudge after the slot. |
| Predict usage as one timestamp (regression / survival model over time-to-next) | Regression averages two habitual slots (07:00 and 19:00) into a nonsensical 13:00. A classification head per component handles multimodal habits. A time-to-event model (e.g. discrete-time hazard) is the natural next step. |
| Sequence models (RNN / Transformer over booking history) | Too little data per resident (median 63 events) for a deep model to beat engineered features. They would also add heavy dependencies and reduce interpretability. |
| Per-resident Markov / frequency models only | Equivalent to the habit baseline. They cannot borrow strength across residents for cold start or seasonality. |
| Bigger hyper-parameter search | Gains on validation were under 0.5 pp between configurations, so the effort is better spent on features. |

---

## 5. Results on the holdout (Nov–Dec 2025, 3,843 bookings, never seen)

**Match rules** (one per output; these define "YES/NO" in the review table):

* facility = exact match
* day = exact calendar date
* hour = exact start hour
* nudge = within ±60 min of the actual booking-creation time

| Output | Metric | **Model** | Habit baseline | Popularity baseline |
|---|---|---|---|---|
| Facility | Accuracy (top-1) | **57.8%** | 57.9% | 41.9% |
| | Top-3 accuracy | **86.4%** | 76.9% | 71.1% |
| | Macro F1 | 0.335 | 0.339 | 0.074 |
| | Log loss | **1.23** | 2.48 | 1.74 |
| Usage day | Exact date | **39.2%** | 23.7% | 20.4% |
| | Within ±1 day | **53.2%** | 43.3% | 45.9% |
| | MAE (days) | **3.86** | 4.43 | 4.70 |
| Usage hour | Exact hour | **47.4%** | 46.8% | 17.4% |
| | Within ±1 h | **71.9%** | 71.7% | 36.1% |
| | MAE (h) | **1.80** | 1.89 | 6.33 |
| Notification | Within ±60 min of actual booking | **35.0%** | 18.9% | 6.6% |
| | Within ±3 h | **40.2%** | 21.8% | 16.2% |
| | Median abs. error | **23.5 h** | 47.8 h | 25.3 h |
| | Nudge at/before actual booking | 68.6% | 56.2% | 68.4% |
| **Overall** | **Mean output accuracy** (average of the 4 match rates) | **44.8%** | 36.8% | 21.6% |
| | Outputs matched per record (of 4) | **1.79** | 1.47 | 0.86 |
| | All 4 outputs match | **13.5%** | 8.0% | 1.1% |
| | Facility + day + hour match | **18.0%** | 10.7% | 3.7% |
| | Right facility and usage time within ±2 h | **23.7%** | 14.4% | 7.2% |

The headline overall metric, **mean output accuracy**, is the brief's "x of 4" averaged over all rows
and divided by 4. Validation gave 46.2%; the holdout gives 44.8%. The small gap reflects the later
period's December activity dip and new drift events, not overfitting to the test set, which was never
looked at during development decisions.

Distribution of matches per record: 0 of 4: 762 · 1 of 4: 853 · 2 of 4: 1,161 · 3 of 4: 549 · **4 of 4: 518**.

**Where the model adds value:** it matches the habit baseline on facility top-1 and is far better
calibrated (top-3 accuracy 86% vs 77%, log loss halved). It is **+15.5 pp on the exact day** and **+16 pp on the nudge**,
because it learns booking rhythm, weekday alignment and already-scheduled slots, which a frequency heuristic cannot.

## 6. Error analysis

The full segment table is in the *Error analysis* sheet and the HTML report.

| Finding | Evidence | Interpretation / remedy |
|---|---|---|
| **History length dominates** | Mean output accuracy: 2.5% with ≤5 past bookings → 17% (6–20) → 27% (21–50) → 40% (51–100) → **54% (100+)** | Cold start. Use the onboarding survey or the cohort prior; show low-confidence nudges as generic suggestions. |
| **Switching facility is the main facility error** | When the next booking uses the same facility as the previous one: 84% facility accuracy; when it switches: 35% | Many residents rotate facilities (racket players use badminton, tennis and squash). Top-3 accuracy is 86%, so a nudge listing 2–3 options would capture most of these. |
| **Minority facilities are rarely predicted** | Tennis 1.5%, Table Tennis 2.9%, Squash 9.8%, Clubhouse 9.4% accuracy vs Gym 82% | Class imbalance plus the fact that these are secondary choices for mixed users. Class weighting would trade overall accuracy for macro-F1. The choice depends on the business goal (see §7). |
| **Ad-hoc bookers make nudge timing hard** | Nudge ±60 min: batch bookers 60%, planners 37%, same-day 29%, **ad-hoc 8%** | Ad-hoc booking moments are inherently random. For them, a fixed "evening before" nudge is the most practical choice. |
| **Long lead times make the day hard** | Exact day: same/next-day bookings 45%, 4+ days ahead 15% | Far-ahead bookings (clubhouse parties, batch weeks) have a much larger set of possible days. |
| **Weekends are harder** | Hour accuracy on Sat/Sun is 36–37% vs 49–53% on weekdays | Weekend routines shift hours and include more family or casual use. |
| **Persona** | Early-gym 59% vs casual 19% and family-weekend 23% mean output accuracy | Consistent with how regular each behaviour is. |
| **Behaviour drift** | Residents whose drift happened in Sep–Oct, just before the holdout: mean output accuracy 41.4% vs 44.9% for no-drift, nudge 30% vs 34%. Drift *during* the test period: "all 4" drops to 5.8% (vs 14.3%). | The 30-day decayed features recover within a few weeks. Faster adaptation would need online retraining or shorter half-lives for residents with signs of drift. |
| **Confidence is informative** | Facility probability <0.4: 24% mean output accuracy; ≥0.8: 58–61% | Use a confidence threshold to decide whether to nudge at all. |

## 7. Limitations

* **Synthetic data.** The patterns are the ones I designed. Real data will contain behaviours the
  generator does not model, such as cancellations, no-shows, group bookings, weather and holidays.
  The accuracy figures show that the pipeline *recovers simulated structure*. They are not a forecast
  of production performance.
* **Strict exact-match metrics** (exact calendar day, exact hour, ±60 min) understate usefulness. The
  ±1 day and ±1 h figures and the top-3 accuracy are also reported for that reason.
* **The nudge target is a proxy.** I assume the ideal nudge time is when the resident would book
  naturally. In reality, nudges change behaviour, so the true objective is uplift and would need
  A/B testing.
* **Greedy chain error propagation.** If the facility is wrong, the hour and nudge are conditioned on
  the wrong facility. Joint decoding did not help on this data, but it may help on real data.
* **One fixed snapshot model.** There is no online learning. Residents with recent drift are
  under-served until the model is retrained, which should be weekly in production.
* **Capacity is not used at prediction time.** The model does not know whether the predicted slot is
  already full for other residents.
* **Inactive residents.** Predictions are made relative to the last booking. In live mode the slot is
  rolled forward by whole weeks and marked `inactive >30d`.
* **Single community, one year of data.** Seasonality is learned from a single cycle.

## 8. Reproducibility

* Every random process is seeded (`config.SEED = 42`). `generate_data` is byte-identical across runs.
  Gradient boosting runs with a fixed `random_state` and without early-stopping randomness.
* `models/training_metadata.json` records the split dates, the chosen parameters, the dataset SHA-256,
  the scikit-learn version and the validation metrics.
* `./run_all.sh` regenerates every artifact from scratch.
