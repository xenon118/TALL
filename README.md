# TALL — Tiered API Linking for Legacy wells

Links legacy oil and gas well records that have no API number to the records of a
regulatory database (or any other well database) by probabilistic record linkage with a
calibrated output. Each proposed link carries a probability of being correct; links are
assigned, provisional or withheld from that probability alone, and every withheld record
keeps the reason (missing records or conflicting records).

The method is described in: Risha, M., et al. (in review). *Linking legacy oil and gas well
records without API numbers to state regulatory databases: a validated, confidence-tiered
method tested in the Permian Basin.*

## What it does

1. **fetch** — snapshots every candidate record within a search radius of each legacy record,
   from ArcGIS REST well layers (queried automatically) or CSV exports (any provider), into one
   common record format.
2. **split** — divides the legacy records that already carry an API into a tuning set and a
   locked test set, stratified by state and completion decade.
3. **fit** — estimates Fellegi–Sunter agreement weights (log₂ m/u) for nine graded evidence
   fields on the tuning set, and a logistic calibration from score to probability.
4. **evaluate** — scores the test set once: coverage, accuracy with Wilson intervals,
   calibration by probability band, and the reasons for unresolved records.
5. **apply** — links the records without an API, with one-to-one safeguards.
6. **reconcile** — one status per record across several sources.

Evidence used: distance, well-number agreement (with variant and typo handling), lease-name
similarity, spud-year and depth agreement, same-number rivals nearby, an unnumbered record
lying closer, placeholder/dummy identifiers, and co-located records. Hard filters remove
candidates with an incomplete identifier, another county, the wrong era, a never-drilled
permit, or beyond the assignment limit.

## Install

```
pip install -e .
```

Requires Python 3.10+. Dependencies: numpy, pandas, scipy, scikit-learn, rapidfuzz, requests, pyyaml.

## Use

```
cp config.example.yaml config.yaml      # edit: legacy table columns, states, sources
tall fetch                               # snapshot candidate records (once)
tall split
tall fit
tall evaluate
tall apply
tall reconcile
```

Transfer a fitted model to another source without refitting:

```
tall evaluate --source vendor --weights-from rrc
```

Outputs are written to `workdir`: `cache_<state>_<source>.json.gz` (snapshot),
`split.csv`, `model_<state>_<source>.json` (weights and calibration),
`eval_<state>_<source>.csv`, `links_<state>_<source>.csv`, `links_final.csv`.

## Legacy table

A CSV with one row per record: an id, state, county, a well name that combines lease name
and well number ("Smith A No. 3"), latitude and longitude, and optionally completion year,
total depth and an API (records that carry one form the benchmark; records without one are
the targets). Column names are set in the config.

## Adding a state or source

Add a `states` entry with its two-digit API code, the sequence ranges that cannot belong to
each completion era (`vintage_bands`), any placeholder sequence range (`dummy_from`), and one
or more sources with their field names. Weights and calibration are estimated per state and
per source; a labelled subset of records is needed for each.

## Reproducibility

Snapshots are written atomically and reused on later runs; the split uses a fixed seed; weights
and calibration are deterministic. Keep the snapshot with the results you report.

## License

MIT — see LICENSE.
