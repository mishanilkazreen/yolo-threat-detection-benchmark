# Companion Repository Audit Report (Task 3 Integrity Verification)

- **Total Factorial Runs Audited**: 60
- **Expected Round Metrics Files**: 300
- **Found Round Metrics Files**: 300 (100% complete)
- **Mathematical Consistency Passed Runs**: 60 / 60

## Data Splits Verification
- `train_init.txt`: Expected 709, Actual 709
- `unlabeled_pool.txt`: Expected 2836, Actual 2836
- `val_fixed.txt`: Expected 1013, Actual 1013
- `test_fixed.txt`: Expected 506, Actual 506

## Evolution File Structure & Logging Artifacts
- Total evolution JSON files audited: 60
- 5-entry layout (Rounds 2-5 + duplicate final): 0 files
- 6-entry layout (Rounds 1-5 + duplicate final): 0 files
- **Audit Ruling**: Raw outputs are preserved non-destructively per AGENTS.md Section 3. Ledger generation script (scripts/generate_revision_ledgers.py) parses round_N_metrics.json directly.
