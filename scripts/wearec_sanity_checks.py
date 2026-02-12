#!/usr/bin/env python3
"""
WEARec is run via the official repo (WEARec/) on TRACT-exported data, not as a RecBole model.

- Export data (same split as RecBole): python scripts/export_recbole_to_wearec.py --datasets ml-100k lastfm amazon-beauty
- Run WEARec: python scripts/run_wearec_official.py configs/<dataset>/<dataset>_wearec.yaml
- Or submit PBS: qsub pbs/ml-100k/wearec_ml100k_a100_ce.pbs (and lastfm, beauty)

The former RecBole WEARec port has been removed; this script no longer runs mapping/determinism checks.
"""
