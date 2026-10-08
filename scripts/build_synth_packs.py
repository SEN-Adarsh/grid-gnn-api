"""Build the WP-13 demo packs deterministically into data/synth_v2/<pack>/.

Freeze discipline (WP-13 Section 1.1): pack configs and seeds are committed
BEFORE this script's outputs are used in any model evaluation. Eval seeds
(4100xx) are never used for generator development.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import json  # noqa: E402
import numpy as np  # noqa: E402
from src.common import load_config, config_hash  # noqa: E402
from src.sim.synth_v2 import simulate_v2, export_pack  # noqa: E402

STORIES = {
    'P1_hooking_dt': 'One DT carries hidden hooking; every meter looks normal. DT-level balance finds what meter-level models cannot.',
    'P2_night_theft_cluster': 'Several neighbours on one DT bypass supply in the evening. Meter ranking within a flagged DT.',
    'P3_honest_fault_trap': 'Faulty meters and vacant homes on high-deficit DTs, no theft at all. False-accusation controls.',
    'P4_solar_behaviour_confounder': 'Daytime import drops for honest reasons. A known limitation, shown openly.',
    'P5_comms_outage_storm': 'Heavy missing data and a feeder outage. Robustness and the limits of imputation.',
    'P6_mixed_realistic': 'All families and confounders at default prevalence. The headline demo run.',
    'P7_unseen_family': 'Models trained without t5/t6/t7 face them here. The generalisation gap, shown openly.',
    'seasonal_365': 'A full year for the temporal/seasonal hold-out study.',
}


def build(config_rel):
    cfg = load_config(ROOT / 'configs' / config_rel)
    pack = cfg['pack']
    seed = int(cfg['seed_eval'])
    t0 = time.time()
    obs, hidden = simulate_v2(cfg, seed)
    dest = ROOT / 'data' / 'synth_v2' / pack
    meta = export_pack(obs, hidden, dest)
    lm = hidden.labels_meter
    theft = lm[lm.is_theft]
    card = {
        'pack': pack,
        'story': STORIES.get(pack, ''),
        'profile_source': 'SYNTHETIC',
        'generator': 'synth_v2 (WP-13); every parameter is an assumption, not an Indian measurement',
        'seed_eval': seed,
        'config_hash': config_hash(obs.cfg),
        'config': config_rel,
        'n_meters': int(len(lm)),
        'n_dts': int(obs.dt_kwh.shape[0]),
        'days': int(obs.dt_kwh.shape[1] // 96),
        'achieved_meter_prevalence': float(lm.is_theft.mean()),
        'theft_family_counts': {k: int(v) for k, v in theft.family.value_counts().items()},
        'honest_archetype_counts': {k: int(v) for k, v in
                                    lm[lm.honest_archetype != ''].honest_archetype.value_counts().items()},
        'three_phase_meters': int(obs.meters.three_phase.sum()),
        't3_rule': 'hooking (t3) is ground-truth theft but never a meter-level positive; DT-level only',
        'label_access': 'evaluation only; ground_truth*.parquet never read by the feature pipeline',
        'ts_note': 'ts columns are 15-minute interval indices from 2026-06-01 00:00 Asia/Kolkata',
        'runtime_s': round(time.time() - t0, 1),
        **meta,
    }
    (dest / 'data_card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    print(f"{pack}: {card['n_meters']} meters, prevalence {card['achieved_meter_prevalence']:.3f}, "
          f"families {card['theft_family_counts']}, {card['runtime_s']}s", flush=True)
    return card


if __name__ == '__main__':
    targets = sys.argv[1:] or ['packs/P1.yaml', 'packs/P2.yaml', 'packs/P3.yaml',
                               'packs/P4.yaml', 'packs/P5.yaml', 'packs/P6.yaml',
                               'packs/P7.yaml', 'synth_v2_seasonal.yaml']
    for rel in targets:
        build(rel)
    print('ALL PACKS BUILT', flush=True)
