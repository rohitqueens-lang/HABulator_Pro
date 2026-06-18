"""
gen_selftest_golden.py — Freeze the expected /predict output for SELFTEST_INPUT,
per group, into selftest_golden.json. The startup self-test (main._run_selftest)
asserts the live predict path reproduces these values, catching artifact/code drift.

Run after any *intentional* model update:
    cd habulator-api && ./venv/bin/python gen_selftest_golden.py

Provenance: values come from the deployed artifacts via main._predict_core (the exact
production prediction path), which is independently verified by repo-root
verify_wiring.py (artifact == trained emulator, bit-identical) and verify_api_match.py
(API == independent computation, all groups). So golden == verified production output.
"""
import json
import main

main._load_artifacts()
INP = main.SELFTEST_INPUT
golden = {
    "_note": "Frozen expected outputs for the deployed artifacts; predict path verified "
             "by verify_wiring.py & verify_api_match.py. Regenerate on intentional model update.",
    "input": INP,
    "tolerance": {"atol": main.SELFTEST_ATOL, "rtol": main.SELFTEST_RTOL},
    "groups": {g: main._predict_core(st, **INP) for g, st in main._models.items()},
}
main.GOLDEN_PATH.write_text(json.dumps(golden, indent=2))
print(f"wrote {main.GOLDEN_PATH.name} for groups: {list(golden['groups'])}")
for g, v in golden["groups"].items():
    print(f"  {g:6s} pred={v['pred_mgL']:.6f}  lower={v['lower_mgL']:.6f}  "
          f"upper={v['upper_mgL']:.6f}  base={v['base_val']:.6f}")
