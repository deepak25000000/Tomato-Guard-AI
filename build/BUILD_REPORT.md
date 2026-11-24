# BUILD_REPORT

Generated: 2026-11-22T10:44:00Z

## Phase status

| Phase | Status | Key fields |
|---|---|---|
| 0 | done | {'artifacts': ['build/PLAN.md', 'build/run_log.md', 'build/config.json', 'build/ui_strings.json']} |
| 1 | done | {'threshold': 0.56977624, 'test_acc': 0.6822429906542056, 'test_n': 428, 'onnx_pass': True, 'tflite_pass': True, 'tflite_max_diff': 3.4809112548828125e-05, 'golden_n': 50} |
| 2 | done | {'cells': 18, 'days_1_8_single_class_all_seeds': True} |
| 3 | code_ready_build_pending | {'flutter_toolchain': '~/toolchains (JDK17, Flutter 3.50.1, Android SDK)', 'apk': 'pending', 'real_phone_benchmark': 'PENDING'} |
| 4 | built_node_parity_pass | {'web_dist': True, 'node_parity_pass': True} |
| 5 | api_ready_trackb_exported | {'api_tests_pass': True, 'trackb_all_pass': False, 'trackb_models': 9} |
| 6 | in_progress | {'pytest': '31 passed, 0 skipped (prototype+results+api)'} |

## Pending

- Flutter pub get/analyze/test/build apk --release (running/pending in this environment)
- Real-phone benchmark (PENDING - no physical phone)
- Playwright browser golden parity / Lighthouse PWA+a11y (Node parity PASS)
- Deployment (needs explicit approval)

## Deviations / tolerance misses

- Track B ONNX: rgb seed42 logit max|Δ|=1.24e-4 > 1e-4 (retry no-constant-folding also FAIL); pred_vs_arch 2.57e-5 OK. Recorded in build/trackb_export.json.
- ai-edge-torch renamed to litert-torch; export script handles both. Export venv used CPU torch index to avoid CUDA wheels.
- Server restarts occurred; restart-proof skip-if-exists used where implemented.

## Verification evidence

- `tests/test_prototype.py` + `tests/test_results.py` + `api/tests`: 31 passed, 0 skipped.
- ONNX parity PASS (leaf): 3.28e-5 <= 1e-4. TFLite parity PASS: 3.48e-5 <= 1e-3.
- Web Node golden parity PASS: top1=1.0, mean|Δprob|=4.09e-7.
- Forbidden-phrase and number-traceability tests pass.
