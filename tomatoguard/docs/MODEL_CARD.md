# Model card - TomatoGuard AI RGB classifier (fill in after running Step 3)

**Intended use:** research prototype / decision support for tomato leaf condition screening. Not a diagnostic tool.
**Not intended for:** pre-symptomatic detection, ToBRFV confirmation, or any agronomic/treatment decision.

| Item | Value |
|---|---|
| Architecture | (from runs/<run>/meta.json) |
| Training data | Mendeley "Tomato Leaf Disease Classification" (jttrv2w27r), 7 classes |
| Split | near-duplicate-group-disjoint, class-stratified (NOT plant-disjoint) |
| Test metrics | (from runs/<run>/eval.json: accuracy + CI, macro-F1 + CI, ECE) |
| Calibration | temperature scaling, T = (meta.json) |
| Uncertainty rule | prediction shown as "uncertain" below confidence threshold (model_meta.json) |
| Size / latency | (benchmarks/*) |

**Known weak classes:** (copy from classification_report.csv - the paper's baseline was weakest on Early Blight and Bacterial Spot.)
**Known limitations:** single public dataset; lab-style images; no plant IDs; no field validation; no multispectral input on phone.
