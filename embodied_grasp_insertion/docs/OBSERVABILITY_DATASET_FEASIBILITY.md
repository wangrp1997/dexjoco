# Observability Dataset Feasibility (P0-Obs-D0)

- 日期：2026-08-14T16:41:40Z
- overall_verdict：**feasibility_pass**
- episodes scanned：100（现有 sidecar，只读）
- bit-exact repeat（3-ep re-scan）：True
- split leakage ok：True
- roots：200（每 ep early_grasp+transport；zero_roots=0）
- root 接触：200/200 非零（min=2，mean≈4.5）
- root-anchored windows：H1/H4/H8/H16 各 200
- phase-contiguous windows：H1=3700，H4=3400，H8=3000，H16=2200
- exclusions：frame_gaps=0，early_terminate=0，label_invalid=0；`no_contact_at_recorded=23947` 为扫描段内**含抓前**零接触帧计数，不表示 root 无效
- split：train/val/test = 70/15/15（episode 原子；seed=20260814）
- 部署 A `act44`：shape[44] float64，missing_rate=0，frames_sum=30913
- 部署 B `ft12`：shape[12] float64，missing_rate=0，frames_sum=30913
- **claims_observability_p0_pass=false**（单几何，无 geometry held-out）
- 未训练、未采集、未开写盘、未写正式 pilot、未重开 C0/C1/C1.1
- manifest：`data/manifests/observability_dataset_feasibility_v1.json`
- /tmp sample（≤3 ep）：`/tmp/obs_d0_pack_sample/sample_pack.json`

## Decision branch

- **字段和覆盖（单几何）足够**：可审批**完整只读评测包**导出（须另授权）；仍不得称 Obs P0 通过。
- 缺失严重：否（A/B 无缺失；标签可派生；root 全有接触）。
- 即使可导出：训练 Observability 基线仍须**单独授权**。

## Non-claims

- 非多几何 / 非 geometry-held-out  
- 非部署 tactile（档 C）已解决  
- 非训练数据集  
