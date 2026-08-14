# Observability Privileged Label Smoke (P0-L1)

- 日期：2026-08-14T13:27:39Z
- 结论：**pass**
- episodes：[0, 2, 4]；window=8
- bit-exact repeat：True
- snapshot restore labels：True
- timeline contiguous：True
- 未训练、未采集、未开写盘、未重开 C0/C1/C1.1
- schema：`PRIVILEGED_LABEL_SCHEMA_V1.md`
- manifest：`data/manifests/observability_privileged_label_smoke_v1.json`

## Per episode

- ep0 root=transport@302 frames=[302, 303, 304, 305, 306, 307, 308, 309] digest=06d94cd31723… contact0=4
- ep2 root=transport@281 frames=[281, 282, 283, 284, 285, 286, 287, 288] digest=f6c11ef4db37… contact0=4
- ep4 root=transport@250 frames=[250, 251, 252, 253, 254, 255, 256, 257] digest=8b3206c88734… contact0=4

## Claims

- 冻结特权标签契约（pose/velocity/contact/force/outcome/provenance）
- **不**声称 Observability 模型可训或部署 belief 已解
- **不**生成 slip truth / fine contact-mode
