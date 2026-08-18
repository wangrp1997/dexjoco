# Privileged PBVS

## 现状（诚实）

- 协议：`peg_lift_end` 后硬 o2h snap + 虚拟 tip PBVS，**无 demo 续放**
- 先验 1–8：**7/8**，仅 **ep5** 卡在 tip≈3cm（旧）
- 全量 100 旧目视：**95/100**。真失败 **51 / 62 / 74 / 95 / 98**（pinch 脱手）
- Handoff 抓取：成功/失败都是 **食指+拇指 pinch**，掌心/中指/无名指接触=0。中指离钉轴约 6cm，无名指约 9.5cm
- 失败条 handoff **侧向更大**（约 5.5cm vs 成功 2.7cm），不是另一种抓法
- 把钉子往掌心/中指挪会丢掉拇指，包络抓这条路目前走不通
- 现：PBVS 对准孔沿（lat≤1.2cm, tip≤3.5cm）后 **松手再特权 pin 进孔**。机器 51/62/74/95/98/7/32 均过；片子待目视
- 片子：`videos/fails_all100/`（旧目视）。新：`videos/transfer_seat/`（先看再信）

## 跑法

```bash
python -m priv_snap_insert.run_p0 --episodes 1 2 3 4 5 6 7 8
python -m priv_snap_insert.run_p0 --all
```
