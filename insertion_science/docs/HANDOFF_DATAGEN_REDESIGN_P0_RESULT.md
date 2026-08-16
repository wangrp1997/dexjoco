# Handoff Datagen Redesign P0 — Result

- UTC: `2026-08-15T17:06:59Z`
- Protocol: `HandoffDatagenRedesignP0`
- Verdict: `fail_no_contrast`
- Decision: `pause_boundary_not_operational`
- Reason: in=0.917 out=0.375 gap=0.542 checks={'in_basin_rate_ok': True, 'out_basin_rate_ok': False, 'gap_ok': True, 'min_accepted_ok': True}

## Rates

- In-basin insert_ok: `0.917` (n=24)
- Out-basin insert_ok: `0.375` (n=8)
- Gap: `0.542`
- Accepted: `22`

## Generation limits (conservative × safety)

- `tip_lat`: `0.25`
- `tip_along`: `0.5`
- `axis`: `0.0`
- `o2h`: `0.5`
- `finger`: `1.0`

## Coverage recheck (archived fails vs accepted)

- outside_frac: `0.9090909090909091` (n=11, outside=10)

## Gates

- in_basin_min_rate: `0.75`
- out_basin_max_rate: `0.35`
- min_rate_gap: `0.3`
- min_accepted: `16`
- checks: `{'in_basin_rate_ok': True, 'out_basin_rate_ok': False, 'gap_ok': True, 'min_accepted_ok': True}`

## Note

不训练；不写回生产 sidecar；旧失败仍应在盆地外（预期）。
