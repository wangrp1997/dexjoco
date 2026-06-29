# ContactOpt (ported subset)

Source: `refs/contactopt` (MIT, Facebook Research)

Mirror: `interaction_retarget/contactopt/contactopt/`

| Original | Port |
|----------|------|
| `contactopt/diffcontact.py` | `contactopt/diffcontact.py` (numpy) |
| `contactopt/diffcontact.py` (`calculate_penetration_cost`) | `contactopt/penetration.py` |
| `contactopt/optimize_pose.py` (contact-map L1) | `contactopt/contact_map_loss.py` |

Flat re-export: `interaction_retarget/contactopt/*.py`
