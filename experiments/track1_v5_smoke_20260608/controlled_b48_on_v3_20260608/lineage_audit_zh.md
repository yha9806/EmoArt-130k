# Track1 Controlled-B48 on V3 Lineage Audit

| package | count | same_as_current | same_as_v3_gate7 | same_as_full1000 | changed_vs_current | changed_vs_v3 |
|---|---:|---:|---:|---:|---:|---:|
| current | 1000 |  | 0 | 0 | 0 | 1000 |
| v3_gate7 | 1000 | 0 |  | 993 | 1000 | 0 |
| full1000_no_fallback | 1000 | 0 | 993 |  | 1000 | 7 |
| v5_accept48_old_current_base | 1000 | 952 | 0 | 0 | 48 | 1000 |
| controlled_b48_on_v3 | 1000 | 0 | 952 | 948 | 1000 | 48 |

## Interpretation

- `controlled_b48_on_v3` should not be old-current based.
- Expected: `same_as_current` near 0, `same_as_v3_gate7` = 952, `changed_vs_v3_gate7` = 48.
