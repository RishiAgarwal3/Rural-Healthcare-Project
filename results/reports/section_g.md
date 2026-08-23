# G. Data Quality Analysis

## Manual Review Flags
- Calls needing manual review: **751** / 887 (84.7%)

## Extraction Warning Types
- `PATIENT_NAME_NOT_FOUND`: 703
- `RECOVERY_STATUS_UNKNOWN`: 84

## Confidence Score Analysis
| field           |   low_conf_count (<0.5) |   zero_conf_count |   mean_conf |
|:----------------|------------------------:|------------------:|------------:|
| patient_name    |                     703 |               703 |       0.183 |
| recovery_status |                      84 |                84 |       0.644 |
| compliance      |                     195 |               195 |       0.507 |

## Schema Completeness
- 🟢 Green = >80%  |  🟡 Yellow = 50–80%  |  🔴 Red = <50%

## Audio Quality Flags
- Calls with >50% flagged segments: **0**
- Mean flagged ratio: **0.0000**
