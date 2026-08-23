# F. Correlation & Relationship Analysis

## Recovery Status × Compliance
| recovery_status   |   compliant |   non_compliant |   partial |
|:------------------|------------:|----------------:|----------:|
| fully_recovered   |         159 |               6 |         0 |
| improving         |           2 |               0 |         1 |
| unresolved        |           8 |               3 |         0 |
| worsening         |           1 |               1 |         0 |

## Recovery Status × Urgency
| recovery_status   |   high |   low |   medium |
|:------------------|-------:|------:|---------:|
| fully_recovered   |     40 |   550 |        1 |
| improving         |      0 |     6 |        0 |
| unresolved        |      2 |     8 |        3 |
| worsening         |      2 |     5 |        0 |

## Recovery × Follow-Up Required
| recovery_status   |   No Follow-up |   Follow-up Needed |
|:------------------|---------------:|-------------------:|
| fully_recovered   |            239 |                352 |
| improving         |              6 |                  0 |
| unresolved        |              5 |                  8 |
| worsening         |              5 |                  2 |

## Compliance × Follow-Up Required
| patient_compliance   |   No Follow-up |   Follow-up Needed |
|:---------------------|---------------:|-------------------:|
| compliant            |            164 |                101 |
| non_compliant        |             13 |                  4 |
| not_applicable       |            147 |                261 |
| partial              |              2 |                  0 |
| unknown              |            114 |                 81 |

## Spearman Correlation Matrix
|                    |   conversation_type |   call_outcome |   urgency_level |   recovery_status |   patient_compliance |
|:-------------------|--------------------:|---------------:|----------------:|------------------:|---------------------:|
| conversation_type  |               1     |         -0.091 |           0.142 |            -0.139 |                0.005 |
| call_outcome       |              -0.091 |          1     |           0.079 |            -0.006 |               -0.11  |
| urgency_level      |               0.142 |          0.079 |           1     |             0.055 |                0.12  |
| recovery_status    |              -0.139 |         -0.006 |           0.055 |             1     |               -0.097 |
| patient_compliance |               0.005 |         -0.11  |           0.12  |            -0.097 |                1     |
