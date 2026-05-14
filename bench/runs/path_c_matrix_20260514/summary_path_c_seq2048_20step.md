| dtype | optimizer | state | status | steps | steps/s | tok/s | mean step s | peak GB | loss first->final | aliases |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | adamw | 16 | ok | 20 | 0.1071 | 220.78 | 9.341 | 32.53 | 11.263->6.546 |  |
| bf16 | muon_adamw | 16 | ok | 20 | 0.0682 | 139.86 | 14.658 | 28.45 | 11.263->5.732 | muon |
| bf16 | lion | 16 | ok | 20 | 0.1095 | 225.26 | 9.132 | 33.75 | 11.263->5.269 |  |
| bf16 | adam8bit | 8 | ok | 20 | 0.0296 | 60.96 | 33.744 | 35.74 | 11.263->6.334 | adamw_int8_state |
| bf16 | muon_adamw_int8 | 8 | ok | 20 | 0.0421 | 86.98 | 23.769 | 26.73 | 11.263->5.741 | muon_int8,muon_adamw_int8_state |
| bf16 | lion8bit | 8 | ok | 20 | 0.0540 | 112.40 | 18.509 | 30.70 | 11.263->5.264 | lion_int8_state |
| fp8_path_c | adamw | 16 | ok | 20 | 0.1092 | 224.25 | 9.158 | 32.53 | 11.263->6.600 |  |
| fp8_path_c | muon_adamw | 16 | ok | 20 | 0.0686 | 140.50 | 14.587 | 28.45 | 11.263->5.730 | muon |
| fp8_path_c | lion | 16 | ok | 20 | 0.1083 | 222.34 | 9.232 | 33.75 | 11.263->5.282 |  |
| fp8_path_c | adam8bit | 8 | ok | 20 | 0.0332 | 68.87 | 30.081 | 35.74 | 11.263->5.889 | adamw_int8_state |
| fp8_path_c | muon_adamw_int8 | 8 | ok | 20 | 0.0461 | 96.65 | 21.675 | 26.73 | 11.263->5.688 | muon_int8,muon_adamw_int8_state |
| fp8_path_c | lion8bit | 8 | ok | 20 | 0.0419 | 86.59 | 23.869 | 30.70 | 11.263->5.276 | lion_int8_state |
