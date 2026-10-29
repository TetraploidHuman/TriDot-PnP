# B0 failure diagnosis

## test

| factor | all-vis frames | det@10 | oracle det@10 | detect-miss | assoc-wrong |
|---|---|---|---|---|---|
| ALL | 1244 | 0.918 | 0.999 | 0.001 | 0.081 |
| blur | 221 | 1.000 | 1.000 | 0.000 | 0.000 |
| clean | 237 | 1.000 | 1.000 | 0.000 | 0.000 |
| distractor | 234 | 0.679 | 0.996 | 0.004 | 0.316 |
| light | 206 | 1.000 | 1.000 | 0.000 | 0.000 |
| mixed | 108 | 0.926 | 1.000 | 0.000 | 0.074 |
| noise | 238 | 0.920 | 1.000 | 0.000 | 0.080 |

Per-colour candidate stats (all-visible frames):

| colour | mean cands | p90 cands | near@10px | HSV pass | min-dist med (px) |
|---|---|---|---|---|---|
| red | 1.1 | 1 | 0.999 | 0.154 | 0.8 |
| green | 1.1 | 1 | 1.000 | 0.473 | 0.8 |
| blue | 1.1 | 1 | 1.000 | 0.164 | 0.8 |

missed-frame colours: {'red': 1} (empty candidate: {}, far only: {'red': 1}, HSV-fail: {'red': 1})

latency: propose 6.9 ms, detect 7.0 ms

## hard_test

| factor | all-vis frames | det@10 | oracle det@10 | detect-miss | assoc-wrong |
|---|---|---|---|---|---|
| ALL | 2527 | 0.245 | 0.469 | 0.531 | 0.224 |
| clutter | 360 | 0.078 | 0.572 | 0.428 | 0.494 |
| crosscolor | 345 | 0.223 | 0.484 | 0.516 | 0.261 |
| desk | 360 | 0.214 | 0.583 | 0.417 | 0.369 |
| far | 360 | 0.297 | 0.383 | 0.617 | 0.086 |
| handheld | 269 | 0.245 | 0.294 | 0.706 | 0.048 |
| hardmix | 162 | 0.000 | 0.259 | 0.741 | 0.259 |
| room | 337 | 0.780 | 0.810 | 0.190 | 0.030 |
| screen | 334 | 0.003 | 0.210 | 0.790 | 0.207 |

Per-colour candidate stats (all-visible frames):

| colour | mean cands | p90 cands | near@10px | HSV pass | min-dist med (px) |
|---|---|---|---|---|---|
| red | 3.5 | 5 | 0.609 | 0.250 | 1.6 |
| green | 3.0 | 5 | 0.769 | 0.630 | 1.1 |
| blue | 2.8 | 5 | 0.721 | 0.267 | 1.5 |

missed-frame colours: {'green': 584, 'blue': 704, 'red': 988} (empty candidate: {'red': 12, 'blue': 136, 'green': 38}, far only: {'green': 546, 'blue': 568, 'red': 976}, HSV-fail: {'blue': 612, 'red': 791, 'green': 347})

latency: propose 36.7 ms, detect 36.8 ms

## holdout_hard

| factor | all-vis frames | det@10 | oracle det@10 | detect-miss | assoc-wrong |
|---|---|---|---|---|---|
| ALL | 2631 | 0.231 | 0.416 | 0.584 | 0.184 |
| clutter | 339 | 0.029 | 0.404 | 0.596 | 0.375 |
| crosscolor | 360 | 0.133 | 0.483 | 0.517 | 0.350 |
| desk | 360 | 0.144 | 0.261 | 0.739 | 0.117 |
| far | 360 | 0.358 | 0.456 | 0.544 | 0.097 |
| handheld | 333 | 0.294 | 0.396 | 0.604 | 0.102 |
| hardmix | 167 | 0.006 | 0.419 | 0.581 | 0.413 |
| room | 352 | 0.761 | 0.787 | 0.213 | 0.026 |
| screen | 360 | 0.008 | 0.128 | 0.872 | 0.119 |

Per-colour candidate stats (all-visible frames):

| colour | mean cands | p90 cands | near@10px | HSV pass | min-dist med (px) |
|---|---|---|---|---|---|
| red | 3.5 | 5 | 0.583 | 0.274 | 2.0 |
| green | 3.1 | 5 | 0.775 | 0.655 | 1.1 |
| blue | 2.7 | 5 | 0.694 | 0.292 | 1.6 |

missed-frame colours: {'red': 1098, 'blue': 806, 'green': 592} (empty candidate: {'blue': 140, 'red': 13, 'green': 50}, far only: {'red': 1085, 'blue': 666, 'green': 542}, HSV-fail: {'red': 915, 'blue': 684, 'green': 283})

latency: propose 40.0 ms, detect 40.1 ms
