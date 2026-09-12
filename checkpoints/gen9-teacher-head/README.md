# Gen9 Teacher-Head production checkpoint

Policy-head-only search-teacher distillation candidate promoted from frozen run `34653712220`.

Promotion evidence: 100 paired openings / 200 games at 200 ms, workers=1, tablebases disabled, against Gen8. Result: 104 wins, 84 losses, 12 draws; 55.0% score; +34.86 Elo. Pair-aware bootstrap 95% interval: score 49-61%, Elo -6.95 to +77.71. Promotion was explicitly approved based on the positive point estimate despite the interval crossing zero.

The float and int8 files here are the exact candidate weights used in the promotion arena.
