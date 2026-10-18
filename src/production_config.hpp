#pragma once

// Zquoridor 4.02 uses the 858-input temporal network and dense edge deltas.
// Explicit build flags select archived architectures and search profiles.
#ifndef ZQ_NNUE_CONTACT_FEATURES
#define ZQ_NNUE_CONTACT_FEATURES 1
#endif
#ifndef ZQ_EXP_EDGE_FEATURE_DELTA
#define ZQ_EXP_EDGE_FEATURE_DELTA 1
#endif
#ifndef ZQ_EXP_EDGE_DENSE_DELTA
#define ZQ_EXP_EDGE_DENSE_DELTA 1
#endif
#ifndef ZQ_EXP_EDGE_BFS_REPLAY
#define ZQ_EXP_EDGE_BFS_REPLAY 1
#endif

// Release 4.02: enable both proven MCGS optimizations by default in ALL
// production targets, not only explicit benchmark -D compile invocations.
// Setting -DZQ_EXP_SPARSE_POLICY=0 allows forensic baseline comparisons.
#ifndef ZQ_EXP_SPARSE_POLICY
#define ZQ_EXP_SPARSE_POLICY 1
#endif
