#pragma once

// Zquoridor 4.0 uses the 858-input contact network and dense edge deltas.
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
