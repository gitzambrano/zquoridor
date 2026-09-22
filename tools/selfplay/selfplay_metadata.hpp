#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace qr {

// A missing root value is valid only when this flag is set.
constexpr uint8_t TRAINING_META_ROOT_VALUE_MISSING = 1u << 0;
constexpr uint8_t TRAINING_META_ROOT_VALUE_VALID = 1u << 1;

#pragma pack(push, 1)
struct TrainingMetaV1 {
    uint64_t gameId;
    float rootValue;
    uint16_t pliesToEnd;
    uint16_t gameLength;
    uint8_t sourceClass;
    uint8_t qualityFlags;
    uint16_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(TrainingMetaV1) == 20,
              "TrainingMetaV1 must stay packed at 20 bytes");
static_assert(offsetof(TrainingMetaV1, gameId) == 0,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, rootValue) == 8,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, pliesToEnd) == 12,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, gameLength) == 14,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, sourceClass) == 16,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, qualityFlags) == 17,
              "TrainingMetaV1 field order changed");
static_assert(offsetof(TrainingMetaV1, reserved) == 18,
              "TrainingMetaV1 field order changed");

inline bool trainingMetaRootValueIsValid(const TrainingMetaV1& meta) {
    return (meta.qualityFlags & TRAINING_META_ROOT_VALUE_MISSING) == 0
        && std::isfinite(meta.rootValue)
        && meta.rootValue >= 0.0f
        && meta.rootValue <= 1.0f;
}

inline TrainingMetaV1 makeTrainingMeta(uint64_t gameId, float rootValue,
                                       uint8_t sourceClass = 0,
                                       uint8_t qualityFlags = 0) {
    TrainingMetaV1 meta{};
    meta.gameId = gameId;
    meta.sourceClass = sourceClass;
    meta.qualityFlags = qualityFlags;
    if (std::isfinite(rootValue) && rootValue >= 0.0f && rootValue <= 1.0f) {
        meta.rootValue = rootValue;
        meta.qualityFlags = (uint8_t)(meta.qualityFlags & ~TRAINING_META_ROOT_VALUE_MISSING);
        meta.qualityFlags = (uint8_t)(meta.qualityFlags | TRAINING_META_ROOT_VALUE_VALID);
    } else {
        meta.rootValue = std::numeric_limits<float>::quiet_NaN();
        meta.qualityFlags = (uint8_t)(meta.qualityFlags | TRAINING_META_ROOT_VALUE_MISSING);
        meta.qualityFlags = (uint8_t)(meta.qualityFlags & ~TRAINING_META_ROOT_VALUE_VALID);
    }
    return meta;
}

template <typename Visits, typename QValues>
inline float rootMeanValue(const Visits& visits, const QValues& qValues) {
    double weighted = 0.0;
    double total = 0.0;
    const size_t count = std::min(visits.size(), qValues.size());
    for (size_t i = 0; i < count; ++i) {
        const double visitCount = (double)visits[i];
        const double q = (double)qValues[i];
        if (!(visitCount > 0.0) || !std::isfinite(q)) continue;
        weighted += visitCount * q;
        total += visitCount;
    }
    if (!(total > 0.0)) return std::numeric_limits<float>::quiet_NaN();
    return (float)std::max(0.0, std::min(1.0, weighted / total));
}

template <typename Node, typename = void>
struct has_q_values : std::false_type {};

template <typename Node>
struct has_q_values<Node, std::void_t<decltype(std::declval<const Node&>().q)>>
    : std::true_type {};

template <typename Node, typename = void>
struct has_Q_values : std::false_type {};

template <typename Node>
struct has_Q_values<Node, std::void_t<decltype(std::declval<const Node&>().Q)>>
    : std::true_type {};

template <typename Node, typename = void>
struct has_visits_values : std::false_type {};

template <typename Node>
struct has_visits_values<Node, std::void_t<decltype(std::declval<const Node&>().visits)>>
    : std::true_type {};

template <typename Node, typename = void>
struct has_N_values : std::false_type {};

template <typename Node>
struct has_N_values<Node, std::void_t<decltype(std::declval<const Node&>().N)>>
    : std::true_type {};

template <typename Node, typename = void>
struct has_W_values : std::false_type {};

template <typename Node>
struct has_W_values<Node, std::void_t<decltype(std::declval<const Node&>().W)>>
    : std::true_type {};

// Compute the visit-weighted mean for a node that exposes visits and Q values.
// The N/W form also supports the MCAB node used by the self-play generator.
template <typename Node>
inline float rootMeanValue(const Node& node) {
    if constexpr (has_visits_values<Node>::value && has_q_values<Node>::value) {
        return rootMeanValue(node.visits, node.q);
    } else if constexpr (has_visits_values<Node>::value && has_Q_values<Node>::value) {
        return rootMeanValue(node.visits, node.Q);
    } else if constexpr (has_N_values<Node>::value && has_W_values<Node>::value) {
        using QType = std::decay_t<decltype(node.W)>;
        QType qValues(node.N.size());
        for (size_t i = 0; i < node.N.size(); ++i) {
            qValues[i] = node.N[i] > 0.0f ? node.W[i] / node.N[i]
                                         : std::numeric_limits<float>::quiet_NaN();
        }
        return rootMeanValue(node.N, qValues);
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

}  // namespace qr
