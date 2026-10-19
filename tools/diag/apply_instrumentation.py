#!/usr/bin/env python3
"""Apply identical DIAGNOSTIC-ONLY UCI/MCGS counters to any checked-out ref.

This changes neither selection nor evaluation in ordinary game mode.
--diag-fixed-sims uses a precisely counted MCGS simulation cap and is
never passed to a native-clock game. Strict anchors fail closed.
"""
from pathlib import Path
import sys

def one(old, before, after, label):
    n=old.count(before)
    if n!=1: raise RuntimeError(f"{label}: expected one anchor, got {n}")
    return old.replace(before,after,1)

def last(old, before, after, label):
    index=old.rfind(before)
    if index < 0: raise RuntimeError(f'{label}: last occurrence missing')
    return old[:index]+after+old[index+len(before):]

def patch(root):
    r=Path(root)
    path=r/"src/mcab.hpp"
    s=path.read_text(encoding="utf8")
    s=one(s,"    int nodeBudget = 20000;", "    int nodeBudget = 20000;\n    int diagnosticSimLimit = 0; // 0 is ordinary production search", "sim-param")
    s=one(s,"    bool adaptiveTimeStop = false;", "    bool adaptiveTimeStop = false;\n    int diagnosticStopReason = 0; // 1=time,2=adaptive,3=nodecap,4=terminal,5=simcap", "stop field")
    s=one(s,"if (elapsedMs >= treeBudgetMs) break;", "if (elapsedMs >= treeBudgetMs) { mstats.diagnosticStopReason=1; break; }", "walltime")
    s=one(s,"                            mstats.adaptiveTimeStop = true;\n                            break;", "                            mstats.adaptiveTimeStop = true;\n                            mstats.diagnosticStopReason=2;\n                            break;", "adaptive")
    s=one(s,"            if (mstats.nodesExpanded >= budget) break;\n            if (pool[0].terminal) break;", "            if (mstats.nodesExpanded >= budget) { mstats.diagnosticStopReason=3; break; }\n            if (pool[0].terminal) { mstats.diagnosticStopReason=4; break; }\n            if (params.diagnosticSimLimit > 0 && mstats.simulations >= params.diagnosticSimLimit) { mstats.diagnosticStopReason=5; break; }", "loop cap")
    path.write_text(s,encoding="utf8")
    path=r/"tools/external/zquoridor_uci.cpp"
    u=path.read_text(encoding="utf8")
    u=one(u,"    bool dumpRoot = false;","    bool dumpRoot = false;\n    bool diagNoAdaptiveTime = false;\n    int diagFixedSims = 0;", "UCI options")
    u=one(u,'        else if (a == "--dump-root") o.dumpRoot = true;', '        else if (a == "--dump-root") o.dumpRoot = true;\n        else if (a == "--diag-no-adaptive-time") o.diagNoAdaptiveTime = true;\n        else if (a == "--diag-fixed-sims") o.diagFixedSims = std::max(0, std::atoi(need("--diag-fixed-sims")));', "CLI args")
    u=one(u,"    params.enabled = opt.useMcab;","    params.enabled = opt.useMcab;\n    params.diagnosticSimLimit = opt.diagFixedSims;", "UCI params")
    u=one(u,"            const bool adaptiveClock = movetime <= 0 &&\n                (state.turn == 0 ? wtime : btime) >= 0;", "            const bool adaptiveClock = movetime <= 0 &&\n                (state.turn == 0 ? wtime : btime) >= 0 && !opt.diagNoAdaptiveTime;", "clock adaptive")
    # Keep the *initial expansion cap* matched to native adaptive ON.
    # Otherwise OFF uses effectiveNodeBudget(optimumMs) while ON uses
    # effectiveNodeBudget(maximumMs); that changes TWO independent controls.
    # The wall-clock limit remains optimumMs in OFF.  Volatile escalation is
    # itself an adaptive mechanism and remains OFF by design.
    u=one(u,'            if (adaptiveClock) budgetMs = timeBudget.maximumMs;',
          '''            if (adaptiveClock) budgetMs = timeBudget.maximumMs;
            if (opt.diagNoAdaptiveTime && movetime <= 0 &&
                (state.turn == 0 ? wtime : btime) >= 0) {
                // Preserve the native-clock initial node cap in the fixed-time
                // diagnostic. Do not recompute it from optimumMs.
                runner.params().nodeBudget =
                    mcab::effectiveNodeBudget(params, timeBudget.maximumMs);
                runner.params().autoNodeBudget = false;
            }''', "fixed mode matching initial node cap")

    u=last(u,'                      << " treeHit=" << (mstats.treeReused ? 1 : 0)', '''                      << " diag_sims=" << mstats.simulations
                      << " diag_nodes=" << mstats.nodesExpanded
                      << " diag_target=" << mstats.adaptiveTargetMs
                      << " diag_class=" << static_cast<int>(mstats.adaptiveNodeRootClass)
                      << " diag_stop=" << mstats.diagnosticStopReason
                      << " diag_adaptive=" << (runner.params().adaptiveTime ? 1 : 0)
                      << " diag_optimum=" << timeBudget.optimumMs
                      << " diag_visits=" << mstats.adaptiveVisitRatio
                      << " diag_qgap=" << mstats.adaptiveQGap
                      << " diag_expandcap=" << mstats.effectiveNodeBudget
                      << " diag_simcap=" << runner.params().diagnosticSimLimit
                      << " treeHit=" << (mstats.treeReused ? 1 : 0)''', "UCI metrics")
    path.write_text(u,encoding="utf8")
    print(f"INSTRUMENTED {r}: clock metrics, 5 stop reasons, exact sim cap; ordinary search unchanged",flush=True)

if __name__=="__main__":
    if len(sys.argv)<2: raise SystemExit("usage: apply_instrumentation.py checkout ...")
    for target in sys.argv[1:]: patch(target)
