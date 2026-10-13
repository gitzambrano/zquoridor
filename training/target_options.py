"""Resolve independent value-target settings without loading training libraries."""
import math


def target_settings(mode, horizons=None, weights=None, outcome_weight=.1):
    if mode not in ('current','temporal'):
        raise ValueError('Value targets must use current or temporal search values.')
    if horizons is None:
        horizons=[0] if mode=='current' else [0,4,8,16]
    if not horizons or any(type(h) is not int or h<0 for h in horizons) or horizons!=sorted(set(horizons)) or horizons[0]!=0:
        raise ValueError('Horizons must be unique increasing nonnegative plies that start at zero.')
    if mode=='current' and horizons!=[0]:
        raise ValueError('Current targets cannot include future horizons.')
    if weights is None:
        weights=[1.] if mode=='current' else [.5**(i+1) for i in range(len(horizons))]
    if len(weights)!=len(horizons) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('Each horizon requires a finite positive weight.')
    if not math.isfinite(outcome_weight) or not 0<=outcome_weight<=1:
        raise ValueError('The outcome weight must be between zero and one.')
    return dict(target_mode=mode,horizons=list(horizons),horizon_weights=list(weights),outcome_weight=outcome_weight)


def matches_targets(manifest,settings):
    stored=manifest.get('config',{})
    return all(stored.get(key)==settings[key] for key in ('horizons','horizon_weights','outcome_weight'))
