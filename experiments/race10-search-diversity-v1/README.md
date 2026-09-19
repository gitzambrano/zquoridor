# Diverse 50k search-teaching selection

The current search fine-tune gives roughly 10% of effective loss mass to a
small set of expensive search rows. This branch keeps disagreement mining but
reduces concentration risk.

Selection:

- candidate pool: the historical 2M direct replay;
- hardness score: policy Jensen-Shannon divergence + absolute value disagreement;
- target: 50,000 unique states;
- 70% of slots are allocated across 12 strata:
  `ahead/tie/behind × both-have-walls/own-zero/opp-zero/both-zero`;
- unused/remaining slots are filled globally by hardness score.

This branch only selects positions. The expensive ZQuoridor/Claustrophobia
search relabeling remains a separate step so its compute and manifests stay
auditable.
