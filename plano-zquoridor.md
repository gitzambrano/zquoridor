# Plano ZQuoridor

Este plano busca força mensurável contra Titanium e Claustrophobia. Cada fase
tem artefatos, critérios de entrada e uma verificação antes de promoção.

## Contrato único

Todo selfplay novo, treino NNUE e replay teaching usam
`data/selfplay_canonical_v3/`, no schema
`zquoridor.selfplay.v3.canonical.mover-mirrored.v1`.

- Registro V3: 64 bytes, tabuleiro e ação no referencial do jogador a mover.
- Dataset de teaching: `dataset.npz`; política é 209-probabilidades, valor é
  assinado em `[-1,+1]` na perspectiva do jogador a mover.
- Pesos e dados gerados são locais e ignorados pelo Git; manifestos guardam
  hashes, configurações e contagens.
- `data/selfplay/` é arquivo histórico. Ele só é lido por
  `training/migrate_selfplay_v3.py`, nunca por treino ou teaching.

## Scripts e opções

Todos os scripts têm bloco `CONFIG` no topo e argumentos CLI equivalentes.

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| Migrar | `training/migrate_selfplay_v3.py` | legado/V2/V3 | corpus V3 + manifestos |
| Auditar | `training/audit_selfplay.py` | corpus V3 | relatório de formato e volume |
| Selfplay | `tools/selfplay/run_selfplay.py` | busca e NNUE | V3 + manifesto |
| Replay directo | `training/prepare_replay.py` | V3 | dataset com teachers direct |
| Teaching | `training/run_teaching.py` | história ou partidas novas | dataset direct/search/mixed |
| Search profundo | `training/teachers/zq_deep_relabel.py` | histórias ou `@state` V3 | visitas MCAB, valor e Q |
| Seleção e mistura search | `training/select_replay_disagreement.py`, `training/build_search_priority_dataset.py` | replay directo + dois searches | dataset V3 ponderado |
| Experimento | `training/run_experiment.py` | dataset | pesos QAT, binário e relatório |
| Campanha | `training/run_campaign.py` | corpus e matriz | teaching, treino e arena |
| Benchmark | `tools/run_benchmark.py` | candidato | pares contra Titanium/Claustrophobia |

## Fase 1 — qualidade do corpus

1. Migrar e auditar antes de usar um shard.
2. Medir estados únicos globalmente, não apenas número de linhas.
3. Medir diversidade de aberturas, ply, reservas, margem de BFS, empates e
   partidas cortadas no máximo de plies.
4. Rejeitar ou reduzir peso de lote com repetição alta.

Um milhão de linhas não é um milhão de exemplos. O gen12 V2 ilustra o risco:
1.497.673 registros produziram apenas 83.244 estados independentes elegíveis.
O lote `gen-rich-v1-montecarlo` atual tem 4.713 partidas e 311.930 registros;
ele foi gerado com temperatura e ruído de abertura e deve passar pela auditoria
global de unicidade antes de entrar em teaching.

## Fase 2 — selfplay novo

Selfplay novo traz distribuição on-policy. Ele complementa o histórico; não o
substitui. Usar temperatura de abertura, ruído de raiz e sementes registradas.

```powershell
python tools/selfplay/run_selfplay.py --games 3000 --chunk-games 3000 --threads 14 --time-ms 150 --mode montecarlo --mc-obvious-plies 4 --mc-temp-obvious 0.8 --mc-temp-opening 1.35 --mc-temp-end 0.35 --mc-temp-decay-plies 18 --epsilon-midgame 0.04 --seed 2026091501 --out 'data/selfplay_canonical_v3/gen-rich-v1-montecarlo/selfplay_{shard:03d}.bin'
```

Auditar este lote antes do teaching. Se a unicidade não melhorar sobre os
lotes antigos, mudar a geração, não apenas aumentar jogos.

## Fase 3 — teaching

### Cobertura: relabel direto

`prepare_replay.py` aplica a rede antiga e Claustrophobia em forward direto.
Ele é rápido e serve para cobertura/pretreino, mas não é prova de alvo tático
correto. Seus exemplos devem ter peso moderado no dataset final.

```powershell
python training/prepare_replay.py --source data/selfplay_canonical_v3/gen7-montecarlo --out-dir data/teaching/replay-gen7 --max-positions 1000000 --device cuda
```

### Correção: search selecionado

Posições com divergência de valor/política, paredes críticas, corrida curta ou
risco de wandering devem receber MCAB/MCTS. O bridge
`tools/teacher/zq_deep_relabel.cpp` aceita tanto histórias quanto snapshots
V3 `@state`; snapshots têm histórico de repetição vazio, fato registrado no
manifesto. Usar budget de nós, não relógio, para reprodutibilidade.

Para trajetórias novas com história completa, usar teaching mixed:

```powershell
python training/run_teaching.py --mode mixed --out-dir data/teaching/fresh-mixed --games 1024 --max-positions 16384 --workers 8 --claustro-sims 256 --zq-nodes 1024 --gamma 0.995 --outcome-weight 1 --bootstrap-weight 1 --device cuda
```

`mixed` combina rede antiga, Claustrophobia direto, MCAB ZQuoridor e MCTS
Claustrophobia. As opções incluem pesos por teacher, desconto temporal,
bootstrap, temperatura, nós, simulações, workers e dispositivo.

## Fase 4 — mistura e prioridade

Misturar por grupos completos, sem estado compartilhado entre treino e
validação. Base inicial: 70–85% replay direto/histórico e 15–30% posições
novas ou relabeladas com search.

O trainer já consome `weight` por amostra. O caminho selecionado já gera:

```text
weight = clip(base * (1 + escala * (JS_política + |valor|)) * estabilidade_budget)
```

Comparar sempre uniforme versus priorizado. Divergência aumenta atenção; não
substitui confiança, clipping e validação independente.

```powershell
python training/select_replay_disagreement.py --replay-dir data/teaching/replay-old-gen1-500k --out data/teaching/search-priority/positions.jsonl --max-positions 10000
python training/build_search_priority_dataset.py --source-dataset data/teaching/replay-old-gen1-500k/dataset.npz --positions data/teaching/search-priority/top2000.jsonl --zq-targets data/teaching/search-priority/zq-search.npz --claustro-targets data/teaching/search-priority/claustro-search.npz --out data/teaching/search-priority/dataset.npz
```

## Fase 5 — arquiteturas

| Variante | Features | Uso |
|---|---:|---|
| `base:256` | 354 | controle de produção |
| `base:384` | 354 | ablação de capacidade |
| `race:256` | 456 | corrida e reservas |
| `race:384` | 456 | corrida com mais capacidade |
| `corridor:*` | futuro | só após hipótese e custo medido |

`race` é incremental: três slots ativos derivados de BFS/reservas são
substituídos quando o estado muda. Não reconstrói o acumulador inteiro.
`corridor` só deve entrar se puder reutilizar BFS/cache ou atualizar poucos
slots; métricas globais de múltiplos caminhos não entram no hot path sem
benchmark de custo.

## Fase 6 — treino, QAT e paridade

```powershell
python training/run_experiment.py --no-teaching --data data/teaching/mixed/dataset.npz --architecture race --hidden 384 --epochs 30 --batch-size 4096 --device cuda --out-dir results/experiments/race384
```

Treinar controles e uma hipótese por vez. Cada candidato precisa exportar
float/int8, checkpoint, manifesto de arquitetura e executável com as flags
corretas. Exigir `incremental_check.exe` sem divergências antes da arena.

## Fase 7 — benchmark e promoção

Screening curto:

```powershell
python tools/run_benchmark.py --opponents titanium,claustrophobia --pairs 40 --zq-executable results/experiments/race384/zquoridor.exe --nnue results/experiments/race384/student_int8.bin --zq-move-time-ms 50 --titanium-move-time-ms 50 --claustrophobia-sims 256 --claustrophobia-device gpu --output benchmark_results/race384-screen
```

Depois rodar confirmação com outro livro de aberturas, mais pares e orçamento
maior. Loss não promove rede. Promover somente com pares completos, intervalo
de confiança favorável e repetição em conjunto independente.

## Estado atual

- Corpus V3 auditado: 72.298.778 registros, sem shards legados.
- Replay direto completo: um milhão de estados históricos mistos e 500 mil
  estados distintos do gen1.
- `gen7-montecarlo` é a próxima fonte histórica rica para seleção e search.
- Selfplay rico de 3.000 jogos/14 threads está sendo concluído e será auditado.
- `race:384` e `base:384` foram treinadas; `race:384` teve melhor validação,
  mas o benchmark curto ainda não sustenta promoção contra Claustrophobia.
- O bridge de MCAB já aceita snapshots V3; o próximo uso é relabel seletivo
  por search, antes de criar qualquer feature `corridor`.
