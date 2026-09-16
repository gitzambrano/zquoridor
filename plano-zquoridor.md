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
| Relabel Claustrophobia | `training/teachers/claustrophobia_relabel.py` | posições canônicas | política e valor direct |
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

Fine-tuning parte do melhor candidato com learning rate menor e um corpus que
mistura teaching por search. A loss de um fine-tuning com alvos ou pesos
diferentes não é comparável numericamente à loss do treino direto. Ela só
decide se o artefato está estável; a arena decide se ele é melhor.

## Fase 7 — benchmark e promoção

Todo candidato tem três comparações obrigatórias, registradas no mesmo gate:

| Comparação | Objetivo | Critério |
|---|---|---|
| Candidato × rede anterior | medir ganho real da NNUE | arena pareada, mesmas aberturas e cores invertidas |
| Candidato × Titanium | medir força externa | pares completos e orçamento de tempo registrado |
| Candidato × Claustrophobia | medir força externa forte | pares completos, simulações e dispositivo registrados |

`training/run_campaign.py` já executa a comparação direta e as duas externas
para os candidatos selecionados. Experimentos avulsos devem usar a mesma
matriz antes de qualquer promoção. Loss, nós por segundo e partidas parciais
não substituem esse gate.

Screening curto:

```powershell
python tools/run_benchmark.py --opponents titanium,claustrophobia --pairs 40 --zq-executable results/experiments/race384/zquoridor.exe --nnue results/experiments/race384/student_int8.bin --zq-move-time-ms 50 --titanium-move-time-ms 50 --claustrophobia-sims 256 --claustrophobia-device gpu --output benchmark_results/race384-screen
```

Depois rodar confirmação com outro livro de aberturas, mais pares e orçamento
maior. Loss não promove rede. Promover somente com pares completos, intervalo
de confiança favorável e repetição em conjunto independente.

## Estado atual em 2026-09-16

### Concluído

- Corpus V3 auditado: 72.298.778 registros, sem shards legados.
- Selfplay rico: 4.713 partidas e 311.930 registros em
  `gen-rich-v1-montecarlo`; ele usa temperatura, ruído de abertura e 14
  threads. Falta a auditoria global de unicidade antes de sua entrada no
  teaching.
- Replay direct histórico: 1.000.000 de estados mistos, 500.000 estados
  distintos do gen1 e um novo corpus amplo de 2.000.000 de estados. O último
  foi relabelado em CUDA, com 1.602.929 estados de treino e 397.071 de
  validação.
- Teaching por search: 2.000 posições gen1 selecionadas por divergência,
  relabeladas por ZQuoridor MCAB (512 e 2.048 nós) e Claustrophobia MCTS (256
  e 1.024 simulações). O dataset ponderado está em
  `data/teaching/search-priority-gen1/dataset.npz`.
- Redes QAT treinadas no corpus direct de 2M:

| Rede | Melhor loss de validação | Situação |
|---|---:|---|
| `base:384` | 0,74457 | controle concluído |
| `race:384` | 0,74254 | melhor loss; paridade incremental aprovada |

- Fine-tuning `race:384` com 10% do peso de teaching por search concluído.
  A loss 0,87903 é medida contra alvos ponderados diferentes e não pode ser
  comparada às perdas acima.

### Em execução

- Triagem externa do `race:384` direct de 2M contra Titanium e Claustrophobia:
  40 pares por adversário, 200 ms para ZQuoridor e Titanium, 512 simulações
  para Claustrophobia em CPU. Há 114 das 160 partidas gravadas; aguardar o
  resumo pareado final antes de reportar força.
- O teaching mixed de 1.024 trajetórias mantém as posições e os caches direct
  e Claustrophobia-search. Ele ainda depende do target ZQuoridor para gerar
  seu dataset final.

### Próximos gates

1. Terminar a triagem externa do `race:384` direct de 2M.
2. Rodar arena pareada `race:384` direct de 2M contra a rede anterior.
3. Rodar a mesma matriz para o `race:384` fine-tuned. Somente a arena pode
   decidir se o teaching por search ajuda.
4. Se algum candidato passar a triagem, repetir com livro independente e mais
   pares antes de promoção.
5. Só então ampliar a matriz para `base/race` 256 e 512 ou introduzir uma
   hipótese de corredor incremental. Não criar features de corredor antes de
   medir o valor das features race e do teaching por search.
