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

## Livros de aberturas para benchmark

Os livros são artefatos versionados no repositório, nunca resultados locais.
`tools/external/openings_screen_v1.jsonl` contém 100 aberturas legais de seis
plies para triagem. `tools/external/openings_confirmation_v1.jsonl` contém 400
aberturas legais, sem sobreposição com o livro de triagem nem com o livro
histórico `tools/external/openings_titanium.jsonl`. A confirmação de finalistas
usa o livro de 400, cores invertidas e o mesmo tempo fixo por jogada.

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

O ZQuoridor search é auxiliar, não âncora: enquanto o wandering em corridas
com poucas paredes não estiver resolvido por arena, os novos datasets de
search usam 25% ZQuoridor e 75% Claustrophobia em política e valor. O
ZQuoridor ainda fornece cobertura de visitas e Q; não recebe peso dominante.
WDL histórico não entra no alvo principal de distillation.

```powershell
python training/select_replay_disagreement.py --replay-dir data/teaching/replay-old-gen1-500k --out data/teaching/search-priority/positions.jsonl --max-positions 10000
python training/build_search_priority_dataset.py --source-dataset data/teaching/replay-old-gen1-500k/dataset.npz --positions data/teaching/search-priority/top2000.jsonl --zq-targets data/teaching/search-priority/zq-search.npz --claustro-targets data/teaching/search-priority/claustro-search.npz --out data/teaching/search-priority/dataset.npz
```

## Fase 5 — arquiteturas

Uma arquitetura é uma hipótese mensurável, não apenas uma largura maior. Toda
variante precisa de export float/int8, atualização incremental, paridade
Python/C++ e arena antes de entrar na busca de produção.

### Matriz já suportada

Estas oito combinações já são aceitas por `run_experiment.py`, export e o
binário C++. Cada uma pode ser selecionada por `--architecture` e `--hidden`.

| Família | Features | Larguras | Estado |
|---|---:|---|---|
| `base` | 354: peões, muros, buckets BFS e reservas | 128, 256, 384, 512 | 256, 384 e 512 concluídas; 128 não priorizada |
| `race` | 456: `base` + margem de distância, margem de reservas e interação corrida/reservas | 128, 256, 384, 512 | 256 e 384 concluídas; 512 em treino |

`race` é incremental: três slots ativos derivados das distâncias BFS e das
reservas são removidos e recolocados em cada lance. Não reconstrói o
acumulador inteiro.

Ordem de execução para a matriz já suportada:

1. Encerrar os gates de arena de `base:384`, `race:384` e seu fine-tuning.
2. Treinar `base:256` e `race:256` para medir capacidade menor.
3. Treinar `base:512` e `race:512` para medir capacidade maior e custo de
   search. A rede 512 só continua se a arena justificar sua queda de nós/s.
4. Repetir somente as duas melhores arquiteturas com o corpus de teaching por
   search e comparar fine-tuning contra treino direto.

### Extensões de feature a implementar e testar

| Família proposta | Sinal adicional | Custo esperado | Pré-requisito |
|---|---|---|---|
| `phase` | buckets de ply, total de paredes e fase da partida | constante; contadores incrementais | nova codificação e paridade |
| `topology-lite` | contagem de muros H/V e diferença de orientação | constante; contadores incrementais | nova codificação e paridade |
| `race-phase` | interação margem BFS × fase × reservas | constante; sparse, sem nova BFS | ablação contra `race` |
| `corridor-touch` | slots tocados pelos caminhos mínimos de ambos os lados | usa cache BFS; pode variar com paredes | microbenchmark e atualização incremental comprovada |
| `multi-path` | robustez ou número de desvios de rota | mais caro; múltiplas travessias BFS | cache, limite de custo e ganho em arena |
| `wall-threat` | ameaça local de parede perto do caminho e do peão | depende de geometria local e cache | definição que não faça BFS por candidato |

`phase`, `topology-lite` e `race-phase` são as próximas extensões baratas.
`corridor-touch`, `multi-path` e `wall-threat` ainda não existem no código e
não podem ser declaradas disponíveis. Só entram depois de uma hipótese escrita,
benchmark de custo no hot path e teste incremental. Métricas globais de
múltiplos caminhos não entram no acumulador se exigirem BFS extra por lance ou
por candidato de parede.

## Fase 6 — treino, QAT e paridade

```powershell
python training/run_experiment.py --no-teaching --data data/teaching/mixed/dataset.npz --architecture race --hidden 384 --epochs 30 --batch-size 4096 --device cuda --out-dir results/experiments/race384
```

Treinar controles e uma hipótese por vez. Cada candidato precisa exportar
float/int8, checkpoint, manifesto de arquitetura e executável com as flags
corretas. Exigir `incremental_check.exe` sem divergências antes da arena.

QAT é ligado durante todo o treino. O trainer usa quatro épocas de warmup e
cosine annealing até `min_lr=5e-6`; o padrão passa a 80 épocas e paciência 12.
A cauda de baixa taxa é o fine-tuning final dos pesos já quantizados. A taxa
por época e o schedule entram em `train_report.json` e no checkpoint, portanto
uma retomada mantém o mesmo recozimento.

Fine-tuning também parte do melhor candidato com learning rate menor e um
corpus que mistura teaching por search. A loss de um fine-tuning com alvos ou
pesos diferentes não é comparável numericamente à loss do treino direto. Ela
só decide se o artefato está estável; a arena decide se ele é melhor.

## Fase 7 — benchmark e promoção

Todo candidato tem três comparações obrigatórias, registradas no mesmo gate:

| Comparação | Objetivo | Critério |
|---|---|---|
| Candidato × rede anterior | medir ganho real da NNUE | arena pareada, mesmas aberturas e cores invertidas |
| Candidato × Titanium | medir força externa | pares completos e relógio idêntico registrado |
| Candidato × Claustrophobia | medir força externa forte | pares completos, relógio idêntico e dispositivo registrados |
| Main × Titanium | criar a referência externa | mesmo livro, seed, cores e relógio do candidato |
| Main × Claustrophobia | criar a referência externa forte | mesmo livro, seed, cores e relógio do candidato |

`tools/benchmark_candidate.py` executa a matriz avulsa. O script mede o
candidato contra main, Titanium e Claustrophobia. O script também mede main
contra os dois bots externos no mesmo livro, seed e relógio. As duas linhas de
main são a referência para o delta externo de cada candidato.

Todos os lados usam o mesmo `move_time_ms`. O runner rejeita relógios
diferentes. Claustrophobia recebe um relógio de parede. A ponte limita as
simulações a um valor calibrado e registra as simulações e o tempo medido. Um
resultado com apenas número de simulações não é válido para promoção.

Loss, nós por segundo e partidas parciais não substituem esse gate.

`training/run_architecture_matrix.py` encadeia a matriz inteira. Ele grava
`matrix_status.json`, pula treino e benchmark já concluídos, e só inicia a
próxima arquitetura depois da matriz completa da anterior. Assim um processo
interrompido continua no próximo estágio pendente sem repetir jogos válidos.

```powershell
python training/run_architecture_matrix.py --architectures '["base:256","race:256","base:384","race:384","base:512","race:512"]' --epochs 80 --benchmark-pairs 100 --move-time-ms 200
```

Screening curto:

```powershell
python tools/benchmark_candidate.py --candidate-executable results/experiments/race384/zquoridor.exe --candidate-nnue results/experiments/race384/student_int8.bin --pairs 40 --move-time-ms 200 --claustrophobia-device cpu --output benchmark_results/race384-screen
```

Depois rodar confirmação com outro livro de aberturas, mais pares e orçamento
maior. Loss não promove rede. Promover somente com pares completos, intervalo
de confiança favorável e repetição em conjunto independente.

## Estado atual em 2026-09-17

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
  e 1.024 simulações). O dataset conservador usa 25% ZQuoridor e 75%
  Claustrophobia em política e valor, e está em
  `data/teaching/search-priority-gen1-conservative/dataset.npz`.
- A mistura de fine-tuning já foi materializada em
  `data/teaching/historical2m-search10-conservative/dataset.npz`: 2.001.868
  posições, 10% do peso de teaching por search e 90% de replay direct.
- Redes QAT treinadas no corpus direct de 2M. Todas usaram 80 épocas, quatro
  épocas de warmup e cosine annealing até `5e-6`:

| Rede | Melhor loss de validação | Situação |
|---|---:|---|
| `base:256` | 0,74569 | concluída; não priorizada |
| `race:256` | 0,74319 | concluída; arena completa |
| `base:384` | 0,73939 | concluída; arena completa |
| `race:384` | 0,73686 | melhor rede direta; arena completa |
| `base:512` | 0,73600 | concluída; matriz de referência completa |
| `race:512` | 0,73405 | concluída; matriz em execução |

- Fine-tuning antigo de `race:384` com 10% do peso de teaching por search
  usava a fonte 50% ZQuoridor e 50% Claustrophobia; ele é legado experimental.
  Os novos fine-tunings `race:512` e `base:512` usam a mistura conservadora
  25%/75%, QAT, warmup e cosine annealing. Em 40 épocas, `race:512` reduziu a
  loss do novo corpus de 0,91628 para 0,88574; `base:512`, de 0,92031 para
  0,89128. Essas losses não são comparáveis à loss do treino direct; a arena
  será o gate.

- Matrizes concluídas com 20 pares completos e 200 ms por jogada:

| Rede | Main | Titanium | Claustrophobia | Decisão |
|---|---:|---:|---:|---|
| `race:256` | 65,0% | 42,5% | 25,0% | sem promoção; IC contra main inclui 50% |
| `base:384` | 67,5% | 37,5% | 28,75% | sem promoção; IC contra main inclui 50% |
| `race:384` | 60,0% | 35,0% | 31,25% | sem promoção; IC contra main inclui 50% |
| `base:512` | 65,0% | 45,0% | 26,25% | triagem promissora; aguarda confirmação independente |
| `race:512` | 57,5% | 30,0% | 22,5% | abaixo de `base:512`; sem promoção |

  Os três primeiros relatórios não incluem ainda a linha de main contra os
  bots externos. As matrizes `base:512` e `race:512` já incluem essa referência.
  A repetição da referência no mesmo livro, seed e relógio variou: 35,0%/15,0%
  para `base:512` e 23,75%/20,0% para `race:512`, contra Titanium/Claustrophobia.
  Isso confirma que 20 pares só servem para triagem; a promoção exige a
  confirmação independente de 400 aberturas.

### Em execução

- A triagem externa anterior do `race:384` direct de 2M ficou inválida para
  promoção. Claustrophobia usou simulações em vez de um relógio por jogada.
  O processo foi interrompido. Nenhum resultado desse diretório decide força.
- O benchmark externo usa agora 200 ms por jogada para cada lado. A ponte do
  Claustrophobia calibra uma busca curta, limita as simulações e completa o
  orçamento de parede. O registro inclui o relógio, as simulações e o tempo
  medido de cada resposta.
- A matriz direct de `race:512` terminou a 200 ms por jogada, com o mesmo livro
  histórico e seed usados na referência da `base:512`. Ela não superou
  `base:512` na triagem.
- Os fine-tunings de `race:512` e `base:512` terminaram e seus binários QAT
  foram exportados e verificados incrementalmente. A matriz completa de
  `race512-search10-ft` terminou: 63,75% contra main, 35,0% contra Titanium
  e 21,25% contra Claustrophobia (20 pares por confronto, 200 ms/jogada).
  Os intervalos ainda não autorizam promoção. A matriz de
  `base512-search10-ft` terminou depois: 66,25% contra main, 45,0% contra
  Titanium e 20,0% contra Claustrophobia (20 pares por confronto, 200
  ms/jogada). A referência do main no mesmo run marcou 40,0% e 22,5% contra
  Titanium/Claustrophobia. Os dois candidatos continuam abaixo do gate de
  promoção: 20 pares são triagem e não demonstram superioridade estatística.

### Próximos gates

1. Concluir a consolidação das matrizes das redes com fine-tuning e compará-las
   com `base:512` direct no mesmo protocolo; a triagem atual favorece
   `base512-search10-ft` contra main, mas não contra Titanium.
2. Rodar main contra Titanium e Claustrophobia para cada seed e livro já usado
   pelos três relatórios antigos.
   Guardar a linha como referência histórica, sem misturar livros.
3. Repetir as duas melhores redes em um livro independente com ao menos 100
   pares. Exigir intervalo de confiança favorável contra main antes da
   promoção.
4. Medir as duas redes com fine-tuning em toda a matriz, usando o mesmo livro
   e relógio; promover apenas a vencedora para a confirmação independente.
5. Só então testar `phase`, `topology-lite` e `race-phase`. Não criar features
   de corredor antes de medir o valor do teaching e das arquiteturas atuais.

Em 2026-09-18 foi disparada a confirmação independente da `base512-search10-ft`
com `openings_confirmation_v1.jsonl`, 100 pares por confronto, 200 ms por lance
e Claustrophobia em CPU. O resultado será aceito somente após os intervalos
pareados e a comparação direta com Titanium serem favoráveis.

### Diário da confirmação independente (2026-09-18)

- A confirmação está em execução no diretório
  `benchmark_results/base512-search10-ft-confirm-200ms-s20260918`.
- A primeira etapa é `vs-main`; no último registro verificado havia 37 jogos
  válidos concluídos, sem falhas, de 200 previstos (100 pares, duas cores).
- O processo permanece ativo. Os resultados parciais não são usados para
  promoção, porque ainda não há amostra completa nem intervalo final.
- Depois de `vs-main`, o script executará os confrontos contra Titanium e
  Claustrophobia no mesmo livro, seed e limite de 200 ms por lance. O relatório
  final deverá registrar partidas válidas, pares completos, score, Elo e
  bootstrap pareado.
- Histórico de triagem já concluído: `race512-search10-ft` marcou 63,75%/35,0%/
  21,25% contra main/Titanium/Claustrophobia; `base512-search10-ft` marcou
  66,25%/45,0%/20,0%. Esses 20 pares serviram apenas para seleção inicial.
- Ainda falta: concluir os 200 pares contra main, concluir 100 pares contra
  cada bot externo, calcular os intervalos, comparar com a referência do main
  no mesmo livro e decidir se existe evidência suficiente para promoção.
