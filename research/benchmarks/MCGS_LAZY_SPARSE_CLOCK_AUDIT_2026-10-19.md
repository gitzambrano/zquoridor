# ZQuoridor — auditoria causal de Lazy + Sparse em 3+2 (19 outubro 2026)

> Continuação do [handoff consolidado](https://github.com/gitzambrano/zquoridor/blob/main/research/benchmarks/MCGS_LAZY_SPARSE_HANDOFF_2026-10-19.md). **Nenhum teste de 20 mil simulações foi repetido. Nenhum torneio novo foi disparado.** A release [PR #58](https://github.com/gitzambrano/zquoridor/pull/58) permanece bloqueada para merge.

## 1. Dados originais resgatados

Auditoria dos `games.jsonl` dos **12 artifacts da bateria 3+2 controlada** (run [39106925875](https://github.com/gitzambrano/zquoridor/actions/runs/39106925875)), incluindo os artifacts parciais de jobs depois cancelados: **396 partidas registradas**.

| Oponente / abertura | Motor | Partidas registradas | Tempo médio por lance do ZQ | `nodesExpanded` médio/lance | `treeHit` |
|---|---|---:|---:|---:|---:|
| Claustrophobia / normal | main | 40 | 7126 ms | 115516 | 92,6% |
| Claustrophobia / normal | Lazy | 40 | 6876 ms | 62573 | 91,4% |
| Claustrophobia / normal | Lazy+Sparse | 40 | 6895 ms | 76930 | 91,9% |
| Claustrophobia / central | main | 40 | 7473 ms | 117299 | 89,0% |
| Claustrophobia / central | Lazy | 27, **parcial** | 7402 ms | 112634 | 88,1% |
| Claustrophobia / central | Lazy+Sparse | 40 | 7410 ms | 76990 | 88,7% |
| Titanium / normal | main | 40 | 6403 ms | 98731 | 87,8% |
| Titanium / normal | Lazy | 40 | 6588 ms | 78372 | 90,4% |
| Titanium / normal | Lazy+Sparse | 40 | 6616 ms | 68034 | 89,5% |
| Titanium / central | main | 16, **parcial** | 7253 ms | 103072 | 91,0% |
| Titanium / central | Lazy | 15, **parcial** | 7370 ms | 64226 | 91,7% |
| Titanium / central | Lazy+Sparse | 18, **parcial** | 6980 ms | 77842 | 89,1% |

Não foi registrado nenhum *forfeit* por tempo nas 396 partidas. Em Titanium-normal, ao fim dos primeiros 15 lances próprios em aberturas pareadas, os tempos acumulados médios são main 120,5 s, Lazy 121,8 s e combo 120,0 s, sem evidência de que combo esteja sistematicamente gastando menos tempo.

**Limitações:** `nodesExpanded` não mede o mesmo trabalho no main e em Lazy; um nó Lazy pode receber valor e backup sem expansão. As partidas das diferentes variantes rodaram em *jobs/runners* distintos. Os resultados parciais centrais não podem ser usados como amostras completas intercambiáveis.

## 2. Diagnósticos parciais de jobs cancelados

Os artifacts dos runs [Adaptive ON 39149446311](https://github.com/gitzambrano/zquoridor/actions/runs/39149446311) e [OFF 39149450778](https://github.com/gitzambrano/zquoridor/actions/runs/39149450778) preservaram dados úteis, embora os jobs tenham sido cancelados.

| Modo | Jogos parciais | Lances ZQ instrumentados | Stop=2 adaptativo | Stop=1 tempo rígido | Stop=3 limite nós | Outras saídas |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive ON | 42 | 1199 | 1125 | 5 | 0 | 69 |
| Adaptive OFF | 5 | 131 | 0 | 120 | 0 | 11 |

ON: classes observadas (`diag_class`) Stable 773, Uncertain 229, Normal 88, Volatile 35, Invalid 74. A maior parte das decisões termina pelo sinal adaptativo; **não** foi observado o node cap como razão de parada nessas amostras. OFF é pequena e não pareada em cobertura de motores, logo não autoriza afirmar melhora ou piora de Elo.

Exemplo da mesma abertura normal, mesmo lado, relógio de 3+2, mesmo `diag_optimum=diag_target=8115 ms`: contra Titanium, no primeiro lance, main gerou 360.198 simulações, Lazy 498.391 e combo 306.590. Os três usaram aproximadamente 8,12 s e pararam por Adaptive Time. **Isso não prova regressão de throughput**: os motores rodaram em runners separados, e benchmarks realmente pareados de mesmo hardware mediram ~−0,78% entre combo e Lazy, com igualdade exata de visitas/prior sob 20.000 simulações.

## 3. Problema concreto identificado e correção

O script existente `--diag-no-adaptive-time` da PR [#60](https://github.com/gitzambrano/zquoridor/pull/60) gerava um A/B confundido. Em UCI ON, o MCGS recebe `TimeManager.maximumMs` e usa esse valor em `effectiveNodeBudget`. Em OFF, recebe `TimeManager.optimumMs`, usado **também** para o cálculo do node cap. Assim, desligar o tempo adaptativo também reduzia o teto inicial de expansões, mesmo que a intenção fosse alterar apenas o stop por tempo. Não apareceu `stop=3` no fragmento de partidas, portanto **esse defeito metodológico não está demonstrado como causa da regressão competitiva**.

Corrigido **somente** na branch diagnóstica `exp/diag-3p2-fixed-20261018`, commit [58b2a549](https://github.com/gitzambrano/zquoridor/commit/58b2a5491d987bb143397ea9cd879f17c4925271): o modo OFF fixa o node cap inicial pelo `maximumMs` da mesma posição/relógio, mantendo o stop por tempo em `optimumMs`. A escalada dinâmica de cap permanece desativada no OFF porque integra o próprio controle adaptativo. O `main` e a candidata da PR #58 não foram alterados.

## 4. Auditoria de código: semântica do nó e limite de profundidade

- `Lazy` marca `lazyValueEvaluated` no primeiro encontro de um nó não terminal, avalia a NNUE e dá backup. Somente na revisita executa `expandNode`. Esse bit é parte do nó e sobrevive ao reroot/compaction do MCGS; não foi identificado um reinício incorreto nele.
- `depthCapped` ocorre em `depth >= params.maxTreeDepth` ou no limite de `mcabAccStack` (48 plies por padrão). Main expande um nó novo mesmo ao bater o depth cap, e então avalia; Lazy avalia sem expandir. É **diferença comprovada do código**, não prova de erro de decisão. Uma folha já expandida em depth cap é avaliada em ambos. Se o nó Lazy for reencontrado após reroot em profundidade mais rasa, será expandido. Contagem de ocorrências em jogos reais não está instrumentada: não atribuir a regressão ao depth cap sem observar frequência e impacto.
- Em `graphQCorrection`, o backup do MCGS utiliza os valores/visitas acumulados de nós transpostos. A equivalência de priors e visitas na amostra de 20k se mantém entre Lazy e Lazy+Sparse; isso enfraquece a hipótese de erro funcional na cabeça de política Sparse.
- O `treeHit` real é alto (88–93%), portanto ausência de reuso não explica a regressão. Ele não prova, por si, que todos os nós sobreviventes preservam valor/caches corretamente.

## 5. Experimento mínimo iniciado, sem torneio

Foi criado `tools/diag/targeted_native_clock.py` (commit [198cf582](https://github.com/gitzambrano/zquoridor/commit/198cf582bf0e418034a736d6c6497115db7f69aa)) com **8 posições de meio-jogo realmente extraídas** de vitórias e derrotas de Claustrophobia/Titanium e aberturas centrais/normais; os relógios têm os valores `clock_before_ms` dos logs. A workflow `.github/workflows/diag-targeted-native.yml` (commit [94e584cd](https://github.com/gitzambrano/zquoridor/commit/94e584cdd1a917f377bd52aa1b9c509a920cfd82)) compila main, Lazy e combo no **mesmo runner**, alterna a ordem, e compara Adaptive ON/OFF com `go wtime ... btime ... winc 2000 binc 2000`, registrando simulações, expansões, classe, alvo, stop reason e reuso após repetir a raiz. **Não são partidas nem benchmark de Elo**, e repetir a mesma raiz não substitui reuso real por dois plies. Run: [39236992991](https://github.com/gitzambrano/zquoridor/actions/runs/39236992991). A criação deste documento não pressupõe sucesso nem resultados do run.

## 6. Conclusão técnica e decisão

**Já refutado nesta evidência:** perdas sistemáticas por tempo, falta de reuso da árvore, e node cap como stop reason dominante; não há sinal determinístico de bug nos logits Sparse.

**Problema corrigido:** A/B ON/OFF mudava simultaneamente o modo adaptativo e o teto de expansões, e foi normalizado só no diagnóstico.

**Ainda sem causa demonstrada:** a diferença de Elo entre 200 ms e 3+2 e, em particular, eventual perda exclusiva da combo. Há uma forte divergência de throughput entre jobs distintos que precisa ser confrontada com **mesmo hardware** antes de inferir bug. O Adaptive Time domina a interrupção, mas ainda não foi demonstrado que ele prejudique mais a combo.

**Gate:** não promover nem alterar a PR #58 até analisar os dados focais. Não disparar outra matriz massiva. Se o experimento seriado confirmar divergência determinística sob ON e não sob OFF, localizar e corrigir o ponto de classificação/stop; se confirmar equivalência entre Lazy e combo, tratar a regressão observada como sinal estatístico ou confundimento ambiental, não defeito de Sparse comprovado.
