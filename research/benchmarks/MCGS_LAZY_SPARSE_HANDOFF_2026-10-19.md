# ZQuoridor — Lazy Expansion + Sparse Policy: estado, evidências e próximos passos

**Atualizado em 2026-10-19.** Documento de continuidade para outro chat/Codex. Este texto consolida o que **foi implementado e medido**, identifica o que está **pendente** e evita novos confrontos gigantescos sem diagnóstico. **Não promover a 4.02 nem repetir milhares de partidas antes de examinar as evidências existentes.**

## 0. Leitura rápida / decisão

- O motor de produção ainda se identifica como **4.01**; a combinação **Lazy + Sparse não foi incorporada ao main**. A candidata mínima para 4.02 está na [PR #58](https://github.com/gitzambrano/zquoridor/pull/58), em **draft**, com release gate de corretude aprovado. PUCT cached sum **não** faz parte dessa candidata.
- **A 200 ms**: Lazy + Sparse marcou **55,25% vs main em 400 H2H**, **55,50% vs Lazy isolada em 200 H2H**, e **62,25% vs Claustrophobia CPU** em 200 jogos pareados; na mesma bateria, Lazy obteve **59,00%** e main **52,25%** contra Claustrophobia. Sinais positivos, mas não equivalem a garantia em 3+2.
- **A 3+2**: na bateria controlada ainda incompleta, Lazy + Sparse marcou **36,25% vs Claustrophobia nas normais**, comparado com **47,50% Lazy** e **41,25% main**; marcou **58,75% nas centrais**, contra **55,00% main** (Lazy central cancelado). Contra Titanium normal marcou **62,50%**, vs **70,00% Lazy** e **77,50% main**. **Não promover** com essa evidência.
- Um diagnóstico valioso **JÁ CONCLUÍDO**: [igualdade de 20.000 simulações](https://github.com/gitzambrano/zquoridor/actions/runs/39149453858). Em **20 posições** (10 normais, 10 centrais), **duas buscas por posição = 40 buscas por motor**, exatamente **20.000 simulações por busca**: Lazy + Sparse e Lazy tiveram **os mesmos 40 lances**, **a mesma distribuição de visitas** (variação total média 0,0), **os mesmos priors legais** e **mesmo número de nós expandidos**. vs main, também 100% mesmos lances, variação média de visitas ~0,0000469, e Lazy expandiu em média **12.373,75 nós menos por busca**, confirmando que a semântica de expansão é diferente. Esses resultados refutam, nessas posições/contagem, uma mudança de política causada por Sparse, mas não provam que relógio/gerenciamento de tempo seja equivalente.
- Os workflows novos de **telemetria 3+2 adaptativa ON e OFF foram disparados, porém os jobs de partidas foram CANCELADOS**; só os preflights passaram. **Não** afirmar que há resultados válidos de A/B do Adaptive Time: [ON](https://github.com/gitzambrano/zquoridor/actions/runs/39149446311), [OFF](https://github.com/gitzambrano/zquoridor/actions/runs/39149450778). No diagnóstico de simulações iguais, 1.000 e 6.000 foram cancelados, mas **20.000 foi concluído com sucesso**. Não repetir o 20k sem razão.

## 1. Baselines, versões e referência justa

- Repositório: https://github.com/gitzambrano/zquoridor .
- Base congelada do screening original / confirmação 200 ms: commit **50a3237ea249c30077ee1e21d7902aa48d125f6d** (motor 4.01).
- Base congelada utilizada na investigação posterior: **22af954cf7f109465d9d7ddcf6171584c709a5df**, ainda identificada como motor **4.01**. O main já avançou com outros commits. Na última consulta para esta nota, HEAD estava em **5b51c0866d90df3a1ebde53276de388f95f04c91**. Antes de executar algo, verificar HEAD e os SHAs reais de cada motor.
- Lazy isolada congelada nos diagnósticos: **082abc4df338a3c46340aa7c77707c1259f7db16**.
- Branches de código: **exp/mcgs-lazy-deep-20261017**, **exp/mcgs-sparse-deep-20261017**, **exp/mcgs-puct-deep-20261017**, **exp/mcgs-lazy-sparse-20261018**, **exp/mcgs-lazy-sparse-puct-20261018**.
- Candidata 4.02 em branch **release/zquoridor-4.02-clean-20261018**, [PR #58](https://github.com/gitzambrano/zquoridor/pull/58); alterações mínimas de search + ativação padrão da Sparse no production_config.hpp + identificador UCI. [Release gate passou](https://github.com/gitzambrano/zquoridor/actions/runs/39107236047). **Não fazer merge antes de validação competitiva 3+2**.
- Protocolos: **200 ms** significa tempo real por lance para ambos, nunca número fixo de simulações. **3+2** significa relógios reais de 180.000 ms e incremento de 2.000 ms, fornecidos ao UCI via `go wtime ... btime ... winc 2000 binc 2000`; Claustrophobia CPU usa bridge de deadline real e Titanium seu relógio nativo. Parear abertura/cores/sementes. Não misturar scores de amostras diferentes.

## 2. O que Lazy Expansion de fato faz

Código: `src/mcab.hpp`. Cada nó possui `lazyValueEvaluated`, inicialmente falso. **No primeiro encontro** de um nó não terminal ainda não expandido, calcula-se o valor da folha NNUE e faz-se o backup **sem** `expandNode`, poupando enumeração/verificação das paredes legais, cabeça de política e alocação de arestas. Marca `lazyValueEvaluated=true`. **Se o nó for visitado novamente**, expande-se com `expandNode` e a busca seleciona filhos por PUCT. `McabStats.lazyExpansionsDeferred` conta expansões adiadas. O teste de corretude Lazy e MCAB-core passou.

**NÃO é uma otimização semanticamente gratuita**:
1. O main normalmente chama `expandNode` logo na primeira avaliação do nó. Lazy muda a árvore parcial e o momento de avaliação/expansão.
2. `mstats.nodesExpanded` alimenta o gate de `effectiveNodeBudget`; Lazy adia incrementos. Sob limite de nós, mesma configuração não implica mesma quantidade de simulações.
3. A busca usa **MCGS / transpositionGraph / graphQCorrection / treeReuse**, logo nós podem sobreviver a múltiplos lances. O marcador por nó pode persistir e mudar a dinâmica entre lances.
4. Em profundidade máxima (`depthCapped`) o caminho Lazy difere do main quanto à expansão de um nó novo. É um caso de borda ainda sem análise dirigida.
5. O ganho depende de quantas folhas são visitadas uma só vez; a distribuição pode variar com a duração da busca.

## 3. O que Sparse Policy de fato faz

Código: `src/nnue.hpp`, `forwardPolicyQuantSparse(acc, legalMoves, side, out)`, chamado via dispatcher em `src/mcab.hpp` sob `-DZQ_EXP_SPARSE_POLICY=1`. Substitui a inferência da **cabeça de política INT8** para **somente os logits de movimentos legais**, em vez de calcular todos os 209 logits; slots não utilizados do array permanecem em zero. Usa exatamente o mesmo produto interno inteiro, desescala e pesos. Cada movimento é canonizado com `mirrorMoveForPerspective(m, side)`, obrigatório para o jogador 1. A rede de **valor** não muda.

A validação específica aprovou **138.665 logits legais bit a bit idênticos**, **138.665 probabilidades normalizadas idênticas**, **1.135 casos sem paredes** e testes de ambas perspectivas; em 48 posições, os **209 priors da raiz do MCGS real** coincidiram com a implementação original. A cache NNUE/MCGS é indexada por identidade do estado incluindo contagem de paredes de ambos os lados. Não há bug identificado nos logits legais nem nos priors.

**O teste de 20k simulações iguais mostrou que, com a mesma Lazy, Sparse ON/OFF conserva exatamente os lances e visitas em 40 buscas**. Isso é evidência especialmente forte para tratar Sparse como otimização de cálculo da política nesse regime; não é prova universal para qualquer estado/cache.

Importante: Lazy evita calcular muitas políticas; portanto há menos trabalho para Sparse otimizar. **A aceleração incremental real de Lazy + Sparse sobre Lazy é próxima de zero** em benchmark controlado, apesar dos ~23% da Sparse isolada vs main.

## 4. Resultados 200 ms — sempre diferenciar screening e confirmação

### Testes isolados

| Variante | Screening H2H vs main (100 jogos) | Confirmação independente vs main (400 jogos) | Aceleração pareada inicial |
|---|---:|---:|---:|
| Lazy | 51,00% | **57,13%** (normais 56,50%; centrais 57,75%) | +24,12% |
| Sparse isolada | 45,00% | 51,50% (normais 52,25%; centrais 50,75%) | +22,98% |
| PUCT edge sum isolado | 53,50% | 50,25% (normais 51,00%; centrais 49,50%) | +4,25% |

Ao combinar 100 + 400: Lazy ~55,9%, Sparse ~50,2%, PUCT ~50,9% H2H; não são testes de 3+2. [Workflow Lazy](https://github.com/gitzambrano/zquoridor/actions/runs/39072035225), [Sparse](https://github.com/gitzambrano/zquoridor/actions/runs/39072094069), [PUCT](https://github.com/gitzambrano/zquoridor/actions/runs/39072038326).

### Combinações

| Ensaio | Lazy + Sparse | Lazy + Sparse + PUCT |
|---|---:|---:|
| H2H vs main, 400 jogos | **55,25%** (normais 54,25%; centrais 56,25%) | 54,13% (normais 56,25%; centrais 52,00%) |
| H2H vs Lazy, 200 jogos | **55,50%** (normais 58,00%; centrais 53,00%) | 53,75% (normais 59,50%; centrais **48,00%**) |
| vs Claustrophobia CPU, 200 jogos | **62,25%** (normais 62,50%; centrais 62,00%) | 59,25% |
| Lazy isolada vs Claustro, **mesmo grupo de aberturas** | 59,00% | 55,25% |
| Main vs Claustro, **mesmo grupo de aberturas** | 52,25% | 54,25% |

**Controles externos entre os dois workflows não são intercambiáveis.** Na bateria dupla, Lazy + Sparse superou Lazy +3,25 pp e main +10 pp contra Claustrophobia. Ainda são amostras moderadas; 3+2 não confirmou esse padrão. [Dupla](https://github.com/gitzambrano/zquoridor/actions/runs/39097883508), [Tripla](https://github.com/gitzambrano/zquoridor/actions/runs/39097885643).

### Throughput reproduzível, pareado no mesmo runner

[Dupla](https://github.com/gitzambrano/zquoridor/actions/runs/39097972285): **+22,34% vs main**, mas **−0,78% vs Lazy**. [Tripla](https://github.com/gitzambrano/zquoridor/actions/runs/39097974888): **+26,66% vs main**, **+5,53% vs Lazy**. Valores de até +73% em outro benchmark eram inconsistentes devido à variabilidade entre runners: **não os tratar como ganho real replicado**.

O PUCT edge-W cache mantém soma incremental para calcular Q do pai O(1); ao custo de memória/updates e ordem diferente de soma FP. [Auditoria especial](https://github.com/gitzambrano/zquoridor/actions/runs/39107349132) passou **80 buscas, 152.903 simulações e 46 reusos de árvore**, sem desvio além de tolerância vs recomputação. Não demonstrou Elo extra consistente e **fica fora da 4.02**.

## 5. Resultados 3+2 — o que foi realmente concluído

[Bateria controlada da dupla](https://github.com/gitzambrano/zquoridor/actions/runs/39106925875), 40 jogos por célula; todos abaixo **registrados sem jogos inválidos**:

| Oponente / abertura | Main | Lazy isolada | Lazy+Sparse |
|---|---:|---:|---:|
| Claustrophobia — normal | **41,25%** | **47,50%** | **36,25%** |
| Claustrophobia — central | **55,00%** | **cancelado** | **58,75%** |
| Titanium — normal | **77,50%** | **70,00%** | **62,50%** |
| Titanium — central | cancelado | cancelado | cancelado |

Para Claustrophobia, combinando apenas duas células de **mesmo** workflow (normal + central), combo 47,50% em 80 jogos e main 48,125% em 80: resultado quase igual (**−0,625 pp**), mas o controle Lazy central está faltando. Para Titanium normal, combo **−15 pp** vs main e **−7,5 pp** vs Lazy; central ainda sem resultado. **Não é confirmação da 4.02.** O restante do workflow foi cancelado, não está mais rodando.

Outras baterias 3+2 da Lazy isolada: [principal](https://github.com/gitzambrano/zquoridor/actions/runs/39097765775) completa e [independente](https://github.com/gitzambrano/zquoridor/actions/runs/39097720365) parcialmente cancelada; os controles flutuam significativamente entre amostras. Não misturar percentuais de diferentes livros/coortes como se fossem H2H pareados.

## 6. Diagnóstico: 20.000 SIMULAÇÕES IGUAIS — conclusão importante

[Workflow equal-sims](https://github.com/gitzambrano/zquoridor/actions/runs/39149453858), job 20k **success**, jobs 1k/6k **cancelled**.

- Exatamente **20.000 backups/simulações por busca**, 20 posições distintas (metade normal, metade central), duas buscas consecutivas pela mesma raiz por motor, **40 decisões por motor**.
- **Lazy + Sparse vs Lazy**: 100% mesmo lance; variação total média da distribuição de visitas **0,0**; mesmos priors; diferença média de expansões **0**.
- **Lazy + Sparse vs main**: 100% mesmo lance; variação média de visitas **0,000046875** (praticamente idêntica); diferença média de nós expandidos **−12.373,75** (combo menos).
- **Lazy vs main**: mesmos números de movimento, variação de visitas e delta de expansões da combo vs main.
- Consequência: nesta amostra, **Sparse sobre Lazy é semanticamente gratuita na contagem fixa de simulações**. Lazy muda muito o número de nós classificados como expandidos sem alterar o melhor lance nesse ensaio. Não confundir com menor gasto de tempo total de uma partida ou com 3+2.
- A rotina repetiu cada estado para estimular reuso, mas os relatórios atuais **não comprovam isoladamente** o percentual efetivo de cache hits ou `treeHit`: isso merece checagem pontual, **não** um novo torneio.

## 7. A hipótese de regressão de 3+2: tempo adaptativo + nó-limite

Caminho UCI `tools/external/zquoridor_uci.cpp`: `go movetime 200` **não ativa** Adaptive Time. O comando com relógios reais `go wtime ... btime ... winc ... binc ...` chama `zqtime::allocate()`, obtém `optimumMs` e `maximumMs`, ativa `params.adaptiveTime=true` e entrega o máximo como hard budget. O MCGS classifica a raiz por visit ratio, q gap e estabilidade, buscando **1,00x**, **1,25x (uncertain)** ou **1,60x (volatile)** do optimum. Também pode aumentar o teto de expansões em posições voláteis. **A árvore diferente da Lazy pode classificar a raiz diferentemente e gastar tempos distintos**; um motor mais rápido pode gastar menos tempo total e terminar com menos simulações úteis.

Isso é **HIPÓTESE plausível, não causa medida**. Outra possível questão é a interação com `mstats.nodesExpanded` e max-depth. Precisamos identificar precisamente o stop reason e a alocação por lance.

### Instrumentação já implementada, mas sem partidas válidas do A/B

- [PR #59 adaptive ON](https://github.com/gitzambrano/zquoridor/pull/59), [workflow](https://github.com/gitzambrano/zquoridor/actions/runs/39149446311): preflight success; **12 jobs de jogos cancelados**.
- [PR #60 adaptive OFF](https://github.com/gitzambrano/zquoridor/pull/60), [workflow](https://github.com/gitzambrano/zquoridor/actions/runs/39149450778): preflight success; **12 jobs de jogos cancelados**.
- [PR #61 equal sims](https://github.com/gitzambrano/zquoridor/pull/61), [workflow](https://github.com/gitzambrano/zquoridor/actions/runs/39149453858): **20k concluído**, outros dois cancelados.
- Código instrumental: `tools/diag/apply_instrumentation.py`, `tools/diag/summarize_clock.py`, `tools/diag/equal_simulations.py`; workflows `.github/workflows/diag-3p2-adaptive.yml`, `diag-3p2-fixed.yml`, `diag-equal-sims.yml` nos branches mencionados. Há flags diagnósticas `--diag-no-adaptive-time` (ainda com relógio 3+2, mas alocação fixa optimum por lance) e `--diag-fixed-sims` (SOMENTE experimento de paridade, não torneio). Captura por lance tempo, sims, expansões, classificação, tempo-alvo, razão visitas/Q, stop reason, clocks antes/depois.
- **Não recriar a matriz de 576 jogos por protocolo**: é cara, foi cancelada e é desnecessária antes da análise pontual. Diagnosticar primeiro nos dados já produzidos.

## 8. Próximo chat — instrução de execução de baixo custo

**Ordem proposta (uma etapa por vez, mínima e falsificável):**

1. **Ler este relatório e abrir logs dos workflows citados. NÃO iniciar nenhuma rodada nova automaticamente.** Verificar também PR #58 ainda em draft, versão produção 4.01, HEAD atual, jobs cancelados e se há outras rodadas genuinamente em curso. Evitar duplicar experimentos.
2. **Auditar as partidas 3+2 já existentes**: para as mesmas aberturas, extrair dos arquivos `games.jsonl` / logs o tempo total de busca e por lance (`move_times[].elapsed_ms`, `clock_before_ms`, `clock_after_ms`, `search_last`), derrotas por tempo, incremento, timeouts, número de lances. Se os artifacts não existirem ou não estiverem acessíveis, declarar isso; não inventar telemetria nem rerodar tudo.
3. **Preferir 8 a 16 posições críticas** extraídas das derrotas reais, com mesma posição/relógio/NNUE, execução serializada no **mesmo hardware**, comparando main / Lazy / combo, com adaptive ON e OFF. Primeiro medir decisões, budget óptimo/máximo, sims/s, número de sims, expansões, root class, stop reason e reuso de árvore. Pode ser **sem jogos completos** para triagem. Controlar `go wtime` e incremento idênticos; nunca chamar `go movetime` de controle nativo.
4. Se a diferença continuar e o mecanismo ficar claro, executar **no máximo um pequeno teste de confirmação** com algumas dezenas de partidas pareadas e aberturas centrais/normais representativas, não 576 + 576 nem 3.600 novamente. Planejar poder estatístico e custo **antes** de lançar qualquer novo jogo.
5. Checar pontualmente: (a) contagem de `nodesExpanded` vs simulações para Lazy, (b) mecanismo de cutoff do adaptive class, (c) `depthCapped`, (d) reuso da árvore no segundo lance/mesma raiz, (e) distribuição de gasto de relógio ao longo da partida, (f) tempos perdidos versus derrotas estratégicas, (g) se o livro aberto é pareado realmente nos três motores. Testes pequenos com oráculos/replays são preferíveis a H2H longos.
6. Critério de merge 4.02: não piorar 3+2 de forma relevante contra main nem Lazy, sobretudo em centrais e nos dois adversários, sem incremento de time losses. Exigir paridade de compilações/default `ZQ_EXP_SPARSE_POLICY=1`, versão UCI e passagem do release gate antes do merge. Não trazer PUCT para a 4.02 sem experimento separado.

**Uma conclusão adicional:** não confundir **+55% H2H em 200ms** com aumento garantido de força em 3+2: o sistema é sensível ao Adaptive Time, orçamento de expansões e tempos efetivamente consumidos. Na paridade de 20k simulações, a Sparse não alterou o melhor lance ou as visitas sobre Lazy; logo **a primeira ação deve ser diagnosticar alocação de tempo e estatísticas reais, não retreinar NNUE nem aumentar N de jogos**.

## 9. Links condensados

- [200ms Lazy vs main](https://github.com/gitzambrano/zquoridor/actions/runs/39072035225)
- [200ms Sparse vs main](https://github.com/gitzambrano/zquoridor/actions/runs/39072094069)
- [200ms dupla Lazy+Sparse H2H + Claustro](https://github.com/gitzambrano/zquoridor/actions/runs/39097883508)
- [200ms tripla](https://github.com/gitzambrano/zquoridor/actions/runs/39097885643)
- [Bench de throughput dupla](https://github.com/gitzambrano/zquoridor/actions/runs/39097972285)
- [Bench de throughput tripla](https://github.com/gitzambrano/zquoridor/actions/runs/39097974888)
- [3+2 controlado dupla](https://github.com/gitzambrano/zquoridor/actions/runs/39106925875)
- [3+2 Lazy vs main](https://github.com/gitzambrano/zquoridor/actions/runs/39097765775)
- [Equal-sims 20k aprovado](https://github.com/gitzambrano/zquoridor/actions/runs/39149453858)
- [Adaptive ON — jogos cancelados](https://github.com/gitzambrano/zquoridor/actions/runs/39149446311)
- [Adaptive OFF — jogos cancelados](https://github.com/gitzambrano/zquoridor/actions/runs/39149450778)
- [Release candidata 4.02 (não mergear ainda)](https://github.com/gitzambrano/zquoridor/pull/58)

**Pedido exato ao próximo chat:** continuar investigação causal de Lazy + Sparse no relógio 3+2 com os artefatos e testes já existentes, SEM disparar outra matriz grande. Entregar hipótese confrontada com telemetria observada, correção ou teste focalizado de poucas posições, e recomendação objetiva de manter/ajustar/rejeitar a PR #58.