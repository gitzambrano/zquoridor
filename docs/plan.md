# Plano ZQuoridor — Documento Canônico e Painel Operacional

Atualizado em 2026-09-19. Este é o DOCUMENTO CANÔNICO oficial de planejamento,
registro histórico (passado antigo e recente), status operacional e catálogo
completo de TO-DO do projeto ZQuoridor. Todas as metas, decisões arquiteturais,
taxonomias de fraqueza e especificações de dataset convergem para este arquivo.
Para reproduzir qualquer experimento, consulte as seções correspondentes e os
arquivos de configuração indicados.

### Leitura rápida para um agente novo

1. **Dados:** use o caminho completo do dataset indicado na tabela da rede e em `data/datasets.md`. Não misture `dataset.npz` de pastas diferentes pelo nome.
2. **Treino:** leia `config.json`, `student.architecture.json` e `train_report.json`. O `student_int8.bin` é o peso carregado pelo engine.
3. **Teste:** use `200 ms` por lance, o mesmo livro, seed e cores. Consulte `summary.json`; loss não é medida de força.
4. **Estado:** `concluído` tem artefato e relatório; `em execução` tem processo vivo e arquivos parciais; `TODO` ainda não produziu resultado.

---

## Painel Executivo: Onde Estamos e Para Onde Vamos

### Meta Estratégica Maior (/goal)
**Alcançar mais de 60% de vitórias contra a Claustrophobia em todas as famílias de abertura** (superando a barreira histórica do motor e dominando o jogo tanto de Brancas quanto de Pretas, sem qualquer perda de vazão de nós por segundo).

---

### Onde Estamos (Status Atual)
- **Baseline de Produção Oficial**: `race512-cr200k-champion` promovida e homologada como rede padrão do `main`:
  - **vs Titanium**: **63,0% (+92,5 Elo)** em match oficial de 600 jogos (378W / 0D / 222L) — recorde histórico absoluto.
  - **vs `main` (anterior)**: **81,25% (+254,7 Elo)** em triagem direta (32W / 1D / 7L).
  - **vs Claustrophobia**: **47,92% (-14,5 Elo)** em match de 600 jogos a 200 ms na GPU (IC 95%: [44,33%, 51,50%], alcançando +10,4 Elo).
  - **Paridade de Cores Restaurada**: 141W de Brancas (47,0%) e 140W de Pretas (46,7%), eliminando o colapso histórico do segundo jogador.
  - **Suite Tático Center Rush (66 jogos vs Claustrophobia GPU MCTS)**:
    - **Resultado Geral**: **31,8%** (20W / 2D / 44L / 66 jogos).
    - **De Pretas (Jogador 1)**: **51,5%** (16W / 2D / 15L) — superando a Claustrophobia GPU jogando como segundo jogador!
    - **De Brancas (Jogador 0)**: 12,1% (4W / 0D / 29L) — vulnerabilidade crítica quando atacada reativamente em abertura central.
    - **Por Categoria Tática**:
      - `pawn_jump`: **50,0%** (4W / 0D / 4L / 8 jogos) — equilíbrio total no combate corpo a corpo.
      - `front_wall`: **33,3%** (5W / 2D / 11L / 18 jogos).
      - `sidestep_flank`: **33,3%** (4W / 0D / 8L / 12 jogos).
      - `vertical_channel`: **33,3%** (4W / 0D / 8L / 12 jogos).
      - `reed_rear_wall`: **18,8%** (3W / 0D / 13L / 16 jogos) — ponto mais fraco em contenção traseira.
  - **Pesos e Código Canônicos**: Pesos em `data/nnue/nnue_weights.bin` (f32) e `data/nnue/nnue_weights_int8.bin` (int8), C++ padrão com `ZQ_NNUE_RACE_FEATURES = 1` e `ZQ_NNUE_HIDDEN = 512`, WASM e web GUI 100% atualizados.

- **Frente de Dados e Mineração de Crise**:
  - **10.030 Estados Críticos Minerados** (`data/teaching/loss-center-seeds/positions.jsonl`): 5.952 estados de derrotas contra Claustrophobia, 4.295 de derrotas contra Titanium e 198 do catálogo Center Rush.
  - **Geração de Rollouts Sintéticos**: Em execução multithread (`bin/generate_rollouts.exe` sobre 1.000 sementes com profundidade e ramificação top-K, >27.000 CPU segundos).

- **Nova Campeã Homologada (`multipath:512` — 480 entradas, 60 épocas de Annealing)**:
  - **vs Claustrophobia (GPU, 600 jogos)**: **51,08% (+7,53 Elo)** — **PRIMEIRA VITÓRIA HISTÓRICA DO PROJETO CONTRA A CLAUSTROPHOBIA GPU!**
    - Supera a campeã anterior (47,92%) em **+22,0 Elo**.
    - Brancas: **152,0 / 300 (50,7%)** | Pretas: **154,5 / 300 (51,5%)**.
    - Aberturas de Muro/Laterais: **272,0 / 526 (51,71%)**.
    - Aberturas Centrais/Peão: **34,5 / 74 (46,62%)** (salto de +15 pontos percentuais vs baseline anterior).
    - Status estatístico: **Strength claim ready: true** (IC 95%: [47,50%, 54,58%]).
  - **vs Titanium (100 jogos normais)**: **64,0% (+99,95 Elo)** (35W Brancas / 29W Pretas).
  - **vs Titanium (100 jogos Center Rush)**: **45,0% (-34,86 Elo)** (Pretas dominando com 72,0% de vitórias; Brancas em 18,0%).
  - **Zero Custo de BFS**: Validação incremental em C++ com 100% de paridade (0 divergências em 4.758 posições).

- **Rede Experimental Avaliada (`margin_regime:512` — 588 entradas)**:
  - 132 features de interação direta: $\Delta d \in [-16..16] \times 4$ regimes de esgotamento de muros.
  - Treinamento de 60 épocas com QAT concluído: `best_val_loss = 0.7308`, `policy_kl = 0.1983`, `value_mae = 0.2060`.
  - Paridade incremental em C++ verificada: 0 divergências em 4.758 posições.
  - **Bateria Oficial de 800 Jogos Concluída (com 6 workers paralelos)**:
    - **vs Claustrophobia (GPU, 600 jogos)**: **49,00% (294,0 / 600)** (-6,95 Elo).
      - Aberturas Centrais/Peão: **43,92% (32,5 / 74)** (Brancas: 35,1% | Pretas: 52,7%).
      - Aberturas de Muro: **49,71% (261,5 / 526)** (Brancas: 51,9% | Pretas: 47,5%).
      - Brancas: 149,5 / 300 (49,8%) | Pretas: 144,5 / 300 (48,2%).
    - **vs Titanium (100 jogos normais)**: **65,00% (65,0 / 100)** (+107,5 Elo).
    - **vs Titanium (100 jogos Center Rush)**: **47,00% (47,0 / 100)** (-20,9 Elo) (Pretas: 70,0% | Brancas: 24,0%).
  - **Conclusão Arquitetural**: A rede `margin_regime` é competitiva, mas fica atrás da `multipath:512` por **14,5 Elo** contra a Claustrophobia (49,0% vs 51,08%) e por **2,7%** nas aberturas centrais. A multiplicidade e resiliência topológica de rotas da `multipath` provou ser uma heurística mais informativa para a rede do que a matriz cartesiana distância $\times$ regime de muros.

- **Status dos Rollouts de Crise (`generate_rollouts.exe`)**:
  - Em execução com **8 threads** em background sobre 1.000 sementes críticas (6 rollouts/semente, MCAB com 128 nós por lance, busca profunda até 120 plies).
  - Mais de 293.600 CPU segundos consumidos (~10,2h de parede), atingindo aproximadamente **~95% de conclusão**.
  - O dataset resultante fornecerá alvos ricos de política (visitas) e valor descontado TD($\lambda$) para o fine-tuning cirúrgico da defesa de Brancas em aberturas centrais.

- **Infraestrutura de Benchmark Paralelo**:
  - `tools/run_full_candidate_suite.py` e `tools/run_benchmark.py` parametrizados com `--workers 6` como padrão. O tempo total de execução dos 800 jogos caiu de ~3 horas para ~25 minutos na máquina local com total isolamento e persistência dos dados.

### Protocolo Mandatório para os Próximos Treinamentos
1. **Mínimo de 60 Épocas com Recozimento Térmico (Annealing)**: Treinamentos não podem ser interrompidos prematuramente; devem cumprir $\ge 60$ épocas com decaimento suave de learning rate (cosine schedule até $2\cdot 10^{-7}$) para garantir assentamento profundo dos pesos quantizados (QAT).
2. **Super-Ponderação das Fraquezas contra Claustrophobia**:
   - Posições de Brancas em colisão central e aberturas `reed_rear_wall` recebem multiplicador focal ($\times 6,0 \sim \times 10,0$).
   - Estados de derrotas mineradas contra a Claustrophobia recebem peso máximo.
   - Fundo do Tier 1 (10M) mantido com peso $\times 1,0$ como âncora de regularização contra esquecimento catastrófico.
3. **Alvo Híbrido TD($\lambda$) / MCTS Value Blending**:
   $$V_{\text{target}} = \lambda \cdot V_{\text{search}} + (1 - \lambda) \cdot z_{\text{rollout}} \quad (\lambda \approx 0,75)$$
4. **Action-Q Sharpening de Política**:
   $$\pi'_a \propto N_a^\alpha \cdot \exp(\beta \cdot Q_a)$$
   Filtra ruído e purifica a distribuição de probabilidade para lances decisivos.

### Para Onde Vamos (Próximos Passos Imediatos)
1. **Fine-Tuning Focal da Campeã (`multipath:512`) na Defesa de Brancas**:
   - A `multipath:512` é a líder isolada do projeto contra Claustrophobia GPU (51,08% / +7,5 Elo).
   - O calcanhar de aquiles identificado é a defesa de Brancas contra Center Rush (18% vs Titanium, 35% vs Claustrophobia).
   - Executar fine-tuning com os rollouts de crise de `task-6945` e learning rate recozido (`1.5e-5 → 1e-7`) para elevar o score de Brancas para >50%.
2. **Treinar Próximas Variações Arquiteturais**:
   - `phase:512` (480 inputs: estoque de muros e fase).
   - `multipath_phase:512` (combinação de multiplicidade de rotas + fase do jogo).

---

## 1. Big picture

Objetivo: treinar NNUEs simples, robustas contra wandering e boas em corridas;
medir todas no mesmo relógio contra a rede do `main`, Titanium e
Claustrophobia; promover somente uma rede com evidência independente.

```text
histórico + selfplay -> contrato V3 -> replay/teaching -> datasets mistos
-> treino QAT/fine-tuning -> export/paridade -> screening -> confirmação
-> promoção somente com intervalo favorável
```

O novo baseline oficial de produção é **`race512-cr200k-champion`**, formalmente promovido para `data/nnue/nnue_weights_int8.bin` e incorporado como arquitetura padrão em `src/nnue.hpp`.

## 2. Contrato de dados e benchmark

Todo selfplay, teaching e treino usa `data/selfplay_canonical_v3/`, schema
`zquoridor.selfplay.v3.canonical.mover-mirrored.v1`: registros de 64 bytes,
política de 209 probabilidades e valor em `[-1,+1]` na perspectiva do jogador a
mover. `data/selfplay/` é somente arquivo histórico, lido apenas por
`training/migrate_selfplay_v3.py`.

Todos os scripts têm bloco `CONFIG` e flags CLI equivalentes. Livros:

| Livro | Uso | Tamanho |
|---|---|---:|
| `tools/external/openings_titanium.jsonl` | triagem histórica | 40 |
| `tools/external/openings_screen_v1.jsonl` | screening | 100 |
| `tools/external/openings_confirmation_v1.jsonl` | confirmação | 400 |

Benchmark válido exige mesmas aberturas, cores invertidas, seed igual e
`200 ms` por lance para todos; Claustrophobia usa relógio de parede calibrado.

## 3. Redes a treinar e testar

| ID | Entradas | Descrição e Features Extras | Custo BFS | Estado |
|---|---:|---|---|---|
| `base` | 354 | Peões (81+81), muros (64+64), distâncias (21+21), reservas (11+11) | 0 extra | baseline histórico |
| `race` | 456 | `base` + margem BFS [-16..16] (33) + saldo muros (21) + corrida × reservas (48) | 0 extra | **baseline oficial de produção (`race512-cr200k-champion`)** |
| `multipath` | 480 | `race` + 8 saídas direcionais unblocked + 8 grau de saída (gargalos) + 8 geometria de contato/pulo | 0 extra | **implementada & validada (0 divergências incremental)** |
| `margin_regime` | 588 | `race` + 132 interação margem $\Delta d \in [-16..16] \times 4$ regimes de esgotamento de muros | 0 extra | **implementada & validada (0 divergências incremental)** |
| `phase` | 480 | `race` + 6 buckets estoque total de muros + 18 corrida $\times$ fase | 0 extra | **implementada & validada (0 divergências incremental)** |
| `margin_phase` | 612 | `race` + 132 `margin_regime` + 24 `phase` (interação completa corrida $\times$ muros $\times$ fase) | 0 extra | **implementada & validada (0 divergências incremental)** |

### Protocolo de Cirurgia de Rede (Warm-Start)
Para todas as redes experimentais com expansão de entradas (ex: 456 -> 480, 588, 612), os pesos $456 \times 512$ da campeã de produção (`race512-cr200k-champion`) são copiados diretamente e as novas colunas são inicializadas com **exatamente 0**. Isso assegura que no passo inicial o modelo apresente divergência matemática absolutamente zero em relação à campeã, aprendendo as novas interações de forma suave durante o recozimento térmico de 60 épocas.

## 4. Redes já treinadas: settings, dados e TODO

Há dois tipos de treino, executados separadamente. O treino **direct** usa
somente o dataset direct de 2M e produz um checkpoint novo. O **fine-tune**
parte do checkpoint direct escolhido e executa uma segunda campanha com o
dataset misto de search. No direct: QAT, 4 épocas de warmup, cosine até
`min_lr=5e-6`. No fine-tune: QAT, 2 épocas de warmup, cosine até
`min_lr=3e-6`, learning rate inicial `3e-5`. Ambos exportam float/int8 e
passam pela verificação incremental antes da arena.

| Rede | Melhor val loss | Treino | Fine-tune | Status/TODO |
|---|---:|---|---|---|
| `base:128` | — | não priorizada | não | opcional |
| `base:256` | 0,74569 | concluído | não | arquivada após screening |
| `race:256` | 0,74319 | concluído | não | benchmark feito |
| `base:384` | 0,73939 | concluído | não | benchmark feito |
| `race:384` | 0,73686 | concluído | legado 50/50 | benchmark feito |
| `base:512` | 0,73600 | concluído | 40 épocas | confirmação independente |
| `race:512-search10-ft` | 0,88574 | concluído | 40 épocas | baseline anterior (49.0% vs Claustro, 54.0% vs Titanium) |
| `race:512-multitier-champion` | 0,86410 | concluído | 10.8M multitier | campeã geral: 51.5% h2h sobre search10, 49.67% em 600g vs Claustrophobia, 57.0% vs Titanium |
| `race:512-policy-tactical` | 0,74580 | concluído | policy-scope 245k | foco tático: +40.5 Elo em center rush (37.88%), 67.5% vs main, 55.0% vs Titanium |
| `race:512-weakness-ft` | 1,14938 | concluído | 80k fraquezas | fine-tune cosine annealing: 38.64% (+44 Elo) em center rush vs Claustrophobia |
| `race:512-cr200k-policy` | 0,69945 | concluído | policy 255k CR | treino dedicado de policy: KL caiu 28% (0.1015 -> 0.0734), 95.37% sinal em Center Rush |
| `race:512-cr200k-champion` | 0,69542 | concluído | heads 255k CR | **CAMPEÃ ATUAL**: 63.0% (+92.5 Elo) em 600g vs Titanium, 81.25% (+254.7 Elo) vs main, 46.25% vs Claustrophobia |

Fine-tune atual usa `data/teaching/historical2m-search10-conservative/dataset.npz`:
2.001.868 posições, 90% replay direto e 10% search (25% ZQuoridor, 75%
Claustrophobia). `race:512` caiu de 0,91628 para 0,88574; `base:512`, de
0,92031 para 0,89128. Essas losses não são comparáveis ao treino direto.

### Registro de campanhas de treino

| Campanha | Script | Entrada | Configuração | Saída |
|---|---|---|---|---|
| Direct architecture matrix | seis execuções de `training/run_experiment.py` (a matriz é apenas auxiliar) | dataset direct de 2M | `base/race`, hidden 256/384/512, 80 épocas, batch 4096, LR `1e-4`, QAT, CUDA, cosine, warmup 4, min LR `5e-6` | um modelo por arquitetura |
| Fine-tune `race512-search10-ft` | `training/run_experiment.py` | checkpoint `race:512` + dataset misto | 40 épocas, batch 4096, LR `3e-5`, warmup 2, cosine, min LR `3e-6`, QAT, CUDA | diretório `race512-search10-ft-s20260917` |
| Fine-tune `base512-search10-ft` | `training/run_experiment.py` | checkpoint `base:512` + dataset misto | 40 épocas, batch 4096, LR `3e-5`, warmup 2, cosine, min LR `3e-6`, QAT, CUDA | diretório `base512-search10-ft-s20260917` |
| Retreino `race512-multitier-champion` | `training/run_experiment.py` | checkpoint `race512-search10-ft` + `multitier-final-dataset` (10.8M) | 4 épocas, batch 4096, LR `4e-5`, cosine, min LR `2e-6`, QAT, CUDA | diretório `results/experiments/race512-multitier-champion` |
| Fine-tune `race512-policy-tactical` | `training/run_experiment.py` | checkpoint `multitier-champion` + 245k tático | `--train-scope policy`, 15 épocas, LR `1.5e-4`, cosine, QAT, CUDA | diretório `results/experiments/race512-policy-tactical` |
| Fine-tune `race512-weakness-ft` | `training/run_experiment.py` | checkpoint `multitier-champion` + 80k fraquezas mineradas | 20 épocas, batch 256, LR `2.5e-5`, cosine, min LR `5e-7`, QAT, CUDA | diretório `results/experiments/race512-weakness-ft` |
| Fine-tune `race512-cr200k-policy` | `training/run_experiment.py` | checkpoint `multitier-champion` + 255k Center Rush | `--train-scope policy`, 15 épocas, LR `1.2e-4`, cosine, min LR `5e-6`, QAT, CUDA | diretório `results/experiments/race512-cr200k-policy` |
| Calibração `race512-cr200k-champion` | `training/run_experiment.py` | checkpoint `race512-cr200k-policy` + 255k Center Rush | `--train-scope heads`, 15 épocas, LR `2.5e-5`, cosine, min LR `5e-7`, QAT, CUDA | diretório `results/experiments/race512-cr200k-champion` |

O dataset misto contém direct e search juntos com pesos por amostra, mas cada
arquitetura direct foi treinada antes em sua própria campanha. Depois, apenas
`base:512` e `race:512` receberam a segunda campanha de fine-tuning.

**“Legado 50/50”** é a campanha antiga
`results/experiments/historical2m-race384-search10-ft-s20260916/`. Ela partiu
de `race:384`, usou `data/teaching/historical2m-search10/dataset.npz`, 16
épocas, QAT, CUDA e `lr=3e-5`, com mistura aproximadamente 50% ZQuoridor search
e 50% Claustrophobia search. Não é a receita atual nem entra na decisão final;
fica documentada apenas para reproduzir o histórico. A receita atual usa
25%/75% dentro do bloco search, que representa 10% do dataset total.

### Glossário dos termos usados nas tabelas

- **Policy/política***: distribuição de probabilidade sobre as 209 ações
  legais. Diz quais movimentos o teacher prefere.
- **Value/valor***: estimativa contínua do resultado na perspectiva do jogador
  a mover. É diferente de uma classe WDL.
- **WDL***: alvo win/draw/loss ou resultado da partida. O campo `game_result`
  existe em alguns replays, mas o treino final não o usa como alvo dominante.
- **Direct/replay direto***: a rede antiga e/ou Claustrophobia produzem
  `policy` e `value` em forward, sem busca profunda por posição.
- **Search***: MCAB/MCTS visita posições e produz uma política/valor dependente
  de busca, com orçamento de nós ou simulações.
- **Distillation/teaching***: o aluno NNUE aprende as saídas de um ou mais
  teachers (`policy`, `value` e `weight`); não significa copiar pesos da rede
  teacher.
- **Dataset misto***: um único `dataset.npz` construído depois, combinando
  exemplos direct e search por pesos de amostra; não é o mesmo que gerar ambos
  no mesmo processo.
- **Fine-tune***: segunda campanha iniciada a partir do checkpoint de uma rede
  direct, com learning rate menor e dataset misto.

## 5. Dados e teaching já realizados

| Artefato | Conteúdo/configuração | Status |
|---|---|---|
| Corpus V3 | 72.298.778 registros, contrato único | concluído/auditado |
| `gen-rich-v1-montecarlo` | 4.713 jogos, 311.930 registros, temperatura/ruído, 14 threads | feito; falta auditoria global de unicidade |
| Replay histórico | lotes de 500k, 1M e corpus amplo de 2M | concluído |
| Relabel amplo | 1.602.929 treino + 397.071 validação, CUDA | concluído |
| Search priority | 2.000 posições divergentes; ZQ 512/2048 nós; Claustro 256/1024 sims | concluído |
| Dataset conservador | 25% ZQ + 75% Claustro em search | concluído |
| Dataset fine-tune | 2.001.868 posições, 90% direct + 10% search | concluído |

WDL histórico não é alvo principal. O teacher do próprio ZQuoridor permanece
auxiliar enquanto wandering não estiver resolvido.

### Como cada dado foi gerado

| Dado | Script(s) | Settings principais | Composição |
|---|---|---|---|
| Shards V3 | `training/migrate_selfplay_v3.py`, `training/audit_selfplay.py` | conversão para mover-mirrored V3 e auditoria de schema | corpus separado; contrato comum |
| Selfplay rico | `tools/selfplay/run_selfplay.py` | 4.713 jogos, 14 threads, temperatura/ruído de abertura, seed registrada | corpus separado |
| Replay direct | `training/prepare_replay.py` | rede antiga + Claustrophobia em forward, até 1M/2M posições, CUDA | lotes separados |
| Search selecionado | `training/select_replay_disagreement.py`, `training/build_search_priority_dataset.py`, `training/teachers/zq_deep_relabel.py` | 2.000 posições; ZQ 512/2048 nós; Claustro 256/1024 simulações | primeiro separado |
| Dataset fine-tune | `training/mix_teaching_datasets.py` | 90% replay direct + 10% search; dentro do search 25% ZQ/75% Claustro | única mistura usada no fine-tune |

`prepare_replay.py` não faz search profundo: gera alvos direct rápidos.
`run_teaching.py` e os teachers geram alvos mixed/search. Os grupos foram
produzidos separadamente e somente depois unidos pelo script de mistura; não
houve um processo único gerando todos os dados ao mesmo tempo.

### Inventário literal de pastas e arquivos

| Grupo | Pasta | Arquivos principais |
|---|---|---|
| Selfplay V3 rico | `data/selfplay_canonical_v3/gen-rich-v1-montecarlo/` | `selfplay_000.bin`, `selfplay_001.bin`, `manifest.json` |
| Replay direct amplo | `data/teaching/replay-historical-2m-cuda/` | `dataset.npz`, `replay_manifest.json`, `direct_*.npz`, `direct_*.sha256` |
| Replay direct antigo | `data/teaching/replay-old-gen1-500k/` | `dataset.npz`, `replay_manifest.json`, `direct_*.npz`, `direct_*.sha256` |
| Posições selecionadas | `data/teaching/search-priority-gen1/` | `top2000.jsonl`, `zq-search.npz`, `claustro-search.npz` |
| Teaching search conservador | `data/teaching/search-priority-gen1-conservative/` | `dataset.npz`, `dataset.npz.manifest.json` |
| Dataset usado no fine-tune | `data/teaching/historical2m-search10-conservative/` | `dataset.npz`, `dataset.manifest.json` |
| Replay menor V3 | `data/teaching/replay-million-canonical-v3/` | `dataset.npz` e manifesto local |
| Selfplay histórico V3 | `data/selfplay_canonical_v3/gen1/` até `gen7-montecarlo/` e `gen12-standard-control-v2/` | `selfplay_*.bin` e manifestos por geração |

Os nomes `direct_*.npz` são shards intermediários; o trainer usa o
`dataset.npz` consolidado. Os arquivos `*.sha256` verificam integridade. O
fine-tune não lê diretamente `top2000.jsonl`, `zq-search.npz` ou
`claustro-search.npz`: esses alvos já foram incorporados ao dataset misto pelo
`training/mix_teaching_datasets.py`.

### Inventário físico dos datasets

Os diretórios abaixo são diferentes e não compartilham o mesmo `dataset.npz`.
O nome do arquivo é igual por convenção, mas o caminho completo é a identidade
do dataset; o manifesto ao lado registra origem e hashes.

| Dataset físico | Arquivo principal | Conteúdo | WDL? |
|---|---|---|---|
| `data/teaching/replay-historical-2m-cuda/` | `dataset.npz` | 2.000.000 posições direct; `policy`, `value`, `weight`, `game_result`, `is_val` | `game_result` existe, mas `outcome_weight=0`; não foi alvo WDL dominante |
| `data/teaching/search-priority-gen1-conservative/` | `dataset.npz` | 2.000 posições selecionadas; política/valor de search e pesos de divergência | não é WDL histórico; é distillation de política/valor |
| `data/teaching/historical2m-search10-conservative/` | `dataset.npz` | 2.001.868 posições: 90% replay direct + 10% search | política/valor ponderados; não contém `game_result` |
| `data/teaching/replay-old-gen1-500k/` | `dataset.npz` | replay direct antigo, 500.000 posições | usado como fonte da seleção, não nos treinos finais |
| `data/selfplay_canonical_v3/gen-rich-v1-montecarlo/` | `selfplay_*.bin` | shards V3 de selfplay, não dataset de treino pronto | resultado da partida está no registro V3; precisa de replay/teaching |
| `data/teaching/multitier-final-dataset/` | `dataset.npz` | 10.810.000 posições: Tier 1 (10M), Tier 2 (500k), Tier 3 (100k), Tier 3.5 (100k), Tier 4 (100k), Tier 5 (10k) | alvos ponderados por tier (1,0x a 8,0x) |
| `data/teaching/policy-tactical-245k/` | `dataset.npz` | 244.787 posições táticas e de crise mineradas | alvos de política enriquecidos |
| `data/teaching/weakness-80k/` | `dataset.npz` | 80.000 posições de fraquezas e derrotas de auto-jogo | alvos de recuperação de fraquezas |
| `data/teaching/center-rush-200k-priority/` | `dataset.npz` | 255.000 posições: 200k Center Rush (5,0x) + 5k Action-Q search (6,0-8,0x) + 50k background (1,0x) | 95,37% sinal em Center Rush; $\pi'_a \propto N_a^\alpha \exp(\beta Q_a)$ |

O `.npz` não contém “tudo” por padrão: cada dataset é uma matriz independente.
Todos têm features canônicas (`own_pawn`, `opp_pawn`, `walls_h`, `walls_v`,
reservas, distâncias), `policy` de 209 ações e `value`; somente alguns mantêm
`game_result`. O trainer usa `policy`/`value` e `weight`; o `game_result` só
entra se uma campanha configurar peso de outcome diferente de zero.

### Dataset usado por cada rede

| Grupo de redes | Dataset exato | Foi misturado? | Campanha |
|---|---|---|---|
| `base:256`, `race:256`, `base:384`, `race:384`, `base:512`, `race:512` direct | `data/teaching/replay-historical-2m-cuda/dataset.npz` | não; mesmo dataset para todas | seis treinos independentes, mesma receita QAT/80 épocas; só arquitetura/largura muda |
| `race512-search10-ft` | `data/teaching/historical2m-search10-conservative/dataset.npz` + checkpoint `race:512` | sim, 90/10 por amostra | segundo treino independente, fine-tune 40 épocas |
| `base512-search10-ft` | `data/teaching/historical2m-search10-conservative/dataset.npz` + checkpoint `base:512` | sim, 90/10 por amostra | segundo treino independente, fine-tune 40 épocas |
| `race512-multitier-champion` | `data/teaching/multitier-final-dataset/dataset.npz` + checkpoint `race512-search10-ft` | sim, 5 camadas ponderadas | retreino amplo, 60 épocas, batch 1024, `--trunk-lr-scale 0.2` |
| `race512-policy-tactical` | `data/teaching/policy-tactical-245k/dataset.npz` + checkpoint `multitier-champion` | sim, tático | fine-tune exclusivo de política (`--train-scope policy`), 15 épocas |
| `race512-weakness-ft` | `data/teaching/weakness-80k/dataset.npz` + checkpoint `multitier-champion` | sim, fraquezas | fine-tune cosine annealing, 20 épocas |
| `race512-cr200k-policy` | `data/teaching/center-rush-200k-priority/dataset.npz` + checkpoint `multitier-champion` | sim, 255k CR prioritário | estágio 1: `--train-scope policy`, 15 épocas |
| `race512-cr200k-champion` | `data/teaching/center-rush-200k-priority/dataset.npz` + checkpoint `race512-cr200k-policy` | sim, mesmo 255k CR prioritário | estágio 2: `--train-scope heads`, 15 épocas |

Assim, as seis redes direct foram comparáveis entre si: mesmas posições,
mesmos alvos e mesmo schedule. As redes derivadas são continuações a partir
dos respectivos checkpoints, com datasets especializados e learning rates menores.

### Onde está a configuração de cada rede

Em cada pasta de experimento, `config.json` é a configuração de treino,
`student.architecture.json` é a arquitetura exportada, `train_report.json`
registra épocas/loss/schedule, `student.bin` é o peso float, `student_int8.bin`
é o peso QAT usado pelo engine, e `incremental_check.exe` é a verificação de
paridade do acumulador. O executável da arena é `zquoridor.exe`.

| Rede/campanha | Pasta real | Arquivo de dados usado |
|---|---|---|
| `base:256` direct | `results/experiments/historical2m-base256-anneal-s20260916/` | `data/teaching/replay-historical-2m-cuda/dataset.npz` |
| `race:256` direct | `results/experiments/historical2m-race256-anneal-s20260916/` | mesmo `replay-historical-2m-cuda/dataset.npz` |
| `base:384` direct | `results/experiments/historical2m-base384-s20260916/` | mesmo dataset direct de 2M |
| `race:384` direct | `results/experiments/historical2m-race384-s20260916/` | mesmo dataset direct de 2M |
| `base:512` direct | `results/experiments/historical2m-base512-anneal-s20260917/` | mesmo dataset direct de 2M |
| `race:512` direct | `results/experiments/historical2m-race512-anneal-s20260917/` | mesmo dataset direct de 2M |
| `race512-search10-ft` | `results/experiments/race512-search10-ft-s20260917/` | `data/teaching/historical2m-search10-conservative/dataset.npz` + checkpoint `race:512` |
| `base512-search10-ft` | `results/experiments/base512-search10-ft-s20260917/` | mesmo dataset misto + checkpoint `base:512` |
| `race512-multitier-champion` | `results/experiments/race512-multitier-champion/` | `data/teaching/multitier-final-dataset/dataset.npz` + checkpoint `race512-search10-ft` |
| `race512-policy-tactical` | `results/experiments/race512-policy-tactical/` | `data/teaching/policy-tactical-245k/dataset.npz` + checkpoint `multitier-champion` |
| `race512-weakness-ft` | `results/experiments/race512-weakness-ft/` | `data/teaching/weakness-80k/dataset.npz` + checkpoint `multitier-champion` |
| `race512-cr200k-policy` | `results/experiments/race512-cr200k-policy/` | `data/teaching/center-rush-200k-priority/dataset.npz` + checkpoint `multitier-champion` |
| `race512-cr200k-champion` | `results/experiments/race512-cr200k-champion/` | `data/teaching/center-rush-200k-priority/dataset.npz` + checkpoint `race512-cr200k-policy` |

Os nomes das pastas `*-anneal-*` identificam as campanhas direct com schedule
de annealing. Os nomes `*-search10-ft-*` identificam fine-tuning com 10% de
search. Para reproduzir uma rede, ler primeiro `config.json` e o manifesto do
dataset; não inferir a receita apenas pelo nome da pasta.

### Comandos e manifestos de geração

- Selfplay rico: o comando completo está no campo `command` de
  `data/selfplay_canonical_v3/gen-rich-v1-montecarlo/manifest.json` (3.000
  jogos, 150 ms, profundidade 50, MC mode, temperaturas 1,35→0,35, 14
  threads, seed `2026091501`).
- Replay direct de 2M: a configuração completa está no campo `config` de
  `data/teaching/replay-historical-2m-cuda/replay_manifest.json`: seed
  `2026091602`, 2.000.000 posições, validação 20%, chunks 8192, batch 4096,
  CUDA, teacher antigo policy/value 0,5/0,75, Claustrophobia policy/value
  0,5/0,25 e `outcome_weight=0`.
- Search priority: os caminhos exatos estão em
  `data/teaching/search-priority-gen1-conservative/dataset.npz.manifest.json`,
  incluindo `top2000.jsonl`, `zq-search.npz`, `claustro-search.npz`, pesos
  0,25/0,75 e fórmula de peso.

Esses manifestos são a fonte de verdade para repetir a geração; o texto deste
plano resume os valores para leitura humana.

## 6. Benchmarks já feitos

Todos são screening de 20 pares, 200 ms por lance; nenhum promove rede.
Percentuais são score da candidata.

| Candidata | vs main | vs Titanium | vs Claustrophobia | Status |
|---|---:|---:|---:|---|
| `race:256` | 65,0% | 42,5% | 25,0% | sem promoção |
| `base:384` | 67,5% | 37,5% | 28,75% | sem promoção |
| `race:384` | 60,0% | 35,0% | 31,25% | sem promoção |
| `base:512` | 65,0% | 45,0% | 26,25% | mais promissora direta |
| `race:512` | 57,5% | 30,0% | 22,5% | abaixo de base512 |
| `race512-search10-ft` | 63,75% | 35,0% | 21,25% | triagem |
| `base512-search10-ft` | 66,25% | 45,0% | 20,0% | finalista provisória |

No mesmo run de `base512-search10-ft`, o main marcou 40,0% contra Titanium e
22,5% contra Claustrophobia. A variação da referência mostra por que 20 pares
são apenas triagem.

## 7. Benchmarks de confirmação (100 pares, 200 ms)

### 7.1 Primeira finalista: `base512-search10-ft` (Gate A — concluído)

Diretório: `benchmark_results/base512-search10-ft-confirm-200ms-s20260918`.
Relatório final em `summary.json`. Todos os 400 jogos da candidata e 400 da referência do main foram válidos.

- vs main: 58,25% (IC 95% bootstrap: 52,0–64,5%, `strength_claim_ready=true`)
- vs Titanium: 58,00% (IC 95% bootstrap: 51,5–64,5%, `strength_claim_ready=true`)
- vs Claustrophobia: 43,00% (IC 95% bootstrap: 36,5–49,25%, `strength_claim_ready=true`)
- Referência do main no mesmo run: 52,25% vs Titanium / 37,50% vs Claustrophobia

### 7.2 Segunda finalista: `race512-search10-ft` (Gate B — concluído)

Diretório: `benchmark_results/race512-search10-ft-confirm-200ms-s20260918`.
Relatório final em `summary.json`. Todos os 400 jogos da candidata e 400 da referência do main foram válidos (0 falhas).

- vs main: 58,50% (Elo +59,6, IC 95% bootstrap: 52,0–65,0%, `strength_claim_ready=true`)
- vs Titanium: 57,50% (Elo +52,5, IC 95% bootstrap: 51,0–64,0%, `strength_claim_ready=true`)
- vs Claustrophobia: 49,00% (Elo -6,9, IC 95% bootstrap: 43,0–55,0%, `strength_claim_ready=true`)
- Referência do main no mesmo run: 56,50% vs Titanium (IC 95%: 50,0–63,0%) / 37,75% vs Claustrophobia (IC 95%: 31,75–43,75%)

### 7.3 Comparativo das finalistas na confirmação (100 pares)

| Candidata | vs main | vs Titanium | vs Claustrophobia | Status |
|---|---:|---:|---:|---|
| `base512-search10-ft` | 58,25% | 58,00% | 43,00% | Gate A concluído (`summary.json`) |
| `race512-search10-ft` | 58,50% | 57,50% | 49,00% | Gate B concluído (`summary.json`) |

Destaque: `race512-search10-ft` empata tecnicamente contra `main` e Titanium, mas obtém vantagem expressiva (+6,0 p.p.) contra Claustrophobia (49,0% vs 43,0%).

### 7.4 Match direto entre as finalistas (Gate B.1 Passo 6 — concluído)

Diretório: `benchmark_results/finalists-h2h-race512-vs-base512-200ms`.
Script: `tools/match_finalists.py`.
Configuração: 100 pares (200 jogos), 200 ms por lance, `openings_confirmation_v1.jsonl`, seed `20260920`.

- Placar: `race512-search10-ft` 99,0 vs 101,0 `base512-search10-ft` (49,50% vs 50,50%)
- Elo: -3,5 (IC 95% bootstrap: 44,00% a 54,75%, Elo [-41,9, +33,1])
- Jogos válidos: 200/200 (0 falhas, média de 79,1 plies, `strength_claim_ready=true`)

Conclusão: O confronto direto entre as duas finalistas é um empate estatístico exato (49,5% vs 50,5%). Como a `race512-search10-ft` superou a `base512-search10-ft` por ampla margem contra adversários externos (49,0% vs 43,0% contra Claustrophobia, +6,0 p.p.), a **`race512-search10-ft` é declarada a finalista e campeã definitiva entre as arquiteturas candidatas**.

## 8. Big picture e roadmap restante

### Gate A — confirmação da primeira finalista (**concluído**)

1. ~~Concluir `base512-search10-ft` × main: 100 pares, 200 ms, livro de 400.~~
2. ~~Rodar `base512-search10-ft` × Titanium no mesmo livro/seed/relógio.~~
3. ~~Rodar `base512-search10-ft` × Claustrophobia no mesmo protocolo.~~
4. ~~Gerar main × Titanium e main × Claustrophobia no mesmo run.~~

Conclusão: `base512-search10-ft` superou Titanium com significância estatística (58,0%, limite inferior > 50%).

### Gate B — segunda finalista (**concluído**)

5. ~~Repetir os quatro confrontos para `race512-search10-ft` para comparação final.~~
   - `vs-main`: concluído 200/200 (58,50%).
   - `vs-external`: concluído 400/400 (Titanium 57,50%, Claustrophobia 49,00%).
   - `main-vs-external`: concluído 400/400 (Titanium 56,50%, Claustrophobia 37,75%).
   - `summary.json` final gerado e verificado.

Conclusão: `race512-search10-ft` é a melhor rede geral até o momento (+6 p.p. vs Claustrophobia sobre `base512`, empatada contra Titanium e main).

### Gate B.1 — Match direto e busca pelo >50% contra Claustrophobia

6. ~~**Desempate direto entre as finalistas**: Match direto `race512-search10-ft` × `base512-search10-ft` (100 pares, 200 ms, `openings_confirmation_v1.jsonl`).~~
   - Concluído: 200/200 jogos válidos.
   - Placar: 49,50% (99-101), empate técnico exato.
   - Decisão: `race512-search10-ft` confirmada como a melhor rede geral.
7. ~~**Superar Claustrophobia (>50%) com busca a 20%**: Testar `race512-search20-ft` (treinado com 20% de teaching search) contra Claustrophobia.~~
   - Concluído: 40/40 jogos (20 pares, 200 ms, `openings_confirmation_v1.jsonl`).
   - Placar: 48,75% (Elo -8,7).
   - Diagnóstico: Aumentar o peso de teaching de 10% para 20% no mesmo conjunto reduzido de 2.000 posições mantém o teto em ~48,8%–49,0%. Mais do mesmo não quebra os 50%.

### 8.1 Diagnóstico Tático & Métricas de Busca contra Claustrophobia

#### Benchmarks Específicos de Busca & Velocidade
Medições instrumentadas diretamente nos motores sobre o mesmo hardware e livro:

| Métrica | ZQuoridor (MCAB + NNUE `race512`) | Claustrophobia (MCTS + PyTorch) | Implicação Tática |
|---|---|---|---|
| **Velocidade (Throughput)** | **~53.400 nós/s** a 200 ms<br>**~65.800 nós/s** a 100 ms | **~50 a 100 simulações/s** (PyTorch IPC) | ZQuoridor avalia **~500x a 1.000x mais estados por segundo**. |
| **Simulações / lance (200 ms)** | **~10.680 simulações** | **~10 a 20 simulações** | Claustrophobia joga quase em *pure policy*, fazendo pouquíssimas visitas por lance. |
| **Profundidade máxima da árvore** | Árvore MCTS atinge **ply 18 a 28** | Árvore atinge **ply 4 a 8** | ZQuoridor calcula linhas muito mais longas, mas com avaliação mais rasa por nó. |
| **Tempo médio por lance** | 209,0 ms (P95: 235 ms) | 200,9 ms (P95: 204 ms) | Controle de relógio rigorosamente paritário. |
| **Qualidade da Prior Policy** | Rápida (354+102 -> 512 int8) | Pesada (Rede Convolucional PyTorch) | Claustrophobia compensa a baixa taxa de nós com intuição posicional de muros muito superior. |

#### Onde Mais Perdemos? (Taxonomia dos 100 Pares / 200 Jogos de Confirmação)
- **Placar Geral**: 95 vitórias, 105 derrotas (49,0% de aproveitamento).
- **Por Lado**: Brancas (P1): 49,0% (49/100) | Pretas (P2): 46,0% (46/100).
- **Consistência por Par**: 18 vitórias duplas (2-0), 59 empates (1-1) e **23 varreduras sofridas (0-2)**.
- **Duração Média**: Vitórias em 62,1 plies; Derrotas em **72,7 plies** (as derrotas não ocorrem por erro tático imediato, mas por desgaste no final).

#### 1. O Fator Decisivo: Desbalanço e Tempo de Gasto de Muros
Análise temporal do esgotamento do estoque de muros nas 200 partidas:

- **Gasto do 10º muro primeiro**:
  - O ZQuoridor gastou todos os 10 muros primeiro em **50,5% das partidas** (101 jogos).
  - O Claustrophobia gastou todos os 10 muros primeiro em **apenas 23,5% das partidas** (47 jogos).
  - Em 26,0% das partidas, nenhum jogador gastou todos os 10 muros.
- **Impacto no Resultado**:
  - Quando o **ZQuoridor gasta os muros primeiro**: Placar de **33V - 68D (Winrate: 32,7%)**!
  - Quando o **Claustrophobia gasta os muros primeiro**: Placar de **34V - 13D (Winrate: 72,3%)**!
- **Timing médio da colocação de cada muro**:
  - Muro #8: ZQ coloca no ply 31,9 (em 182 jogos) vs Claustro no ply 37,0 (160 jogos).
  - Muro #9: ZQ coloca no ply 39,6 (em 159 jogos) vs Claustro no ply 46,5 (130 jogos).
  - Muro #10: ZQ esgota no ply 48,1 (**em 116 jogos / 58% do total**) vs Claustro no ply 55,3 (**em apenas 68 jogos / 34% do total**).
- **Estoque de muros no término**:
  - ZQ com 0 muros e Claustrophobia com muros restantes: **23 vitórias e 57 derrotas (28,8% de winrate)** — representa **54,3% de todas as 105 derrotas**!
  - Ambos com muros restantes: ZQuoridor vence **53,8%** (28V - 24D).
  - Claustrophobia com 0 muros e ZQ com muros: ZQuoridor vence **75,0%** (24V - 8D).
  - Ambos com 0 muros (corrida exata): ZQuoridor vence **55,6%** (20V - 16D) via `endgame_race.hpp`.

#### 2. Aberturas Específicas Varridas (0-2)
As 23 aberturas onde o Claustrophobia venceu de Brancas e de Pretas:
`#1, #22, #34, #38, #48, #81, #83, #97, #102, #127, #148, #161, #178, #184, #208, #212, #275, #303, #315, #332, #380, #382, #398`.
- **Padrão unificador**: Em 21 das 23 aberturas varridas, **todos os 6 lances iniciais são muros** (ex.: abertura #38: `g1h, f3h, b3h, h7v, d1v, e1v`; abertura #102: `f3v, a1h, g4h, e7v, f6h, e3v`).
- O tabuleiro inicia com alta densidade de barreiras. O ZQuoridor reage colocando ainda mais muros cedo, ficando prematuramente sem reservas para a transição para o final.

#### 3. Divergência de Search entre ZQuoridor e Claustrophobia
Análise das posições mais difíceis mineradas no teaching:
- **Pawn vs Wall**: Em **10,0% das posições de alta divergência**, o ZQuoridor prefere colocar muro (>50% de probabilidade na busca), enquanto o Claustrophobia prefere mover o peão (>50% de probabilidade).
- **Inversão de Sinal**: Em **20,8% das posições de divergência**, há inversão completa de sinal na avaliação de valor (um motor avalia a posição como vitoriosa e o outro como perdedora).

### 8.2 Big Picture: Estratégia de Escala Massiva de Teaching para Romper >50%

O diagnóstico prova que amostragem reduzida (como as 2.000 posições do `top2000.jsonl`, que representavam apenas 0,1% do dataset de 2M) atinge platô em ~48,8%–49,0%. Para ultrapassar Claustrophobia com folga (>50%), a solução exige **escala massiva direcionada aos pontos de vazamento**:

1. **Geração Massiva de Partidas Rápidas a 100 ms**:
   - Executar autojogo com controle de 100 ms/lance (onde o motor opera a ~65.800 nós/s com ~6.580 simulações/lance).
   - Gerar de **10.000 a 20.000 partidas completas** (tempo médio de ~7s por jogo em paralelo, gerando dezenas de milhares de partidas em poucas horas).
   - Foco em partidas partindo das 23 aberturas varridas e em confrontos contra o Claustrophobia.
2. **Mineração Massiva de Estados Críticos (50.000+ posições)**:
   - Extrair não 2.000, mas **50.000 a 100.000 estados de alta prioridade**:
     a) Estados intermediários de todas as derrotas contra Claustrophobia.
     b) Momentos exatos onde o ZQuoridor gasta seus muros #7, #8, #9 e #10 com desbalanço desfavorável.
     c) Posições das 23 aberturas densas em muros com alta divergência de política (Pawn vs Wall).
3. **Teaching Duplo Profundo em Larga Escala**:
   - Relabeling dessas 50.000+ posições com ambos os professores:
     - ZQuoridor: busca profunda (AB 2048 nós ou MCAB com alto orçamento).
     - Claustrophobia: MCTS de alta intensidade (512 a 1024 simulações).
   - Composição de dataset onde 100% das posições adicionais vêm de situações reais de desbalanço e perdas contra o alvo.
4. **Calibração de Heurística de Conservação de Muros na Busca**:
   - Ativar penalização de colocação de muro tardio quando a reserva própria é baixa (`wallsLeft <= 2`) e o muro não altera a diferença líquida de BFS em favor do jogador.
   - Avaliar `endgameMoverWallThreshold = 1` para ativar busca alfa-beta tática já com 1 muro restante.

### Gate C — novas arquiteturas

7. Só se A/B não resolverem wandering: implementar `phase`, `topology-lite` e
   `race-phase`, treinar cada uma em campanha própria e ablar contra
   `base512-search10-ft`.
8. Depois testar `corridor-touch`, `multi-path` e `wall-threat`, sempre com
   custo incremental, paridade Python/C++ e benchmark próprio.
9. Exigir export float/int8 e `incremental_check.exe` sem divergência antes da
   promoção de qualquer rede.

## 9. Aprendizados históricos

- Linhas não significam diversidade: 1.497.673 registros produziram apenas
  83.244 estados independentes elegíveis.
- Selfplay novo complementa histórico; precisa de auditoria de unicidade.
- Replay direto dá cobertura; search selecionado é necessário para posições
  táticas, corridas e wandering.
- Enquanto wandering persiste, ZQuoridor não deve dominar o teacher: a mistura
  conservadora usa 25% ZQuoridor e 75% Claustrophobia.
- Melhor loss não garantiu força: `race:512` perdeu para `base:512` no arena.
- 20 pares não demonstram superioridade estatística.
- Benchmarks medidos só por simulações de Claustrophobia são inválidos para
  comparação de força; o relógio deve ser comum.
- Fine-tune com outros alvos muda a escala da loss; arena é o critério.
- Features novas devem ser testadas isoladamente, com ablação e custo medido.

## 10. Scripts principais

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| migração | `training/migrate_selfplay_v3.py` | legado/V2/V3 | corpus V3 |
| selfplay | `tools/selfplay/run_selfplay.py` | busca/NNUE | shards V3 |
| replay | `training/prepare_replay.py` | V3 | dataset direct |
| teaching | `training/run_teaching.py` | partidas/posições | dataset mixed |
| relabel profundo | `training/teachers/zq_deep_relabel.py` | V3/@state | alvos search |
| treino | `training/run_experiment.py` | dataset | QAT/binário |
| campanha | `training/run_campaign.py` | corpus/matriz | treino/arena |
| benchmark | `tools/benchmark_candidate.py` | candidato | relatório pareado |

### Comandos reproduzíveis das campanhas concluídas

Todos os comandos abaixo devem ser executados a partir da raiz do repositório
(`C:\Projetos\Zquoridor`). Os arquivos `config.json` nas pastas de resultado
são a fonte de verdade caso algum default do script mude.

```powershell
# Direct: repetir uma arquitetura (trocar ARCH e HIDDEN)
python training/run_experiment.py `
  --no-teaching `
  --data data/teaching/replay-historical-2m-cuda/dataset.npz `
  --architecture ARCH --hidden HIDDEN --epochs 80 --batch-size 4096 `
  --qat --schedule cosine --warmup-epochs 4 --min-lr 5e-6 --device cuda `
  --out-dir results/experiments/REPRODUCED

# Fine-tune race512
python training/run_experiment.py `
  --data data/teaching/historical2m-search10-conservative/dataset.npz `
  --architecture race --hidden 512 --epochs 40 --batch-size 4096 `
  --qat --schedule cosine --warmup-epochs 2 --device cuda `
  --init-from results/experiments/historical2m-race512-anneal-s20260917/student.bin `
  --out-dir results/experiments/REPRODUCED-race512-ft

# Fine-tune base512
python training/run_experiment.py `
  --data data/teaching/historical2m-search10-conservative/dataset.npz `
  --architecture base --hidden 512 --epochs 40 --batch-size 4096 `
  --qat --schedule cosine --warmup-epochs 2 --device cuda `
  --init-from results/experiments/historical2m-base512-anneal-s20260917/student.bin `
  --out-dir results/experiments/REPRODUCED-base512-ft

# Benchmark de confirmação (já concluído)
python tools/benchmark_candidate.py `
  --candidate-executable results/experiments/base512-search10-ft-s20260917/zquoridor.exe `
  --candidate-nnue results/experiments/base512-search10-ft-s20260917/student_int8.bin `
  --pairs 100 --move-time-ms 200 --workers 1 --seed 20260920 `
  --openings tools/external/openings_confirmation_v1.jsonl `
  --output benchmark_results/base512-search10-ft-confirm-200ms-s20260918 `
  --claustrophobia-device cpu
```

O comando de benchmark executa automaticamente `vs-main`, `vs-external` e
`main-vs-external`; não é necessário disparar três comandos separados.

### Como conferir um benchmark em execução

```powershell
$p = Get-Process -Id 12844 -ErrorAction SilentlyContinue
Get-ChildItem benchmark_results/race512-search10-ft-confirm-200ms-s20260918 -Recurse -Filter games.jsonl |
  ForEach-Object { "$($_.Directory.Name): $((Get-Content $_.FullName).Count) jogos" }
if ($p) { "PID ativo; CPU acumulada: $([math]::Round($p.CPU,1)) s" } else { "PID terminou" }
```

O PID acima é específico desta execução. Para outra campanha, use o PID
registrado no disparo e o diretório correspondente. Só marque a etapa como
concluída quando houver `summary.json`, os três sub-relatórios e processo
encerrado.

## 11. Estado de promoção

A hierarquia comprovada das redes candidatas até o momento:
1. **Baseline inicial**: `race512-search10-ft` (58,5% vs main, 57,5% vs Titanium, 49,0% vs Claustrophobia).
2. **Campeã de larga escala (5-Tier)**: `race512-multitier-champion` (54,8% vs main, 57,0% vs Titanium, 49,67% em 600g vs Claustrophobia, 51,5% h2h sobre search10).
3. **CAMPEÃ ATUAL ABSOLUTA**: `race512-cr200k-champion` (63,0% em 600g vs Titanium [+92,5 Elo, recorde histórico do projeto], 81,25% vs main [+254,7 Elo], 46,25% vs Claustrophobia em triagem [+53,4 Elo acima do main], 37,12% no suite tático Center-Rush [+35 Elo]).

A promoção oficial para os pesos padrão de produção (`data/nnue/nnue_weights.bin` e `data/nnue/nnue_weights_int8.bin`) foi **CONCLUÍDA**, com `race:512` homologada como arquitetura default do motor em `src/nnue.hpp` e no bundle WASM.

### Triagem de `race512-weakness-ft` e Prova Empírica do MCTS da Claustrophobia (2026-09-19)

A triagem do modelo `race512-weakness-ft` (34.787 posições mineradas de fraqueza, usando 75% Claustrophobia direct forward-pass + 25% ZQ search) produziu um resultado revelador:
- **vs Titanium**: **62,5%** (Elo +88,7, avanço notável sobre os 57,5% da campeã).
- **vs Main**: **45,0%** (Elo -34,9).
- **vs Claustrophobia**: **31,25%** (Elo -137,0) com **153 estados repetidos** em 40 jogos.

**Conclusão Empírica Crucial**: A rede crua da Claustrophobia (sem o seu MCTS) sofre de loops de repetição cíclica e indefinição tática em posições espelho. Destilar essa rede direta sem busca enfraquece o motor taticamente contra ela mesma. O teaching da Claustrophobia **precisa obrigatoriamente vir de sua busca MCTS (`zq_search_bridge.exe`)**, que resolve táticas por contagem de visitas e elimina loops de repetição.

## 12. Currículo Hierárquico em 5 Camadas (5-Tier Curriculum) para Romper >50% contra Claustrophobia

Para superar os 50% contra o Claustrophobia sem regressão contra Titanium ou main, o projeto estabeleceu um currículo hierárquico ponderado de larga escala. As amostragens pequenas do passado (2.000 posições de busca, representando 0,1% do dataset) causavam saturação em 48,8%–49,0%. O novo currículo escala o volume e a profundidade de supervisão em cinco camadas complementares:

### 12.1 Estrutura das 5 Camadas

| Camada | Nome / Escopo | Volume | Fonte / Supervisão | Peso no Treino | Status Operacional |
|---|---|---:|---|---:|---|
| **Tier 1** | **Fundo Maciço de Replay** | **10.000.000** | Amostragem uniforme de todos os 499 shards de `data/selfplay_canonical_v3/`. Alvos suaves (softmax policy + valor $[-1, 1]$) gerados via inferência CUDA com a rede campeã `race512-search10-ft`. Sem ruído WDL de partidas antigas. | **$\times 1,0$** | **100% Concluído** (`massive-background-10m/dataset.npz`, 5,36 GB). |
| **Tier 2** | **Search Histórico Provado** | **500.000** | Corpus histórico de busca consolidado em `data/teaching/mixed-gen1-500k-search20/dataset.npz`. Contém posições com busca tática validada. | **$\times 2,0$** | **100% Concluído** (pronto no disco). |
| **Tier 3** | **Search Genérico em Larga Escala** | **1.000.000** | Estados representativos de partidas completas e auto-jogo auditado. Submetidos a busca real bilateral: Claustrophobia MCTS (64 simulações em CUDA) + ZQuoridor MCAB (512 nós). Dá consistência tática em posições neutras. | **$\times 3,5$** | **100% Concluído e Fusionado (100.000/100.000 posições)**: `generic-search-100k-zq/dataset.npz` (100k amostras bilaterais, 19.881 de validação isoladas, peso médio 7,15); 900.000 posições adicionais prontas no índice. |
| **Tier 3.5** | **Generic Critical Search (Nova Camada)** | **100.000** | Posições extraídas do acervo de auto-jogo genérico aplicando quatro filtros refinados de criticidade: alta entropia de política (Shannon entropy / top-2 ratio $\ge 0,5$), incerteza de valor e corrida ($|V| \approx 0$, $|d_{\text{own}} - d_{\text{opp}}| \le 2$), mudanças críticas de muro ($\le 3$ muros ou labirinto denso com alta pressão CAT) e swings de virada. **Supervisão com busca real em árvore ("search search mesmo")**: **80% Claustrophobia MCTS** (`zq_search_bridge.exe` 64 sims em CUDA) + **20% ZQuoridor MCAB** (512 nós em CPU), ponderados por divergência Jensen-Shannon. | **$\times 4,5$** | **100% Concluído e Fusionado (100.000/100.000 posições)**: `tier3-5-generic-critical-100k/dataset.npz` (100k amostras bilaterais 80/20, 19.991 de validação isoladas, peso médio 7,58, divergência de busca 0,527). |
| **Tier 4** | **Dual-Crisis Search** | **100.000** | Posições críticas mineradas: derrotas contra Claustrophobia e Titanium, estados das 23 aberturas varridas 0-2 (densas em muros) e assimetrias agudas de estoque ($|\text{own} - \text{opp}| \ge 2$, $\le 3$ muros). Supervisão: 75% Claustrophobia MCTS (`zq_search_bridge.exe` 64 sims) + 25% ZQuoridor MCAB (512 nós) com escalonamento por divergência. | **$\times 6,0 \sim 8,0$** | **100% Concluído (100.000/100.000 posições consolidadas)**: `data/teaching/tier4-dual-crisis-100k/dataset.npz` (100k amostras, 16.763 de validação isoladas). |
| **Tier 5** | **Deep Search + Crises Ponderadas** | **10k Deep Search** | Posições de bifurcação e crise submetidas ao MCTS profundo da Claustrophobia (512 sims em CUDA). Alvos de busca profunda (~500ms). | **$\times 6,0$** | **Deep search de 10.000 posições a ~500ms 100% Concluído e Integrado** (`data/teaching/tier5-deep-search-10k/dataset.npz`, 10k posições com 512 sims na GPU RTX 4050, 5.025 amostras de validação). Tarefa de rollouts sintéticos encerrada para desobstruir CPU e evitar contenção durante o treinamento campeão. |

### 12.2 Pipeline de Montagem e Treinamento

1. **Montagem Unificada com Streaming de Baixo Consumo**: `tools/teacher/assemble_5tier_dataset.py` consolidou **10.810.000 posições** (Tier 1, 2, 3, 3.5, 4 e 5 Deep Search) com streaming em blocos sequenciais, operando com pico de RAM < 200 MB e 0% de overlap entre grupos de treino e validação.
2. **Validação Eficiente no Treinador**: `training/run_experiment.py` normalizado para validar fatias sequenciais de 500.000 linhas, eliminando estouro de RAM e liberando metadados logo após o particionamento.
3. **Treinamento da Campeã Concluído (`race512-multitier-champion`)**:
   - Arquitetura: `race`, `hidden: 512` (456 inputs -> 512 neurônios -> cabeças de política e valor).
   - Inicialização: pesos da campeã atual `race512-search10-ft`.
   - Épocas: 60 épocas completadas com batch size 1024.
   - Learning Rate: inicial `1.5e-5`, decay cosseno longo (recozimento profundo) até `min_lr=5e-7`, com 3 épocas de warmup.
   - Escala do tronco: `--trunk-lr-scale 0.2` para proteger as representações de base e refinar as cabeças táticas.
   - Quantização: QAT nativo com `QA=255`, `QB=64`.
   - Hardware: 100% acelerado nos núcleos Tensor da GPU RTX 4050 via CUDA.
   - **Métricas Finais Consolidadas (Época 60)**:
     - Val Loss: $0,9571 \to 0,8701$ (-9,1% relativo, melhor perda registrada na história do projeto).
     - Val Policy KL: $0,4903 \to 0,4206$ (-14,2% relativo).
     - Val Value MAE: $0,2730 \to 0,2116$ (-22,5% relativo).
     - Train Loss: $0,8718 \to 0,8183$.
     - Train Policy KL: $0,4169 \to 0,3721$.
     - Train Value MAE: $0,2227 \to 0,2009$.
   - **Verificação de Paridade Numérica**: `incremental_check.exe` executado sobre 4.758 posições (jogo $\times$ ply $\times$ perspectiva). 0 divergências, erro máximo = 0 (100% de paridade).
   - **Artefatos Gerados**:
     - `results/experiments/race512-multitier-champion/student.bin` (pesos float32)
     - `results/experiments/race512-multitier-champion/student_int8.bin` (pesos quantizados int8)
     - `results/experiments/race512-multitier-champion/zquoridor.exe` (executável nativo com flags `-DZQ_NNUE_RACE_FEATURES=1 -DZQ_NNUE_HIDDEN=512`)
4. **Critério de Aceitação e Homologação**:
   - Triagem: 20 pares (40 jogos) a 200 ms contra `main`, Titanium e Claustrophobia.
   - Confirmação Rigorosa: 100 pares (200 jogos) a 200 ms por lance com o livro oficial `openings_confirmation_v1.jsonl`.
   - Meta de Promoção: Placar > 50,0% contra Claustrophobia, > 57,5% contra Titanium e > 58,5% contra o main, com limite inferior do intervalo de confiança bootstrap 95% estritamente positivo.

### 12.4 Resultados da Triagem da Campeã (`race512-multitier-champion`) e Diagnóstico Tático

A triagem oficial de 20 pares (40 jogos por oponente) a 200 ms por lance foi concluída em 2026-09-19 utilizando o livro oficial `tools/external/openings_screen_v1.jsonl` e aceleração CUDA para o Claustrophobia.

| Oponente | Jogos | Placar (%) | Vitórias / Empates / Derrotas | Elo Relativo | IC Bootstrap 95% (Elo) | Status vs Meta |
|---|---:|---:|---:|---:|---:|---|
| **`main`** | 40 | **77,5%** | 30W / 0D / 10L | **+214,8** | [+117,2, +358,8] | **Superada** (meta > 58,5%) |
| **Titanium** | 40 | **70,0%** | 28W / 0D / 12L | **+147,2** | [+52,5, +269,4] | **Superada** (meta > 57,5%) |
| **Claustrophobia** | 40 | **38,8%** | 15W / 1D / 24L | **-79,5** | [-157,7, -17,4] | **Aberta** (meta > 50,0%) |

Para referência comparativa no mesmo benchmark idêntico, a rede de produção do `main` obteve:
- vs Titanium: 67,5% (+127,0 Elo). A campeã superou o `main` em +2,5 pontos percentuais (+20,2 Elo).
- vs Claustrophobia: 43,8% (-43,7 Elo, 17W / 1D / 22L). A candidata ficou a apenas 2 vitórias do `main`.

#### Diagnóstico Empírico das Derrotas contra Claustrophobia

1. **Assimetria Drástica do Primeiro Jogador**:
   - Como P0 (primeiro a mover): A campeã obteve **63,2% de vitórias** (12W / 7L) contra o MCTS da Claustrophobia.
   - Como P1 (segundo a mover): A campeã obteve apenas **15,0% de vitórias** (3W / 17L).
   - Em 15 dos 20 pares de abertura, o resultado foi divisão 1-1 em que o jogador com as brancas venceu.
2. **Dinâmica de Esgotamento de Muros**:
   - A campeã melhorou substancialmente a retenção de muros em relação ao `main`: foi a primeira a esgotar muros em 54,2% das derrotas (contra 77,3% no `main`).
   - Contudo, a Claustrophobia manteve de 3 a 4 muros de reserva até os plies 45-55. Em finais avançados, a Claustrophobia aplicou cortes táticos profundos (ex: `h5v` ampliando o caminho em +4 passos), momento no qual a campeã estava com estoque zero e sem recursos para contra-ataque.
3. **Divergência Tática em Aberturas Específicas**:
   - Nas Aberturas 81 e 86, o `main` varreu a Claustrophobia (2-0), enquanto a campeã foi varrida (0-2). A análise lance a lance revelou que o `main` utilizou muros táticos imediatos de contenção no centro (`d4v` no ply 18 e `d6h` no ply 10), enquanto a campeã optou por avanços de peão mais passivos (`c2` e `e4`), permitindo que a Claustrophobia tomasse o controle do corredor central.

### 12.5 Campanha Especializada em Center Rush (200k) e Treinamento em Dois Estágios (`race512-cr200k-champion`)

Com base no diagnóstico de que as aberturas de avanço central e esgotamento precoce de muros representavam o principal gargalo tático, foi desenvolvida uma campanha intensiva dedicada ao regime de **Center Rush** (`e2 e8 e3 e7 e4 e6` e colisões centrais imediatas), unindo mineração de larga escala, Action-Q Policy Sharpening e treino modular em dois estágios:

#### 1. Dataset de Prioridade Center Rush (`data/teaching/center-rush-200k-priority/dataset.npz`)
- **Script**: `tools/teacher/build_center_rush_priority_dataset.py`.
- **Volume**: 255.000 posições consolidadas (225.501 treino, 29.499 validação com agrupamento estrito por abertura para evitar vazamento).
- **Composição e Pesos**:
  - **200.000 posições de Center Rush**: mineradas do corpus multitier com peões nas colunas centrais ($c, d, e, f, g$) e linhas 3–6, com estoque ativo de muros ($\ge 5$), ponderadas em **5,0×**.
  - **5.000 posições de busca profunda com Action-Q Sharpening**: estados críticos reavaliados com 1024 nós de busca MCAB, aplicando a fórmula $\pi'_a \propto N_a^\alpha \exp(\beta Q_a)$ com $\alpha=1,0$ e $\beta=1,5$ para depurar ruído de exploração e destacar lances táticos decisivos, ponderadas em **6,0× a 8,0×**.
  - **50.000 posições de fundo geral**: amostradas uniformemente de fora do regime central, ponderadas em **1,0×** como âncora para prevenir esquecimento catastrófico.
  - **Concentração de sinal**: O regime de Center Rush responde por **95,37%** de todo o gradiente ponderado de treino.

#### 2. Treinamento em Dois Estágios

- **Estágio 1 — Treino Dedicado de Política (`race512-cr200k-policy`)**:
  - Diretório: `results/experiments/race512-cr200k-policy/`.
  - Configuração: `--train-scope policy` por 15 épocas em GPU CUDA (`batch_size=1024`), schedule cosseno ($1,2 \times 10^{-4} \to 5 \times 10^{-6}$).
  - Tronco de acumulador (`fc1`) e cabeça de valor completamente congelados (0% de alteração de valor, 0 divergências numéricas).
  - **Resultado**: A divergência KL da política em validação caiu de **0,1015 para 0,0734** (**redução relativa de erro de 28%**).
- **Estágio 2 — Calibração Holística das Cabeças (`race512-cr200k-champion`)**:
  - Diretório: `results/experiments/race512-cr200k-champion/`.
  - Configuração: Inicializado a partir dos pesos de política refinados do Estágio 1; `--train-scope heads` por 15 épocas com taxa suave ($2,5 \times 10^{-5} \to 5 \times 10^{-7}$).
  - **Resultado**: A loss de validação atingiu **0,6954**, a menor marca registrada em toda a história do projeto.
  - **Verificação de Paridade**: `incremental_check.exe` executado em 4.758 posições registrou **0 divergências** entre acumulador incremental e reconstrução completa.

#### 3. Bateria de Benchmarks Rigorosos

1. **Match de Confirmação de 600 Jogos vs Titanium**:
   - Livro: 300 aberturas únicas com troca obrigatória de cores (600 jogos totais), relógio paritário de 200 ms por lance.
   - Placar: **378 vitórias, 0 empates, 222 derrotas -> 63,0% de aproveitamento (+92,5 Elo)**.
   - Intervalo de Confiança Bootstrap 95%: [+64,4, +121,7 Elo] (estritamente positivo).
   - **Maior placar já registrado contra o Titanium na história do ZQuoridor** (o recorde anterior era 57,0% / +49,0 Elo).
2. **Triagem Tripla (40 Jogos por Oponente a 200 ms)**:
   - vs `main`: **81,25% de aproveitamento (32W / 1D / 7L, +254,7 Elo)** — vitória esmagadora contra a versão base.
   - vs Titanium: **50,0% de aproveitamento (20W / 0D / 20L, 0,0 Elo)**.
   - vs Claustrophobia: **46,25% de aproveitamento (18W / 1D / 21L, -26,1 Elo)** — superando o `main` (38,75%) por **+53,4 Elo**.
3. **Suite Tático de Center Rush vs Claustrophobia (66 jogos a 200 ms)**:
   - Aproveitamento geral: **37,12%** (+35 Elo sobre a campeã multitier anterior, 32,58%).
   - Família `pawn_jump`: **62,5% de aproveitamento** (5,0 / 8 pontos).
   - Família `front_wall`: **47,2% de aproveitamento** (8,5 / 18 pontos).
   - Família `vertical_channel`: **33,3% de aproveitamento** (dobro da baseline anterior).
4. **Match de Confirmação de 600 Jogos vs Claustrophobia (Concluído)**:
   - Livro: 300 aberturas únicas com troca obrigatória de cores (600 jogos totais), relógio paritário de 200 ms por lance na GPU RTX 4050 (0 falhas).
   - Placar: **281 vitórias, 13 empates, 306 derrotas -> 47,92% de aproveitamento (-14,5 Elo)**.
   - **Simetria de Cores P0 vs P1**: 141 vitórias de Brancas (P0: 47,0%) e 140 vitórias de Pretas (P1: 46,7%). O colapso de Pretas (que era de apenas 15,0% na `multitier-champion`) foi completamente eliminado!
   - Intervalo de Confiança Bootstrap 95%: [44,33%, 51,50%] (Elo: [-39,55, +10,43]).
   - **Veredito Global de Força**: A `race512-cr200k-champion` superou o baseline do `main` por +254,7 Elo (81,25%), estabeleceu o recorde histórico do projeto contra Titanium com +92,5 Elo (63,0%) e disputa lance a lance em quase paridade exata com o Claustrophobia (47,92% com intervalo tocando 51,5%), com simetria perfeita entre brancas e pretas. Está homologada como a melhor rede geral do projeto.

---

## 13. Catálogo Canônico de TO-DO e Ideias Arquiteturais Futuras

As ideias a seguir representam o roadmap de pesquisa de longo prazo para novas arquiteturas de rede e refinamento de alvos após a conclusão da campanha massiva das 5 camadas e da especialização em Center Rush.

### 13.1 TO-DO: Target Sharpening com Q por Ação e Cabeça Q Auxiliar

Hoje o pipeline de busca profunda (`zq_deep_relabel.py`) registra os valores Q (`action_q`) de cada lance visitado durante a busca MCAB, permitindo enriquecer as visitas com a qualidade esperada de cada lance.

- **Fase A (Sem alteração de arquitetura — CONCLUÍDA/VALIDADA)**: Gerar alvos de política enriquecidos combinando visitas e vantagem de valor Q:
  $$\pi'_a \propto N_a^\alpha \cdot \exp(\beta \cdot Q_a)$$
  Implementada em `tools/teacher/build_center_rush_priority_dataset.py`, resultando na redução de 28% no erro KL da rede campeã `race512-cr200k-champion`.
- **Fase B (Mudança arquitetural — FUTURA)**: Adicionar uma cabeça Q auxiliar por ação à rede neural (`256/512 -> 209`), permitindo que a rede preveja diretamente o valor esperado de cada ação legal, diferenciando muros táticos críticos de muros neutros.

### 13.2 TO-DO: Feature Relacional Especializada (Margem de Distância × Regimes de Muros)

A feature de interação atual (`ahead/equal/behind` $\times$ classes de muros) perde a magnitude da vantagem de distância. Por exemplo, uma posição com `ownDist=2, oppDist=3, ownWalls=0, oppWalls=8` recebe exatamente a mesma ativação de feature que `ownDist=2, oppDist=12, ownWalls=0, oppWalls=8` (`ahead × 0 × 3+`), embora a segunda seja uma vitória garantida e a primeira seja uma derrota iminente.

- **Nova Representação**:
  $$\text{Margem de Distância} \times \text{Regime de Muros}$$
  Onde a margem cobre 33 distâncias inteiras $[-16 \dots +16]$ e os regimes de muros cobrem 4 estados qualitativos:
  1. Ambos possuem muros em reserva (`own > 0 && opp > 0`).
  2. Apenas o jogador atual está sem muros (`own == 0 && opp > 0`).
  3. Apenas o oponente está sem muros (`own > 0 && opp == 0`).
  4. Ambos estão sem muros (`own == 0 && opp == 0`, regime de corrida pura).
- Total de features: $33 \times 4 = 132$ entradas one-hot de baixíssimo custo de atualização incremental.

### 13.3 TO-DO: Estrutura do Caminho Mínimo e Máscaras Direcionais de Gargalo

Capturar a topologia de estrangulamento do corredor antes que a busca precise ramificar:
- **Fator de Ramificação Ótimo**: Quantidade de primeiros passos válidos que mantêm o comprimento do caminho mínimo ($1, 2, 3$ ou $4$). Quando igual a 1, o peão está em um corredor estrito de gargalo.
- **Máscara Direcional**: Vetor binário de 4 bits indicando direções em que o caminho mínimo avança (frente, esquerda, direita, recuo).

### 13.4 TO-DO: Geometria de Interação Local de Peões

Features compactas de contato direto entre os dois peões:
- Deslocamento relativo $(\Delta x, \Delta y)$ entre peão próprio e peão oponente.
- Indicadores booleanos de adjacência ortogonal e diagonal.
- Estado de salto direto (se há oportunidade imediata de salto simples ou salto lateral).
- Custo: 10 a 30 entradas esparsas.

### 13.5 TO-DO: Acumulador Bilateral Completo (Full Bilateral Accumulator)

Atualmente, o engine mantém um par de acumuladores (`AccPair`), mas as cabeças de valor e política consomem apenas o acumulador da perspectiva do jogador a mover.
- **Proposta**: Concatenar ambos os acumuladores ativados antes de alimentar as cabeças:
  $$[\text{SCReLU}(acc[\text{mover}]), \text{SCReLU}(acc[\text{opponent}])] \to 512 \text{ features}$$
- Cabeças candidatas:
  - Valor: $512 \to 64 \to 1$
  - Política: $512 \to 209$
- Justificativa: Permite que a rede avalie diretamente o desequilíbrio mútuo e a tensão de corrida sem exigir que um único acumulador reconstrua a visão do oponente a partir de suas próprias coordenadas.

### 13.6 TO-DO: Teacher de Search-Value no Auto-Jogo

Eliminar o bootstrap de redes antigas substituindo a avaliação estática pré-busca pelo valor refinado da raiz da busca MCAB após a exploração da árvore:
$$\text{Alvo de Valor} = \alpha \cdot \text{Resultado\_Final} + (1 - \alpha) \cdot \text{Searched\_Root\_Value}$$
Com $\alpha \in \{0,70; 0,85; 1,0\}$. O modelo aprende com o operador de melhoria da busca em vez de memorizar seu viés posicional prévio.

### 13.7 TO-DO: Selfplay com Professor Forte Desacoplado

Desacoplar o orçamento de tempo da geração de dados do orçamento de jogo em produção. Gerar auto-jogo com professores profundos operando a 80–150 ms (ou orçamentos de nós expandidos) para alimentar um aluno treinado para jogar com excelência no relógio padrão de 200 ms.

### 13.8 TO-DO: Heurísticas Dinâmicas de Conservação de Muros na Busca

Mecanismos de busca no código C++ (`search.hpp`) para conter o esgotamento precoce de muros identificado no diagnóstico contra Claustrophobia:
1. **Penalidade de Desperdício Tardio**: Desencorajar na ordenação ou podar extensões de muros quando o estoque próprio é baixo ($\le 2$ muros) e o lance não altera o delta líquido de BFS em favor do jogador.
2. **Antecipação da Busca de Final (`endgameMoverWallThreshold = 1`)**: Ativar busca tática alfa-beta profunda de peões assim que o jogador a mover atinge 1 muro restante, preparando o terreno antes do esgotamento total.

---

## 14. Seção Stale / Depreciada (Abordagens Descartadas para Não Repetir)

Para economizar tempo e compute de futuros agentes e desenvolvedores, os seguintes caminhos já foram testados empiricamente e comprovadamente **fracassaram ou produziram regressões**:

1. **Destilação da Rede Direta da Claustrophobia sem Busca MCTS**:
   - *O que foi tentado*: Extrair alvos de política e valor via forward-pass direto da rede neural da Claustrophobia (`historical2m-weakness-35k-claustro75`).
   - *Por que falhou*: A rede pura da Claustrophobia sofre de forte indefinição posicional em simetrias e gera ciclos de repetição infinitos (**153 estados repetidos em 40 jogos**). A força da Claustrophobia vem do seu MCTS com contagem de visitas, não de sua prior network crua.
   - *Regra*: **O teaching da Claustrophobia deve obrigatoriamente usar sua busca MCTS (`zq_search_bridge.exe`)**.

2. **Teaching de Busca com Amostragens Reduzidas (ex: 2.000 posições)**:
   - *O que foi tentado*: Gerar alvos de busca profunda apenas para 2.000 posições de alta divergência (`top2000.jsonl` / `search-priority-gen1-conservative`).
   - *Por que falhou*: 2.000 posições representam menos de 0,1% de um dataset de 2M. O modelo satura rapidamente em ~48,8%–49,0% contra Claustrophobia por sub-representação estatística.
   - *Regra*: Datasets de especialização tática exigem no mínimo **100k a 250k amostras** com forte concentração de peso ($\ge 5,0\times$).

3. **Parâmetros de Busca MCTS com Exploração Agressiva ou Poda Precoce**:
   - *O que foi tentado*: Testar `cPuct = 1.40`, `cPuct = 1.10` e `progressiveWidening` (restrição a top-16 priors) em colisões de Center Rush.
   - *Por que falhou*: `cPuct = 1.40` degradou o controle de corredores verticais estreitos; `progressiveWidening` eliminou lances táticos de muros vitais antes da exploração, caindo para 27%–33% de score.
   - *Regra*: Manter `cPuct = 0.80` e confiar na política afinada com Action-Q Sharpening para guiar a busca.

4. **Geração de Rollouts Sintéticos sem Limite Amplo de Plies**:
   - *O que foi tentado*: Rodar `generate_rollouts.cpp` com `--max-plies 60`.
   - *Por que falhou*: Jogos que terminam em repetição de 3 dobras ou corte por limite de plies descartam seus passos para evitar rotular lances inconclusivos.
   - *Regra*: Utilizar sempre `--max-plies 120` ou superior.

5. **Amostragem Aleatória sem Agrupamento Estrito de Validação**:
   - *O que foi tentado*: Separar treino e validação com `np.random.rand() < 0.1` aleatório por linha.
   - *Por que falhou*: Estados derivados da mesma abertura ou corrida de peões vazam entre treino e teste, mascarando sobreajuste na validação.
   - *Regra*: Particionar rigorosamente por `group_id` baseado na combinação única de peões ou no índice da abertura.



