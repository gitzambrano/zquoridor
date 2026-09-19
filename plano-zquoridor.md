# Plano ZQuoridor — Documento Canônico e Painel Operacional

Atualizado em 2026-09-19. Este é o DOCUMENTO CANÔNICO oficial de planejamento,
registro histórico (passado antigo e recente), status operacional e catálogo
completo de TO-DO do projeto ZQuoridor. Todas as metas, decisões arquiteturais,
taxonomias de fraqueza e especificações de dataset convergem para este arquivo.
Para reproduzir qualquer experimento, consulte as seções correspondentes e os
arquivos de configuração indicados.

### Leitura rápida para um agente novo

1. **Dados:** use o caminho completo do dataset indicado na tabela da rede.
   Não misture `dataset.npz` de pastas diferentes pelo nome.
2. **Treino:** leia `config.json`, `student.architecture.json` e
   `train_report.json`. O `student_int8.bin` é o peso carregado pelo engine.
3. **Teste:** use `200 ms` por lance, o mesmo livro, seed e cores. Consulte
   `summary.json`; loss não é medida de força.
4. **Estado:** `concluído` tem artefato e relatório; `em execução` tem processo
   vivo e arquivos parciais; `TODO` ainda não produziu resultado.

## 1. Big picture

Objetivo: treinar NNUEs simples, robustas contra wandering e boas em corridas;
medir todas no mesmo relógio contra a rede do `main`, Titanium e
Claustrophobia; promover somente uma rede com evidência independente.

```text
histórico + selfplay -> contrato V3 -> replay/teaching -> datasets mistos
-> treino QAT/fine-tuning -> export/paridade -> screening -> confirmação
-> promoção somente com intervalo favorável
```

O estado de produção continua `data/nnue/nnue_weights_int8.bin`. Nenhum
candidato foi promovido.

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

| ID | Features | Larguras | Estado |
|---|---|---|---|
| `base` | 354: peões, muros, buckets BFS, reservas | 128/256/384/512 | implementada |
| `race` | 456: `base` + margens BFS, reservas e interação corrida | 128/256/384/512 | implementada |
| `phase` | fase por ply, paredes e reservas | futura | a implementar |
| `topology-lite` | contagem/orientação de muros | futura | a implementar |
| `race-phase` | corrida × fase × reservas | futura | a implementar |
| `corridor-touch` | slots tocados por caminhos mínimos | futura | a implementar |
| `multi-path` | robustez/número de rotas alternativas | futura | a implementar |
| `wall-threat` | ameaça local perto de caminho/peão | futura | a implementar |

Prioridade: terminar `base`/`race`, depois `phase`, `topology-lite` e
`race-phase`; features de corredor só entram após medir o ganho atual.

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
| `race:512` | 0,73405 | concluído | 40 épocas | confirmação em fase final (jogos da candidata 600/600 concluídos) |

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

Assim, as seis redes direct foram comparáveis entre si: mesmas posições,
mesmos alvos e mesmo schedule. As duas redes `*-ft` não são treinos direct
novos; são continuações separadas a partir dos respectivos checkpoints, com o
dataset misto e learning rate menor.

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

**A candidata campeã definitiva atual é**: `race512-search10-ft` (58,5% vs main, 57,5% vs Titanium, 49,0% vs Claustrophobia).

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
| **Tier 5** | **Branching Rollouts Dinâmicos + Deep Search** | **10.000 jogos + 10k Deep Search** | Partidas completas geradas a partir de estados de crise com perturbação top-4 nos plies 1–4 e jogo determinístico MCAB subsequente. Alvos temporais descontados: $V = \text{sign} \cdot 0,98^{\text{plies\_restantes}}$. **Busca profunda (~500ms)**: submissão de 10.000 posições de crise/bifurcação do Tier 5 ao MCTS profundo da Claustrophobia (512 sims em CUDA). | **$\times 6,0$** | **Deep search de 10.000 posições a ~500ms 100% Concluído** (`data/teaching/tier5-deep-search-10k/dataset.npz`, 10k posições com 512 sims na GPU RTX 4050, 5.025 amostras de validação); Rollouts do Batch 1 nos plies finais (`task-1589`, PID 6764, >222.000s CPU em 6 threads). |

### 12.2 Pipeline de Montagem e Treinamento

1. **Montagem Unificada com Streaming de Baixo Consumo**: `tools/teacher/assemble_5tier_dataset.py` suporta todas as camadas (Tier 1, 2, 3, 3.5, 4, 5 Deep Search e 5 Rollouts). Para viabilizar a união de 10,7+ milhões de posições dentro do limite de 16 GB de RAM física, o script grava diretamente array a array no arquivo ZIP comprimido via blocos sequenciais (*chunked streaming* de 50.000 amostras), evitando alocações redundantes do array de política de 4,5 GB e mantendo o pico de memória abaixo de 200 MB durante todo o processo. Além disso, grupos de validação (`group_id`) e flags `is_val` são particionados de forma estritamente disjunta por camada (`overlap = 0`), satisfazendo as checagens rigorosas de `split_indices`.
2. **Validação Eficiente no Treinador**: `training/run_experiment.py` atualizado para validar a conformidade e normalização da matriz de política em fatias sequenciais de 500.000 linhas, eliminando o estouro de memória (8,95 GB) na leitura de datasets maciços de 10M+.
3. **Receita de Treinamento**:
   - Arquitetura: `race`, `hidden: 512` (456 inputs -> 512 neurônios -> cabeças de política e valor).
   - Inicialização: pesos da campeã atual `race512-search10-ft`.
   - Épocas: 60 épocas com batch size 1024 e paciência de 20 épocas.
   - Learning Rate: inicial `1.5e-5`, decay cosseno longo (recozimento profundo) até `min_lr=5e-7`, com 3 épocas de warmup.
   - Escala do tronco: `--trunk-lr-scale 0.2` para proteger as representações de base e refinar as cabeças táticas.
   - Quantização: QAT nativo com `QA=255`, `QB=64`.
   - Hardware: inferência de treinamento na GPU (CUDA) e até 14 threads para processamento paralelo de dados.
   - Recursos de Hardware e Alocação Balanceada Homologada:
     - **CPU**: 16 núcleos físicos disponíveis -> teto estrito de **14 threads simultâneas**, preservando 2 núcleos para o sistema operacional.
     - **Divisão Concorrente**: 6 threads alocadas para rollouts (`generate_rollouts.exe` PID 6764) + GPU livre e pronta para o fine-tuning final.
     - **GPU (VRAM)**: NVIDIA GeForce RTX 4050 com **6 GB de VRAM**.
       - Consumo medido no Deep Search: **1.221 MiB de VRAM**, acelerado pelos núcleos Tensor.
       - Treino supervisionado QAT subsequente (batch size 1024): consumo projetado de **~1,2 GB de VRAM**, operando com ampla folga de memória.
3. **Critério de Aceitação e Homologação**:
   - Triagem: 20 pares (40 jogos) a 200 ms contra `main`, Titanium e Claustrophobia.
   - Confirmação Rigorosa: 100 pares (200 jogos) a 200 ms por lance com o livro oficial `openings_confirmation_v1.jsonl`.
   - Meta de Promoção: Placar > 50,0% contra Claustrophobia, > 57,5% contra Titanium e > 58,5% contra o main, com limite inferior do intervalo de confiança bootstrap 95% estritamente positivo.

---

## 13. Catálogo Canônico de TO-DO e Ideias Arquiteturais Futuras

As ideias a seguir representam o roadmap de pesquisa de longo prazo para novas arquiteturas de rede e refinamento de alvos após a conclusão da campanha massiva das 5 camadas.

### 13.1 TO-DO: Target Sharpening com Q por Ação e Cabeça Q Auxiliar

Hoje o pipeline de busca profunda (`zq_deep_relabel.py`) registra os valores Q (`action_q`) de cada lance visitado durante a busca MCAB, mas o construtor do dataset (`build_search_priority_dataset.py`) descarta essa informação e mantém apenas as visitas de política ($N_a$). Isso desperdiça o conhecimento de quão boa ou má cada jogada alternativa realmente é.

- **Fase A (Sem alteração de arquitetura)**: Gerar alvos de política enriquecidos combinando visitas e vantagem de valor Q:
  $$\pi'_a \propto N_a^\alpha \cdot \exp(\beta \cdot Q_a)$$
  Isso acentua lances taticamente sólidos que receberam visitas mas tinham valor muito superior a lances armadilha.
- **Fase B (Mudança arquitetural)**: Adicionar uma cabeça Q auxiliar por ação à rede neural (`256/512 -> 209`), permitindo que a rede preveja diretamente o valor esperado de cada ação legal, diferenciando muros táticos críticos de muros neutros.

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


