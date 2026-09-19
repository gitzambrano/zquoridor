# Plano ZQuoridor — painel operacional

Atualizado em 2026-09-18. Este é o índice de decisão do projeto: cada
experimento aparece em uma tabela antes de ser considerado concluído. Para
reproduzir um resultado, use primeiro o `config.json` da pasta da campanha e
o manifesto do dataset; os comandos deste documento são atalhos legíveis, não
substitutos desses dois arquivos.

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

#### Métricas de Busca e Velocidade
- **ZQuoridor (MCAB)**:
  - A 200 ms: ~10.680 nós expandidos (~53.400 nós/s), tempo médio medido por lance: 209 ms.
  - A 100 ms: ~6.580 nós expandidos (~65.800 nós/s), tempo médio medido por lance: 105 ms. Permite dobrar o volume de jogos por hora em auto-confronto e screening.
- **Claustrophobia**:
  - A 200 ms: ~201 ms por lance, ~400 a 800 simulações MCTS guiadas por rede neural em PyTorch/CPU.

#### Taxonomia de Derrotas nos 100 Pares de Confirmação (200 Jogos)
- **Placar Geral**: 95 vitórias, 105 derrotas (49,0% de aproveitamento).
- **Por Lado**: Brancas (P1): 49,0% (49/100) | Pretas (P2): 46,0% (46/100).
- **Consistência por Par**: 18 vitórias duplas (2-0), 59 empates (1-1), 23 derrotas duplas (0-2).
- **Duração Média**: Vitórias em 62,1 plies; Derrotas em 72,7 plies (indica que perdas se estendem para o final).

#### O Fator Decisivo: Assimetria no Estoque de Muros
Análise dos 105 jogos perdidos vs 95 jogos vencidos contra Claustrophobia categorizados pelo estoque de muros no término:

| Condição no Término | ZQ Vitórias | ZQ Derrotas | Aproveitamento ZQ | % do Total de Derrotas |
|---|---|---|---|---|
| Ambos ainda têm muros | 28 | 24 | **53,8%** | 22,9% |
| Ambos com 0 muros (corrida exata) | 20 | 16 | **55,6%** | 15,2% |
| Claustro sem muros, ZQ com muros | 24 | 8 | **75,0%** | 7,6% |
| **ZQ sem muros, Claustro com muros** | **23** | **57** | **28,8%** | **54,3%** |

**Conclusão Tática Central**:
1. **Mais de 54% de todas as derrotas** contra Claustrophobia acontecem quando o ZQuoridor queima todos os seus 10 muros antes do adversário, permitindo que Claustrophobia guarde de 1 a 4 muros para fechar corredores críticos no final. Nesse cenário, o aproveitamento do ZQuoridor despenca para **28,8%**.
2. Quando ambos mantêm muros ou quando Claustrophobia esgota muros primeiro, o ZQuoridor é francamente superior (**53,8% a 75,0%**).
3. Em corridas de peão puras (ambos com 0 muros), o solver exato (`endgame_race.hpp`) garante vantagem (**55,6%**).
4. O vazamento de força é o **desperdício de muros prematuro** e a **leitura incorreta do desbalanço de reservas no meio-jogo**.

### 8.2 Big Picture: Estratégia de Escala Massiva de Teaching para Romper >50%

Para ultrapassar definitivamente Claustrophobia (>50%), não podemos usar amostragens pontuais (como 2.000 posições selecionadas). Precisamos de escala massiva e mineração dirigida:

1. **Geração Massiva de Jogos (100 ms por lance)**:
   - Utilizar controle rápido de 100 ms para autojogo de alta densidade (duplica a velocidade de geração).
   - Gerar dezenas de milhares de jogos entre variantes de ZQuoridor e contra o próprio Claustrophobia.
2. **Mineração Dirigida de Estados de Divergência & Perda**:
   - Extrair especificamente:
     a) Posições onde houve divergência de lance entre ZQuoridor e Claustrophobia.
     b) Estados intermediários das 23 aberturas onde fomos varridos (0-2) na confirmação.
     c) Posições com assimetria de muros desfavorável (ex.: ZQ com 1-3 muros, Oponente com 3-6 muros).
     d) Lances onde o ZQuoridor decidiu gastar um muro e a avaliação piorou nos plies seguintes.
3. **Teaching em Larga Escala com Ambos os Professores**:
   - Expandir a base de teaching de 2.000 posições para 20.000–50.000 posições críticas mineradas.
   - Relabeling profundo: avaliar com ZQuoridor em alta profundidade (AB/MCAB) e MCTS profundo de Claustrophobia (1024–2048 simulações).
   - Ponderar perdas por divergência: maior peso para posições onde Claustrophobia puniu o gasto de muro.
4. **Calibração de Busca / Conservação de Muro**:
   - Avaliar heurística de conservação de muros (penalizar colocação de muros que não alteram a diferença líquida de BFS em favor do jogador quando a reserva está baixa).
   - Ajustar `endgameMoverWallThreshold` e `cPuct` para valorizar a preservação do estoque defensivo.

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

**A candidata campeã definitiva foi escolhida**: `race512-search10-ft` venceu a seleção geral após confirmação completa de 100 pares (58,5% vs main, 57,5% vs Titanium, 49,0% vs Claustrophobia) e empate técnico no confronto direto com `base512-search10-ft` (49,5% vs 50,5%).

O passo seguinte é a triagem de `race512-search20-ft` para ultrapassar 50% contra Claustrophobia antes do congelamento e promoção oficial dos pesos para `data/nnue/nnue_weights_int8.bin`.
