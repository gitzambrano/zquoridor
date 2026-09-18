# Plano ZQuoridor — painel operacional

Atualizado em 2026-09-18. Este é o índice de decisão do projeto: cada
experimento aparece em uma tabela antes de ser considerado concluído.

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
dataset misto de search. Em ambos: QAT, 4 épocas de warmup, cosine annealing
até `min_lr=5e-6`, export float/int8 e verificação incremental antes da arena.

| Rede | Melhor val loss | Treino | Fine-tune | Status/TODO |
|---|---:|---|---|---|
| `base:128` | — | não priorizada | não | opcional |
| `base:256` | 0,74569 | concluído | não | arquivada após screening |
| `race:256` | 0,74319 | concluído | não | benchmark feito |
| `base:384` | 0,73939 | concluído | não | benchmark feito |
| `race:384` | 0,73686 | concluído | legado 50/50 | benchmark feito |
| `base:512` | 0,73600 | concluído | 40 épocas | confirmação independente |
| `race:512` | 0,73405 | concluído | 40 épocas | comparação já triada |

Fine-tune atual usa `data/teaching/historical2m-search10-conservative/dataset.npz`:
2.001.868 posições, 90% replay direto e 10% search (25% ZQuoridor, 75%
Claustrophobia). `race:512` caiu de 0,91628 para 0,88574; `base:512`, de
0,92031 para 0,89128. Essas losses não são comparáveis ao treino direto.

### Registro de campanhas de treino

| Campanha | Script | Entrada | Configuração | Saída |
|---|---|---|---|---|
| Direct architecture matrix | `training/run_experiment.py` + `training/run_architecture_matrix.py` | dataset direct de 2M | `base/race`, hidden 256/384/512, 80 épocas, QAT, CUDA | um modelo por arquitetura |
| Fine-tune `race512-search10-ft` | `training/run_experiment.py` | checkpoint `race:512` + dataset misto | 40 épocas, LR reduzido, warmup 2, cosine, QAT, CUDA | diretório `race512-search10-ft-s20260917` |
| Fine-tune `base512-search10-ft` | `training/run_experiment.py` | checkpoint `base:512` + dataset misto | 40 épocas, LR reduzido, warmup 2, cosine, QAT, CUDA | diretório `base512-search10-ft-s20260917` |

O dataset misto contém direct e search juntos com pesos por amostra, mas cada
arquitetura direct foi treinada antes em sua própria campanha. Depois, apenas
`base:512` e `race:512` receberam a segunda campanha de fine-tuning.

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

## 7. Benchmark em execução

Diretório: `benchmark_results/base512-search10-ft-confirm-200ms-s20260918`.

| Etapa | Configuração | Último estado salvo | Status |
|---|---|---:|---|
| candidato × main | 100 pares, livro de 400, 200 ms | 37/200 jogos | ativo |
| candidato × Titanium | mesmo livro/seed/relógio | não iniciado | pendente |
| candidato × Claustrophobia | mesmo livro/seed/relógio, CPU | não iniciado | pendente |
| main × externos | referência no mesmo run | pendente | pendente |

O relatório final deve conter jogos válidos, pares completos, score, Elo,
bootstrap pareado e intervalo de decisão. Nenhum parcial decide promoção.
Esta seção é TODO em andamento, não resultado concluído.

## 8. Big picture e roadmap restante

### Gate A — confirmação da finalista atual

1. Concluir `base512-search10-ft` × main: 100 pares, 200 ms, livro de 400.
2. Rodar `base512-search10-ft` × Titanium: 100 pares, mesmo livro/seed/relógio.
3. Rodar `base512-search10-ft` × Claustrophobia: 100 pares, mesmo protocolo.
4. Gerar main × Titanium e main × Claustrophobia no mesmo run como referência.

### Gate B — segunda finalista

5. Repetir os quatro confrontos para `race512-search10-ft` se a primeira não
   demonstrar ganho ou se os intervalos se sobrepuserem.
6. Comparar as duas finalistas contra a rede do main sem misturar livros,
   seeds ou relógios.

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

## 11. Estado de promoção

**Nenhuma rede está promovida.** `base512-search10-ft` é a melhor candidata
provisória, mas aguarda confirmação ampla no mesmo relógio contra main,
Titanium e Claustrophobia.
