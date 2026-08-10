# GA Tuning — zQuoridor

O tuner antigo chamado `tune_spsa.cpp` foi transformado em um Genetic Algorithm
(GA). O nome do executável/script foi mantido para preservar compatibilidade.

## Por que GA

A função objetivo é o resultado de partidas. Ela é discreta, ruidosa e pode
ficar praticamente constante enquanto um parâmetro muda dentro de uma região.
Isso torna uma estimativa de gradiente SPSA pouco informativa.

O GA mantém uma população de soluções e usa:

- elitismo;
- crossover BLX-alpha;
- mutação gaussiana;
- imigrantes aleatórios;
- confrontos antitéticos (cada candidato joga com as duas cores);
- múltiplos adversários por indivíduo;
- checkpoint do melhor global;
- histórico de fitness e diversidade.

## Parâmetros

| Parâmetro | Faixa | Motivo |
|---|---:|---|
| `contempt` | -150..0 | preferência por posições que evitam empate |
| `policyOrderScale` | 0..2000 | peso da policy head na ordenação |
| `catScoreScale` | 0..20 | peso do CAT na ordenação de muros |
| `lmrMinDepth` | 1..8 | profundidade mínima para LMR |
| `lmrMinMoveIndex` | 1..8 | quantos primeiros lances escapam de LMR |
| `lmrDivisor` | 1.0..4.5 | intensidade da redução LMR |
| `catHotCm` | 50..220 | limiar para não reduzir muro quente |
| `catColdCm` | 0..100 | limiar para redução extra de muro frio |
| `wallBfsOrderMaxPly` | 0..5 | profundidade em que ordenação BFS cara é usada |
| `qsCriticalBfsDelta` | 1..5 | sensibilidade da quiescência a mudanças de rota |
| `policyOrderingMinDepth` | 0..6 | profundidade mínima para executar policy ordering |
| `policyOrderingEnabled` | 0/1 | permite ao GA descobrir se policy ordering ajuda |
| `quiescenceEnabled` | 0/1 | permite A/B automático da quiescência |
| `lmrPvsEnabled` | 0/1 | permite A/B automático de LMR+PVS |

Os três últimos são parâmetros categóricos. O GA os trata como genes binários,
não como variáveis contínuas.

## Paralelismo

`--threads` (default 14) controla quantos indivíduos da população são
avaliados **simultaneamente**, cada um em sua própria thread com engines
independentes (os parâmetros de busca são membros de instância, então isso é
seguro). Não há paralelismo aninhado dentro de uma avaliação -- as partidas
de `--games-per-match` de um mesmo confronto rodam sequenciais na thread do
indivíduo, evitando oversubscription (threads × jogos simultâneos). Ajuste
`--threads` para o número de núcleos físicos disponíveis; população menor
que `--threads` simplesmente deixa algumas threads ociosas naquela geração.

A seleção de "elite" e dos adversários de cada indivíduo usa uma referência
**congelada**: a população já avaliada e ordenada da geração anterior. Isso é
o que torna a paralelização acima segura e determinística por seed -- nenhum
indivíduo depende de quais outros já foram avaliados nesta mesma geração.

## Estratégia recomendada

Para uma primeira busca séria:

```text
population = 24–32
generations = 30–50
games-per-match = 4–8
threads = número de núcleos físicos disponíveis (default 14)
depth = igual ou ligeiramente abaixo do uso normal
time-ms = suficiente para representar o regime real do motor
```

Depois de encontrar um campeão:


1. fixe os parâmetros encontrados;
2. faça um torneio maior contra a versão atual;
3. use várias seeds;
4. aumente profundidade/tempo para validação;
5. só então aceite os parâmetros no motor.

O fitness do GA não deve ser interpretado como Elo. Ele é uma medida interna
de score contra os adversários amostrados.

## Execução

```bash
python teste/run_spsa.py
```

O launcher recompila automaticamente quando `tune_spsa.cpp` ou `search.hpp`
mudam.

Arquivos gerados:

- `ga_checkpoint.txt`
- `ga_history.csv`
- `ga_result.txt`

Para visualizar:

```bash
python teste/plot_spsa.py
```

O script de plot aceita o novo formato GA e também reconhece o formato antigo
do SPSA.

## Importante

O GA é deliberadamente exploratório. Ele não garante o ótimo global, mas é
muito menos dependente de gradiente local do que SPSA. A diversidade da
população é registrada no CSV; uma queda próxima de zero durante muitas
gerações indica convergência prematura e é um sinal para aumentar mutação,
imigrantes ou população.
