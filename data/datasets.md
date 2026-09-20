# ZQuoridor — Catálogo Canônico de Datasets e Metodologia de Supervisão

Atualizado em 2026-09-20. Este documento descreve detalhadamente cada dataset gerado no projeto ZQuoridor, incluindo os filtros de extração, métodos de relabeling (supervisão), esquemas de ponderação e estado de ciclo de vida (Ativo, Base, Especializado ou Depreciado/Stale).

---

## 1. Contrato e Formato Canônico dos Dados

Todos os datasets consolidados no projeto residem em formato `.npz` compactado (`float32`/`int32`/`uint8`), estruturados com as seguintes chaves padrão:

| Chave | Tipo / Shape | Descrição |
|---|---|---|
| `features` | `uint8` `[N, 456]` (ou 354) | Vetor de características esparsas ativadas na perspectiva do jogador a mover (`own_pawn`, `opp_pawn`, `walls_h`, `walls_v`, distâncias BFS em buckets, reservas de muros e interações de corrida). |
| `policy` | `float32` `[N, 209]` | Distribuição de probabilidade sobre os 209 lances legais possíveis (soma = 1,0). Derivada de contagem de visitas ($N_a$) ou sharpened com valor $Q$. |
| `value` | `float32` `[N]` | Avaliação posicional escalar contínua no intervalo $[-1,0; +1,0]$ na perspectiva do jogador da vez. |
| `weight` | `float32` `[N]` | Peso de importância da amostra no cálculo do gradiente da função de perda ponderada. |
| `is_val` | `bool` `[N]` | Máscara booleana separando treino e validação (particionamento sempre estrito por abertura/grupo para vazamento zero). |
| `group_id` | `int32` `[N]` | Identificador da linhagem da abertura ou combinação de peões geradora. Garante independência estatística total entre treino e validação. |

---

## 2. Catálogo de Datasets Ativos e de Alta Performance

### 2.0 Dataset Mestre Unificado Multi-Caminho (11,06M) — `data/teaching/multipath-master-11m/`
- **Arquivo**: `data/teaching/multipath-master-11m/dataset.npz` (5,65 GB).
- **Volume**: 11.065.000 amostras consolidadas.
- **Script Gerador**: `tools/teacher/assemble_multipath_master_dataset.py`.
- **Filtros e Composição**:
  1. **Multitier Completo (10.810.000 amostras, escala $\times 1,0$)**: Reúne integralmente os Tiers 1 (10M replay suave), 2 (500k search histórico), 3 (100k search bilateral), 3.5 (100k search crítico genérico), 4 (100k dual crisis) e 5 (10k deep search MCTS a 512 sims em CUDA).
  2. **Prioridade Center Rush (255.000 amostras, escala $\times 2,0$)**: Adiciona reforço intensivo de colisões centrais com Action-Q Sharpening ($\pi'_a \propto N_a^\alpha \exp(\beta Q_a)$).
- **Arquitetura Alvo**: `multipath:512` (480 entradas: 456 base+corrida + 24 features de grau de saída, estrangulamento de corredores e geometria de salto de peões).

### 2.0.1 Sementes Mineradas de Derrotas e Center Pawn (10k) — `data/teaching/loss-center-seeds/`
- **Arquivo**: `data/teaching/loss-center-seeds/positions.jsonl` (10.030 estados críticos).
- **Script Gerador**: `tools/teacher/mine_losses_and_center.py`.
- **Filtros**:
  1. **5.952 Estados de Derrotas contra Claustrophobia**: Extraídos lance a lance das 306 partidas perdidas no match de 600 jogos, focando em plies de abertura/meio-jogo onde o ZQuoridor cedeu o centro ou esgotou muros.
  2. **4.295 Estados de Derrotas contra Titanium**: Extraídos das 222 derrotas do match de 600 jogos.
  3. **198 Estados do Catálogo Center Rush**: Todas as ramificações das famílias táticas de colisão frontal.
- **Uso**: Alimentação direta do gerador multithread de rollouts sintéticos (`bin/generate_rollouts.exe`).

### 2.1 Dataset Prioritário de Center Rush (255k) — `data/teaching/center-rush-200k-priority/`
- **Arquivo**: `data/teaching/center-rush-200k-priority/dataset.npz` (234,3 MB).
- **Volume**: 255.000 amostras (225.501 treino, 29.499 validação).
- **Script Gerador**: `tools/teacher/build_center_rush_priority_dataset.py`.
- **Filtros e Composição**:
  1. **200.000 Posições de Center Rush ($\times 5,0$ de peso)**:
     - Peão próprio e oponente simultaneamente nas colunas centrais ($c, d, e, f, g$) e linhas 3 a 6.
     - Estoque ativo de muros ($\ge 5$ muros por jogador) para selecionar o momento exato de colisão tática e colocação de barreiras.
  2. **5.000 Posições de Busca Profunda com Action-Q Sharpening ($\times 6,0 \sim 8,0$ de peso)**:
     - Estados de crise reavaliados com 1024 nós de busca MCAB.
     - **Fórmula de Sharpening**:
       $$\pi'_a \propto N_a^\alpha \cdot \exp(\beta \cdot Q_a) \quad (\alpha=1,0; \beta=1,5)$$
       Limpa o ruído de exploração da árvore MCTS/MCAB e canaliza o alvo de política para o lance verdadeiramente vencedor.
  3. **50.000 Posições de Fundo Geral ($\times 1,0$ de peso)**:
     - Amostradas de fora do regime central como âncora de regularização para prevenir esquecimento catastrófico de finais e corredores laterais.
- **Concentração de Sinal**: O regime de Center Rush representa **95,37%** de todo o gradiente efetivo ponderado.
- **Redes Produzidas**: `race512-cr200k-policy` (redução de 28% no erro de política) e `race512-cr200k-champion` (63,0% em 600g vs Titanium, +92,5 Elo; 81,25% vs `main`).

---

### 2.2 Dataset Multitier Hierárquico em 5 Camadas (10,8M) — `data/teaching/multitier-final-dataset/`
- **Arquivo**: `data/teaching/multitier-final-dataset/dataset.npz` (5,91 GB).
- **Volume**: 10.810.000 amostras (10.000.000 treino, 810.000 validação).
- **Script Gerador**: `tools/teacher/assemble_5tier_dataset.py`.
- **Estrutura por Camadas (Tiers)**:
  - **Tier 1 (Fundo Maciço de Replay — 10M, peso $\times 1,0$)**: Amostragem uniforme dos 499 shards canônicos V3. Inferência suave CUDA com a campeã `race512-search10-ft`.
  - **Tier 2 (Search Histórico Provado — 500k, peso $\times 2,0$)**: Corpus consolidado de busca de auto-jogo prévio validado.
  - **Tier 3 (Search Bilateral em Larga Escala — 100k, peso $\times 3,5$)**: 64 simulações MCTS Claustrophobia (CUDA) + 512 nós MCAB ZQuoridor.
  - **Tier 3.5 (Search Crítico Genérico — 100k, peso $\times 4,5$)**: Posições com alta entropia de política (top-2 ratio $\ge 0,5$), incerteza de corrida ($|V| \approx 0$, $|\Delta d| \le 2$) e alta pressão CAT.
  - **Tier 4 (Dual-Crisis Search — 100k, peso $\times 6,0 \sim 8,0$)**: Estados de derrotas contra Claustrophobia/Titanium, assimetrias agudas de estoque ($|\Delta w| \ge 2$, $\le 3$ muros) e aberturas densas em muros.
  - **Tier 5 (Deep Search MCTS — 10k, peso $\times 6,0$)**: Posições submetidas a 512 simulações MCTS profundas na GPU RTX 4050 (~500ms por posição).
- **Redes Produzidas**: `race512-multitier-champion` (51,5% h2h sobre baseline, 49,67% em 600g vs Claustrophobia).

---

### 2.3 Dataset Tático Especializado de Política (245k) — `data/teaching/policy-tactical-245k/`
- **Arquivo**: `data/teaching/policy-tactical-245k/dataset.npz` (228 MB).
- **Volume**: 244.787 posições.
- **Foco**: Treinamento dedicado do cabeçote de política (`--train-scope policy`).
- **Supervisão**: Alvos de política refinados com busca profunda MCTS.
- **Redes Produzidas**: `race512-policy-tactical` (+40,5 Elo em center rush sobre baseline anterior).

---

### 2.4 Datasets Históricos Base de Replay e Fine-Tune
- **`data/teaching/replay-historical-2m-cuda/dataset.npz` (2.000.000 posições)**:
  - Replay direto puro da rede antiga e Claustrophobia em forward.
  - Serviu como base limpa para o treinamento inicial das 6 redes da matriz de arquitetura (`base`/`race` de 128 a 512).
- **`data/teaching/historical2m-search10-conservative/dataset.npz` (2.001.868 posições)**:
  - 90% replay direct + 10% search (dentro do bloco search: 25% ZQ MCAB e 75% Claustrophobia MCTS).
  - Gerou as finalistas originais `base512-search10-ft` e `race512-search10-ft`.

---

## 3. Datasets Depreciados / Stale (Não Repetir)

Os datasets abaixo cumpriram papel experimental em ciclos anteriores, mas foram superados ou revelaram falhas metodológicas documentadas. **Não devem ser reutilizados em novos treinamentos**:

| Dataset | Motivo da Depreciação / Falha Metodológica | Lição Aprendida |
|---|---|---|
| `data/teaching/historical2m-weakness-35k-claustro75/` | Utilizou 75% de alvos da Claustrophobia gerados por **inferência direta (forward pass cru)** sem árvore MCTS. A rede gerada (`race512-weakness-ft`) sofreu colapso tático com **153 estados repetidos** em 40 jogos contra a própria Claustrophobia. | A rede neural crua da Claustrophobia tem loops de repetição em posições espelho. **O teaching da Claustrophobia deve obrigatoriamente vir de sua busca MCTS (`zq_search_bridge.exe`)**. |
| `data/teaching/search-priority-gen1-conservative/` | Continha apenas **2.000 posições** de busca (0,1% do dataset total). | Amostragens pequenas atingem platô precoce em ~48,8%–49,0% contra Claustrophobia. Escala massiva (100k–200k) é indispensável. |
| `data/teaching/replay-old-gen1-500k/` | Shards de auto-jogo pré-canônico sem o contrato unificado V3 mover-mirrored. | Substituído integralmente pelo corpus V3 auditado. |
| `data/teaching/pilot/`, `canonical-v3-smoke/`, `rollouts-500-seed/` | Testes sintéticos preliminares de pipeline e fumaça. | Apenas para validação de scripts de teste. |

---

## 4. Pipeline de Filtragem e Métodos de Relabeling

### Filtro A: Detecção de Posições Críticas de Center Rush
```python
# Critério implementado em tools/teacher/build_center_rush_priority_dataset.py
is_center_file = (c_own in [2, 3, 4, 5, 6]) and (c_opp in [2, 3, 4, 5, 6])
is_center_rank = (r_own in [2, 3, 4, 5]) and (r_opp in [3, 4, 5, 6])
has_active_walls = (w_own >= 5) and (w_opp >= 5)
is_center_rush = is_center_file and is_center_rank and has_active_walls
```

### Filtro B: Action-Q Policy Sharpening
```python
# Transforma visitas brutas N_a e valores de ação Q_a em alvo refinado
sharp_logits = alpha * np.log(np.maximum(visits, 1e-4)) + beta * action_q
sharp_policy = softmax(sharp_logits)
```
- Efeito prático: se dois lances receberam número parecido de visitas na busca, mas um deles tinha valor $Q=+0,80$ e o outro $Q=-0,20$, a fórmula transfere a quase totalidade da probabilidade para o lance vencedor.

### Filtro C: Particionamento Estrito de Treino / Validação
Para evitar qualquer forma de vazamento de dados (*data leakage*):
- O agrupamento de validação utiliza `group_id = (own_pawn * 81) + opp_pawn` em amostragens gerais ou o identificador literal da abertura (`opening_idx`).
- Todas as posições derivadas da mesma abertura ou par de coordenadas de peões ficam **100% no treino ou 100% na validação**, nunca divididas entre os dois.
