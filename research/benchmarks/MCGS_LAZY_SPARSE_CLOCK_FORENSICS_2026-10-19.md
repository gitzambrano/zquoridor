# ZQuoridor — auditoria forense de relógio 3+2 (2026-10-19)

**Escopo:** Lazy Expansion, Sparse Policy e Adaptive Time. Auditoria somente dos artifacts existentes do [workflow 39106925875](https://github.com/gitzambrano/zquoridor/actions/runs/39106925875). **Nenhuma partida ou busca nova foi executada nesta auditoria.** A PR #58 segue bloqueada, sem promoção de 4.02.

## Material auditado

Foram lidos `games.jsonl`, `manifest.json`, `summary.json` e a saída UCI dos **8 artifacts concluídos**: 3 variantes × (Claustrophobia normal + Titanium normal), mais main/combo contra Claustrophobia central. Total **320 partidas**, **8.682 lances do ZQuoridor** e **zero** `termination=time` entre os 320 jogos. **530 lances** terminaram pelo atalho do engine com `nodecap=0`, `nodes=0`, `time=0`; não devem entrar em inferências sobre busca MCGS. Restam **8.152 buscas MCGS ativas**. Nos manifests, todas as células normais têm NNUE SHA256 `45fa3ea7db31...`, livro `1dabcf15aa13...`, árbitro `3e46a8c4732f...`, seed `20261117`, relógio nativo `180000+2000`; o livro central é `c970c5199e08...`.

## Tempos, trabalho e reuso

Médias por **lance MCGS ativo** (sem os 530 atalhos; não misturar `nodes` de Lazy e main, pois semântica de expansão muda):

| Oponente / livro | Variante | Buscas MCGS | Tempo médio | Reuso da árvore | Pico nodes / nodecap |
|---|---|---:|---:|---:|---:|
| Claustrophobia normal | main | 1.027 | 7,390 s | 96,01% | 54,1% |
| Claustrophobia normal | Lazy | 1.063 | 7,238 s | 96,24% | 39,6% |
| Claustrophobia normal | Lazy+Sparse | 1.072 | 7,223 s | 96,27% | 57,8% |
| Titanium normal | main | 1.122 | 7,031 s | 96,43% | 45,6% |
| Titanium normal | Lazy | 1.146 | 7,031 s | 96,42% | 58,7% |
| Titanium normal | Lazy+Sparse | 1.090 | 7,120 s | 96,33% | 31,4% |
| Claustrophobia central | main | 808 | 7,981 s | 95,05% | 48,7% |
| Claustrophobia central | Lazy+Sparse | 824 | 7,950 s | 95,15% | 35,6% |

**Deduções:**
- As variantes **não diferem sistematicamente em segundos efetivos por lance**; não há perda por relógio nem falha global de reuso.
- Em nenhuma busca ativa o relatório UCI de `nodes` chegou perto do `nodecap` (máximo por célula abaixo de 59%). **O corte por número de nós não explica a regressão nessa amostra.**
- Apenas 20/8.682 lances tiveram duração pelo menos 90% do teto rígido UCI (`budget=`). As demais durações, frequentemente 35–45% do teto, são **compatíveis com parada pelo Adaptive Time**, mas o `stop_reason` exato **não** estava instrumentado nos arquivos antigos. Não alegar contagens de `adaptive_stop` sem novos dados.

## Divergência observada da decisão adaptativa

No código 4.01/4.02, `adaptiveNodeScaleMode = VolatileOnly`; `nodecap` de **1.280.000** (contra inicial 640.000 nos primeiros lances) comprova que uma raiz foi classificada **Volatile em algum momento da busca**. Isso é um *proxy* observável, **não** um registro completo da classe final.

Compararam-se somente posições cujos prefixos de lances ainda eram idênticos, com diferença de relógio anterior inferior a 2,5 s. Divergências robustas de escalonamento foram contadas quando uma variante tinha `nodecap>800000` e a outra não:

| Pares de motores (mesmo livro / posição) | Buscas compatíveis | Escalonamento divergente |
|---|---:|---:|
| Claustrophobia normal: main vs combo | 224 | 18 (8,0%) |
| Claustrophobia normal: Lazy vs combo | 205 | 19 (9,3%) |
| Titanium normal: main vs combo | 182 | 16 (8,8%) |
| Titanium normal: Lazy vs combo | 128 | 7 (5,5%) |
| Claustrophobia central: main vs combo | 288 | 19 (6,6%) |

Exemplos diretamente presentes nos JSONL de 3+2:
- **Titanium normal, abertura 10, ZQ lado 1, ply 9**: mesmo prefixo e relógio 173.883 vs 173.882 ms. Main: `e7`, 12.625 ms, `nodecap=1280000`, vitória. Combo: `d2v`, 8.263 ms, `nodecap=640000`, derrota.
- **Claustrophobia normal, abertura 2, ZQ lado 0, ply 8**: relógio 173.883 ms em ambos. Main: `d2`, 10.461 ms, `nodecap=1280000`, vitória. Combo: `e3`, 8.547 ms, `nodecap=640000`, derrota.
- Há **exemplos em sentido contrário**, inclusive combo alocando mais tempo e ainda perdendo. Não selecionar apenas casos favoráveis.

Essa correlação oferece **hipótese causal concreta**: a combinação de árvore/reuso e classificação `Volatile` altera a quantidade de trabalho **em alguns lances críticos**, apesar de médias de tempo semelhantes. **Não demonstra um bug**: duas buscas com árvores e trajetórias de tempo diferentes podem classificar legitimamente a raiz de maneira diferente. Os 40 testes a **20.000 simulações fixas** continuam sendo evidência de paridade Lazy vs combo naquela escala, não de igualdade sob relógio e reuso real.

## Estatística das pequenas células

Bootstrap pareado por abertura (20 pares/célula, 15.000 reamostragens, intervalos percentis aproximados 95% para diferença em pontos percentuais):
- Claustrophobia normal: combo–main **−5,0 pp** [−15, +5]; combo–Lazy **−11,25 pp** [−27,5, +3,75].
- Titanium normal: combo–main **−15,0 pp** [−30, 0]; combo–Lazy **−7,5 pp** [−25, +10].
- Claustrophobia central: combo–main **+3,75 pp** [−5, +12,5].

Os intervalos não autorizam afirmar um defeito sistemático de Elo. A piora Titanium é suficientemente preocupante para manter 4.02 bloqueada, mas **não** distingue erro de implementação de interação adaptativa ou flutuação.

## Experimento mínimo justificado, não executado

Selecionados **12 estados reais** de divergência, entre normais/centrais, Claustrophobia/Titanium e com/sem desacordo de `nodecap`; cada caso especifica **histórico completo**, `wtime`, `btime`, incremento, lado e decisões/tempos observados. O conjunto foi exportado para `MCGS_LAZY_SPARSE_CLOCK_SPOTCHECK_12_2026-10-19.json` no material desta investigação.

**Próximo protocolo:** rodar os mesmos 12 estados e mesmos relógios **serialmente no mesmo runner**, builds main congelado `22af954c`, Lazy congelada `082abc4d`, combo do run `978cf53a`, Adaptive ON e OFF com o patch diagnóstico de #59/#60. Registrar `diag_sims`, `diag_nodes`, `diag_target`, `diag_class`, `diag_stop`, `diag_optimum`, `diag_expandcap`, `treeHit`; acrescentar contador de `depthCapped` nos testes dirigidos. Reproduzir também alguns prefixos como buscas sequenciais para separar raiz fria de árvore reaproveitada. Não transformar essas 12 posições em mais um torneio de centenas de partidas.

**Decisão técnica atual:** nenhuma correção de código de produção é justificada até obter reprodução determinística de erro ou demonstração de que o classificador adaptativo corta busca estrategicamente necessária. **Não fazer merge da PR #58; não apagar resultados; não repetir equal-sims 20k.**
