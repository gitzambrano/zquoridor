// test_endgame_race.cpp -- sanidade/regressão do solver exato de final
// "mãos vazias" (src/endgame_race.hpp, plano-additional.md Prioridade 4).
//
// Estratégia: os Níveis 1/2 (raceETAGate/raceDisjointGate) são atalhos
// baratos que, por construção, TÊM que concordar com o Serviço B exato
// (raceExactDTM) sempre que decidirem alguma coisa -- então o teste mais
// importante aqui não é "o resultado X está certo" isolado, é "sempre que
// um nível barato decide, o resultado bate exatamente com a DP retrógrada
// completa" -- roda isso em muitas posições aleatórias (com e sem muro) e
// falha se algum nível barato decidir errado.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include "rules.hpp"
#include "endgame_race.hpp"
#include "search.hpp"
using namespace qr;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FALHOU: %s (linha %d)\n", msg, __LINE__); failures++; } \
} while (0)

// Caso trivial: peão 0 a 1 passo da meta, turno dele -> vitória em dtm=1,
// independente de onde o peão 1 esteja (desde que não atrapalhe).
static void testTrivialOneMoveWin() {
    uint64_t wallsH = 0, wallsV = 0;
    int pawn0 = cellIdx(N - 2, 4);  // 1 passo da fileira de meta (linha N-1)
    int pawn1 = cellIdx(N - 1, 0);  // canto oposto, longe da própria meta (linha 0) e não interfere
    RaceOutcome ro = resolveEmptyHandedEndgame(wallsH, wallsV, pawn0, pawn1, /*turn=*/0);
    CHECK(ro.winner == 0, "vitoria trivial em 1 lance: vencedor esperado = jogador 0");
    CHECK(ro.dtm == 1, "vitoria trivial em 1 lance: dtm esperado = 1");
}

// Corrida em tabuleiro aberto (sem muro nenhum), os dois peões na MESMA
// coluna -- caminhos mais curtos se sobrepõem em todo o corredor central,
// então nenhum atalho barato (Nível 1/2) deveria decidir aqui: cai
// obrigatoriamente no Serviço B exato. Não assume uma fórmula fechada
// pra quem vence -- essa é exatamente a armadilha que o Serviço B existe
// pra evitar (colisão de frente permite ao perseguidor "pular" o líder,
// podendo inverter quem chega primeiro em relação à intuição ingênua de
// "quem move primeiro, com mesma distância, vence") -- só verifica que
// nenhum nível barato decide sozinho uma posição ambígua, e que o
// resultado é autoconsistente entre as duas rotas de cálculo (com e sem
// os atalhos).
static void testOpenBoardStraightRace() {
    uint64_t wallsH = 0, wallsV = 0;
    int pawn0 = cellIdx(0, 4);
    int pawn1 = cellIdx(N - 1, 4);
    int rawDist0 = shortestPathLen(wallsH, wallsV, pawn0, 0);
    int rawDist1 = shortestPathLen(wallsH, wallsV, pawn1, 1);
    CHECK(rawDist0 == N - 1 && rawDist1 == N - 1, "corrida aberta: distancias brutas esperadas == N-1");

    for (int turn = 0; turn < 2; turn++) {
        int w, dtm;
        CHECK(!raceETAGate(rawDist0, rawDist1, turn, w, dtm),
              "corrida aberta: gap de tempo pequeno (mesma distancia) nao deveria decidir no Nivel 1");
        CHECK(!raceDisjointGate(wallsH, wallsV, pawn0, pawn1, turn, rawDist0, rawDist1, w, dtm),
              "corrida aberta: caminhos na MESMA coluna se sobrepoem -- Nivel 2 nao deveria decidir");

        RaceOutcome ro = resolveEmptyHandedEndgame(wallsH, wallsV, pawn0, pawn1, turn);
        RaceOutcome exact = raceExactDTM(wallsH, wallsV, pawn0, pawn1, turn);
        CHECK(exact.winner == ro.winner && exact.dtm == ro.dtm,
              "corrida aberta: Servico B exato deveria concordar com resolveEmptyHandedEndgame");
        CHECK(ro.winner == 0 || ro.winner == 1, "corrida aberta: resultado deveria ser decisivo (sem empate numa colisao frontal de coluna unica)");
    }
}

// Gate de disjuncao: constrói uma topologia de muro que separa o
// tabuleiro em duas metades por coluna (jogador 0 confinado à metade
// esquerda, jogador 1 à direita) -- caminhos mais curtos garantidamente
// disjuntos, então raceDisjointGate tem que decidir sem cair no Serviço B,
// e o resultado tem que bater com a DP exata mesmo assim.
static void testDisjointGateAgreesWithExact() {
    // Muro vertical formando uma parede contínua na coluna 4 (slots V em
    // r=0..7, c=3) -- separa colunas 0-3 (esquerda) de colunas 4-8
    // (direita). wallSlotAvailable normalmente proibiria muros
    // colineares adjacentes formando parede idêntica na mesma "trilha"
    // por causa da regra de sobreposição -- aqui construímos o bitboard
    // diretamente (sem passar por wallSlotAvailable/legalWallMoves), já
    // que só queremos testar o solver de final com uma topologia
    // congelada arbitrária, não a legalidade de chegar nela.
    uint64_t wallsH = 0, wallsV = 0;
    for (int r = 0; r < WS; r++) wallsV |= (1ull << slotIdx(r, 3));

    int pawn0 = cellIdx(4, 1);  // metade esquerda
    int pawn1 = cellIdx(4, 6);  // metade direita
    // Confirma que a parede realmente separa os dois lados antes de testar
    // o solver: a REGIÃO INTEIRA alcançável por cada jogador (não só o
    // caminho mais curto -- ver nota de correção em endgame_race.hpp sobre
    // por que shortest-path-mask sozinha não é a base certa pra essa
    // garantia) fica inteiramente do seu lado da parede.
    auto sanityMask0 = reachableRegionMask(wallsH, wallsV, pawn0);
    auto sanityMask1 = reachableRegionMask(wallsH, wallsV, pawn1);
    bool overlaps = false;
    for (int i = 0; i < N * N; i++) if (sanityMask0[i] && sanityMask1[i]) overlaps = true;
    CHECK(!overlaps, "setup do teste: regioes alcancaveis deveriam ficar em lados opostos da parede");

    for (int turn = 0; turn < 2; turn++) {
        int rawDist0 = shortestPathLen(wallsH, wallsV, pawn0, 0);
        int rawDist1 = shortestPathLen(wallsH, wallsV, pawn1, 1);
        int w, dtm;
        bool decided = raceDisjointGate(wallsH, wallsV, pawn0, pawn1, turn, rawDist0, rawDist1, w, dtm);
        CHECK(decided, "gate de disjuncao: deveria decidir quando os caminhos estao fisicamente separados por parede");
        RaceOutcome exact = raceExactDTM(wallsH, wallsV, pawn0, pawn1, turn);
        CHECK(exact.winner == w && exact.dtm == dtm, "gate de disjuncao: resultado deveria bater com o Servico B exato");
    }
}

// Gate de ETA (raceETAGate): NÃO faz parte do pipeline de decisão de
// resolveEmptyHandedEndgame (ver nota grande em endgame_race.hpp -- achado
// desta suíte: a margem do item 4c do plano-additional.md não é segura em
// geral, bloqueio físico pode custar mais que 1 lance quando os caminhos
// se cruzam). Este teste cobre só o caso em que ela É comprovadamente
// segura por construção (vitória em 1 lance -- nenhuma interação é
// possível dentro de um único lance, então a "margem" nem entra em jogo
// de verdade aqui).
static void testETAGateSafeInOneMoveWinCase() {
    uint64_t wallsH = 0, wallsV = 0;
    int pawn0 = cellIdx(N - 2, 4);       // 1 passo da meta
    int pawn1 = cellIdx(N - 1, 4);       // 8 passos da propria meta (linha 0) -- o mais longe possivel
    for (int turn = 0; turn < 2; turn++) {
        int rawDist0 = shortestPathLen(wallsH, wallsV, pawn0, 0);
        int rawDist1 = shortestPathLen(wallsH, wallsV, pawn1, 1);
        int w, dtm;
        bool decided = raceETAGate(rawDist0, rawDist1, turn, w, dtm);
        CHECK(decided, "gate de ETA: deveria decidir com gap de tempo grande (1 passo vs 8 passos)");
        CHECK(w == 0, "gate de ETA: jogador 0 (1 passo da meta) deveria vencer independente do turno");
        RaceOutcome exact = raceExactDTM(wallsH, wallsV, pawn0, pawn1, turn);
        CHECK(exact.winner == w && exact.dtm == dtm, "gate de ETA: resultado deveria bater com o Servico B exato");
    }
}

// Empate por perseguição infinita (winner == -1): nenhum outro teste
// desta suíte exercitava esse ramo -- os testes acima usam só topologias
// alcançadas por partidas reais (caminhada aleatória a partir da posição
// inicial), e 10.000 partidas assim, checadas manualmente numa sessão de
// depuração separada, não produziram nenhum empate. Esta posição foi
// achada por busca direta sobre topologias de muro SINTÉTICAS (bitboards
// construídos diretamente, mesmo espírito de
// testDisjointGateAgreesWithExact acima -- não precisa ser alcançável por
// uma sequência legal real de lances de muro pra ser um teste válido do
// solver, só precisa ser uma topologia congelada consistente com as
// regras de movimento).
//
// A mesma posição também é o contraexemplo que motivou a correção de
// `raceDisjointGate` (ver nota grande em endgame_race.hpp): os conjuntos
// de caminho MAIS CURTO dos dois jogadores são disjuntos (colunas 2-3 vs.
// 4-5), mas a REGIÃO alcançável de cada um cobre o tabuleiro inteiro (as
// 81 casas são um único componente conexo) -- o gate antigo (baseado só
// em caminho mínimo) decidia errado "vitória do jogador 0 em 10 lances"
// no turno do jogador 1, quando o resultado verdadeiro (confirmado por
// duas reimplementações independentes fora desta suíte, além do Serviço
// B) é EMPATE. Por isso o teste é assimétrico por turno: só o turno do
// jogador 1 é empate; no turno do jogador 0, o jogador 0 já vence em 9
// lances antes que o bloqueio tenha chance de se estabelecer -- ambos os
// resultados são checados explicitamente para não mascarar uma futura
// regressão em nenhum dos dois sentidos.
static void testInfinitePursuitDraw() {
    uint64_t wallsH = 0x48000008000000ull, wallsV = 0x8014020000022000ull;
    int pawn0 = cellIdx(6, 3);  // cellIdx(r,c) = r*N+c -> 57
    int pawn1 = cellIdx(5, 4);  // -> 49
    CHECK(shortestPathLen(wallsH, wallsV, pawn0, 0) >= 0,
          "empate: setup do teste -- jogador 0 precisa ter caminho ate a propria meta isoladamente");
    CHECK(shortestPathLen(wallsH, wallsV, pawn1, 1) >= 0,
          "empate: setup do teste -- jogador 1 precisa ter caminho ate a propria meta isoladamente");

    // Turno do jogador 0: vitoria forcada em 9 lances (nao ha empate aqui).
    {
        RaceOutcome exact = raceExactDTM(wallsH, wallsV, pawn0, pawn1, 0);
        CHECK(exact.winner == 0 && exact.dtm == 9,
              "empate (turno 0): jogador 0 deveria vencer forcado em 9 lances, sem empate, nesta topologia");
        RaceOutcome ro = resolveEmptyHandedEndgame(wallsH, wallsV, pawn0, pawn1, 0);
        CHECK(ro.winner == exact.winner && ro.dtm == exact.dtm,
              "empate (turno 0): resolveEmptyHandedEndgame deveria concordar com o Servico B");
    }
    // Turno do jogador 1: empate por perseguicao infinita -- o caso que
    // motivou a correcao do gate de disjuncao.
    {
        RaceOutcome exact = raceExactDTM(wallsH, wallsV, pawn0, pawn1, 1);
        CHECK(exact.winner == -1, "empate (turno 1): raceExactDTM deveria devolver winner==-1 (perseguicao infinita) nesta topologia");
        RaceOutcome ro = resolveEmptyHandedEndgame(wallsH, wallsV, pawn0, pawn1, 1);
        CHECK(ro.winner == -1, "empate (turno 1): resolveEmptyHandedEndgame (pipeline completo) deveria concordar com o Servico B");
    }

    // Checagem direta da correção: com a base sonora (regiao alcancavel),
    // o gate de disjuncao NAO pode decidir nesta posicao em nenhum turno
    // -- as regioes se tocam (tabuleiro inteiro conectado), entao ele tem
    // que recusar e deixar o Servico B resolver. Se isso voltar a
    // "decidir" aqui, e sinal de regressao pro bug antigo (base de
    // caminho minimo em vez de regiao).
    for (int turn = 0; turn < 2; turn++) {
        int rawDist0 = shortestPathLen(wallsH, wallsV, pawn0, 0);
        int rawDist1 = shortestPathLen(wallsH, wallsV, pawn1, 1);
        int w, dtm;
        bool decided = raceDisjointGate(wallsH, wallsV, pawn0, pawn1, turn, rawDist0, rawDist1, w, dtm);
        CHECK(!decided, "empate: gate de disjuncao NAO deveria decidir aqui -- regioes alcancaveis se tocam (bug antigo usava so caminho minimo)");
    }
}

// Regressão ampla: gera topologias de muro aleatórias (legais, via
// caminhada aleatória de partida real) até os dois jogadores ficarem sem
// muro, então checa -- para várias posições de peão dentro dessa
// topologia -- que sempre que um nível barato (ETA ou disjunção) decide,
// a resposta bate exatamente com o Serviço B. Não gera posição de peão
// aleatória livre (poderia cair numa célula sem caminho até a meta em
// algumas topologias patológicas fora do alcance real de jogo) -- em vez
// disso usa só os peões que a própria caminhada aleatória alcançou, que
// são garantidamente válidos.
static void testRandomWallTopologiesGatesAgreeWithExact() {
    std::mt19937 rng(20260721);
    int decidedByGate = 0, checkedTotal = 0;
    for (int game = 0; game < 40; game++) {
        State s = initialState();
        for (int ply = 0; ply < 200 && (s.wallsLeft[0] > 0 || s.wallsLeft[1] > 0); ply++) {
            MoveList moves = legalMoves(s);
            if (moves.empty()) break;
            s = applyMove(s, moves[rng() % moves.size()]);
        }
        if (s.wallsLeft[0] != 0 || s.wallsLeft[1] != 0) continue;  // partida curta demais, não zerou os dois

        for (int turn = 0; turn < 2; turn++) {
            RaceOutcome viaGates = resolveEmptyHandedEndgame(s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], turn);
            RaceOutcome exact = raceExactDTM(s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], turn);
            checkedTotal++;
            if (!(viaGates.winner == exact.winner && viaGates.dtm == exact.dtm)) {
                std::printf("  DIVERGENCIA game=%d turn=%d pawn0=%d pawn1=%d wallsH=0x%llx wallsV=0x%llx viaGates=(%d,%d) exact=(%d,%d)\n",
                    game, turn, s.pawn[0], s.pawn[1], (unsigned long long)s.wallsH, (unsigned long long)s.wallsV,
                    viaGates.winner, viaGates.dtm, exact.winner, exact.dtm);
            }
            CHECK(viaGates.winner == exact.winner && viaGates.dtm == exact.dtm,
                  "topologia aleatoria: resolveEmptyHandedEndgame (com atalhos) deveria sempre bater com raceExactDTM puro");

            int rawDist0 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0);
            int rawDist1 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[1], 1);
            int w, dtm;
            if (raceETAGate(rawDist0, rawDist1, turn, w, dtm) ||
                raceDisjointGate(s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], turn, rawDist0, rawDist1, w, dtm)) {
                decidedByGate++;
            }
        }
    }
    CHECK(checkedTotal > 0, "topologia aleatoria: deveria ter encontrado pelo menos uma partida que zerou os dois lados");
    std::printf("  (info) topologias mao-vazia checadas=%d, decididas por atalho barato=%d\n", checkedTotal, decidedByGate);
}

// Regressão de PERFORMANCE (não só de corretude) -- achado numa sessão de
// arena externa: depois da correção do Nível 2 acima (base sonora =
// região alcançável, que decide bem menos vezes que o gate antigo
// baseado em caminho mínimo), a maioria das topologias de tabuleiro reais
// passou a cair no Serviço B (raceExactDTM) em quase todo nó da fase
// "mãos vazias" -- e cada chamada reconstruía o grafo de 13.122 estados e
// rodava as duas BFS retrógradas DO ZERO, medido em ~790 microssegundos
// por chamada (não "microssegundos" desprezíveis como o comentário
// original assumia). Como wallsLeft[0]==0 && wallsLeft[1]==0 significa
// que nenhum muro pode mais ser colocado pro resto daquela subárvore de
// busca -- ou seja, wallsH/wallsV ficam CONGELADOS por potencialmente
// milhares de nós consecutivos -- a correção foi cachear a DP inteira por
// topologia de muro (ver nota de correção #2 em endgame_race.hpp). Sem
// esse cache, um motor de arena real media queda de nós/s de mais de
// 50x nessa fase (confirmado com benchmark ponta-a-ponta numa sessão de
// depuração separada, usando Negamax::chooseMove de verdade com
// orçamento de tempo por lance: ~1.600 nós/s sem cache vs. ~84.000 nós/s
// com cache, mesma posição, mesmo orçamento). Este teste não repete
// aquele benchmark completo (chooseMove real, custoso demais pra rodar
// toda vez que a suíte roda), só verifica que MUITAS chamadas
// consecutivas com a MESMA topologia de muro (o padrão real dentro de uma
// subárvore) não degradam pra escala de "microssegundos por chamada" --
// se o cache for removido ou quebrado por um refactor futuro, este teste
// deve estourar o limite de tempo por uma margem folgada (o limite usado
// aqui é propositalmente frouxo -- várias ordens de grandeza acima do
// custo esperado com cache -- pra não ficar frágil em máquinas lentas ou
// carregadas, só pra pegar uma regressão real de "sem cache nenhum").
static void testRepeatedQueriesSameTopologyAreFast() {
    std::mt19937 rng(2026);
    State s = initialState();
    for (int ply = 0; ply < 200 && (s.wallsLeft[0] > 0 || s.wallsLeft[1] > 0); ply++) {
        MoveList moves = legalMoves(s);
        if (moves.empty()) break;
        s = applyMove(s, moves[rng() % moves.size()]);
    }
    CHECK(s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0,
          "perf: setup do teste -- partida deveria ter zerado os muros dos dois lados");

    constexpr int NCALLS = 5000;
    auto t0 = std::chrono::steady_clock::now();
    long acc = 0;  // só pra impedir o compilador de eliminar as chamadas
    for (int i = 0; i < NCALLS; i++) {
        RaceOutcome ro = resolveEmptyHandedEndgame(s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], i % 2);
        acc += ro.dtm;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  (info) perf: %d chamadas com a mesma topologia em %.4fs (%.0f chamadas/s, acc=%ld)\n",
                NCALLS, secs, NCALLS / secs, acc);
    // Sem cache, isso mede ~0,79s (1267 chamadas/s) nesta máquina -- um
    // limite de 0,3s dá margem generosa (>4x) sem exigir uma máquina
    // rápida, e ainda pega com folga se o cache for removido por
    // completo. Com o cache, isso roda em bem menos de 0,1s na prática.
    CHECK(secs < 0.3,
          "perf: 5000 chamadas com a MESMA topologia de muro deveriam reusar o cache (achado de regressao real de nos/s em arena)");
}

// Regressão do bug de ESCOLHA DE LANCE (não de valor) achado numa arena
// externa: quando a RAIZ real (não um nó interno) já satisfaz
// wallsLeft==(0,0), negamax(root,...) resolve por atalho e devolve só o
// VALOR -- sem gerar/comparar os próprios filhos, porque é tratado como
// nó-folha (igual winner()). Antes da correção em chooseMove, o "melhor
// lance" lido da TT nesse caso era um PLACEHOLDER (legalMoves(s)[0], só
// pra garantir legalidade -- ver nota em negamax) e não o lance que de
// fato realiza o DTM ótimo -- o motor "sabia" o resultado mas jogava
// lances arbitrários pra chegar lá, perdendo partidas que já tinha
// ganho na teoria. Este teste constrói uma posição óbvia (tabuleiro
// aberto, sem muro, jogador 0 a 2 passos da própria meta, oponente longe
// o bastante pra não interferir) onde SÓ o lance "andar pra frente"
// reduz o DTM -- qualquer outro lance (lateral/afastando) é estritamente
// pior -- e confirma que chooseMove escolhe o lance certo, não só que
// resolveEmptyHandedEndgame calcula o valor certo (isso já é coberto
// pelos testes acima).
static void testChooseMoveAtEmptyHandedRootPicksOptimalMove() {
    State root = initialState();
    root.wallsLeft[0] = 0;
    root.wallsLeft[1] = 0;
    root.wallsH = 0;
    root.wallsV = 0;
    root.pawn[0] = cellIdx(N - 3, 4);  // 2 passos da própria meta (linha N-1)
    root.pawn[1] = cellIdx(4, 8);      // linha do meio, canto lateral -- longe o bastante pra nao
                                        // interferir/pular, e (importante) NAO em cima da meta de
                                        // nenhum dos dois lados (linha 0 = meta do jogador 1,
                                        // linha N-1 = meta do jogador 0 -- ambas evitadas de propósito)
    root.turn = 0;
    root.hash = zobrist().pawnKey[0][root.pawn[0]] ^ zobrist().pawnKey[1][root.pawn[1]];

    Negamax eng;
    SearchStats st;
    Move m = eng.chooseMove(root, 20, 50, st);  // orcamento pequeno de proposito -- deveria resolver na hora, sem precisar de busca
    State ns = applyMove(root, m);

    CHECK(rowOf(ns.pawn[0]) == N - 2,
          "chooseMove na raiz ja em maos-vazias: deveria escolher o lance que anda pra frente (reduz DTM), nao um lance lateral/arbitrario");

    // Confere tambem que o valor resultante bate com o Servico B exato
    // pra essa transicao especifica (nao só a direcao do lance).
    RaceOutcome exact = raceExactDTM(ns.wallsH, ns.wallsV, ns.pawn[0], ns.pawn[1], ns.turn);
    CHECK(exact.winner == 0 && exact.dtm == 2,
          "chooseMove na raiz ja em maos-vazias: apos o lance escolhido, deveria restar vitoria do jogador 0 em 2 plies (1 lance do oponente + o lance final do jogador 0 -- dtm conta plies, nao 'lances proprios')");
}

int main() {
    testTrivialOneMoveWin();
    testOpenBoardStraightRace();
    testDisjointGateAgreesWithExact();
    testETAGateSafeInOneMoveWinCase();
    testInfinitePursuitDraw();
    testRandomWallTopologiesGatesAgreeWithExact();
    testRepeatedQueriesSameTopologyAreFast();
    testChooseMoveAtEmptyHandedRootPicksOptimalMove();

    if (failures == 0) {
        std::printf("OK -- todos os testes de endgame_race passaram\n");
        return 0;
    }
    std::printf("%d FALHA(S)\n", failures);
    return 1;
}
