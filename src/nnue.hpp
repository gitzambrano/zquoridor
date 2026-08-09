// nnue.hpp -- rede estilo NNUE para a Fase 5 do plano, aqui com pesos
// ALEATÓRIOS (não treinados). Serve pra: (1) validar que o acumulador é
// sempre incremental pros dois tipos de lance, (2) medir o custo real do
// forward pass, (3) servir de "motor de pesos aleatórios" no lugar de um
// motor puramente aleatório-no-lance.
#pragma once
#include <array>
#include <random>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include "rules.hpp"

namespace qr {

// DIST_BUCKETS: distância BFS (shortestPathLen, rules.hpp) até a linha de
// chegada, codificada em bucket one-hot (não como escalar cru). Por quê
// one-hot e não um único float com o valor da distância: o resto da rede
// já é inteiramente sparse-linear (soma de linhas de w1 por feature
// ativa) -- um escalar exigiria uma coluna de entrada com "range" != 0/1,
// o que quebraria a simetria de escala usada na quantização inteira
// (Seção 7.8: QA assume ativação em [0,QA], coerente com SCReLU em [0,1];
// um escalar de distância cru viveria numa escala totalmente diferente).
// One-hot mantém a mesma família de feature dos outros 290 bits e reusa
// exatamente o mesmo mecanismo incremental (remove bucket antigo, soma
// bucket novo) já usado pra peão. Custo: 2*DIST_BUCKETS colunas extras em
// w1, nada nas cabeças -- a rede continua leve (ver nota de tamanho perto
// de NUM_FEATURES).
//
// 21 buckets (0..19 exatos, 20 = "20 ou mais"): cobre com granularidade
// total a faixa observada em partidas reais (distância inicial é 8;
// muros bem jogados alongam pra algo como 12-18 num jogo competitivo) e
// satura de forma grosseira só as posições bem raras/patológicas com
// muitos muros formando labirinto -- nessas o valor exato importa menos
// que "sei que está muito longe".
constexpr int DIST_BUCKETS = 21;
inline int distBucket(int dist) {
    if (dist < 0) return 0;             // defensivo: shortestPathLen só devolve -1 se não há
                                         // caminho, o que a legalidade de muro já impede.
    return dist >= DIST_BUCKETS ? DIST_BUCKETS - 1 : dist;
}

// WALLS_LEFT_BUCKETS: muros restantes de cada jogador (0..WALLS_PER_PLAYER,
// one-hot -- mesma família de codificação de DIST_BUCKETS acima, mesmo
// motivo: mantém tudo na família sparse-linear/soma-de-linhas-de-w1 usada
// pro resto do acumulador). ANTES desta sessão (2026-08), wallsLeft NÃO
// era feature nenhuma -- confirmado por teste (nnue_sign_check.cpp): duas
// posições idênticas em peões/muros no tabuleiro, diferindo só em
// "jogador 0 com 10 muros vs. jogador 1 com 0", davam eval NNUE
// *idêntico*, porque a rede simplesmente não via essa informação (só
// entrava em evalSimple, a heurística). Rules.hpp já grava
// State::wallsLeft[2] com WALLS_PER_PLAYER=10 de default, e o formato de
// self-play (read_selfplay.py: walls_left_own/walls_left_opp) já registra
// isso por posição -- só faltava usar. WALLS_LEFT_BUCKETS = 11 (0..10,
// sem "ou mais" como em DIST_BUCKETS -- a contagem real nunca passa de
// WALLS_PER_PLAYER, não precisa de bucket de saturação).
constexpr int WALLS_LEFT_BUCKETS = WALLS_PER_PLAYER + 1;   // 11
inline int wallsLeftBucket(int n) {
    if (n < 0) return 0;                              // defensivo
    return n >= WALLS_LEFT_BUCKETS ? WALLS_LEFT_BUCKETS - 1 : n;
}

// NUM_FEATURES = 81 (peão próprio) + 81 (peão oponente) + 64 (muro H) +
// 64 (muro V) + 21 (bucket dist. própria) + 21 (bucket dist. oponente) +
// 11 (bucket muros restantes próprios) + 11 (bucket muros restantes do
// oponente) = 354. Mudança de arquitetura (2026-08): pesos treinados para
// NUM_FEATURES=332 são INCOMPATÍVEIS com esta versão -- não dá pra fazer
// warm-start via --init-from de um checkpoint antigo (o fingerprint em
// compute_fingerprint()/try_load_train_state() já detecta e recusa isso).
// É preciso retreinar do zero com training/train_nnue.py atualizado.
constexpr int NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS;  // 354
constexpr int HIDDEN = 256;
constexpr int POLICY_OUT = N * N + WS * WS * 2;             // 81 destino peão + 128 muro = 209

// Espelha a coordenada bruta do tabuleiro para a perspectiva do jogador 1
// (CORREÇÃO 2026-08 -- bug de assimetria de perspectiva encontrado via
// nnue_sign_check.cpp: a posição inicial, perfeitamente simétrica, dava
// eval NNUE diferente pras duas perspectivas, porque featOwnPawn/
// featWallH/featWallV usavam coordenada CRUA do tabuleiro, só trocando
// qual peão é "meu" -- sem espelhar linha/coluna. Como o jogador 0 nasce
// na linha 0 e vai pra linha N-1 (GOAL_ROW em rules.hpp) e o jogador 1 é o
// espelho disso (nasce em N-1, vai pra 0), canonicalizar exige espelhar a
// LINHA (r -> N-1-r pra peão / WS-1-r pra muro) quando a perspectiva é do
// jogador 1 -- a coluna não muda, o tabuleiro não tem assimetria nesse
// eixo. Perspectiva 0 continua sem transformação nenhuma (é a canônica);
// SEM isso, a rede precisava aprender duas "geografias" diferentes
// usando as mesmas colunas de peso, uma pra cada perspectiva.
inline int mirroredPawnCell(int cell, int perspective) {
    if (perspective == 0) return cell;
    return cellIdx(N - 1 - rowOf(cell), colOf(cell));
}
inline int mirroredWallSlot(int slot, int perspective) {
    if (perspective == 0) return slot;
    int r = slot / WS, c = slot % WS;
    return (WS - 1 - r) * WS + c;
}

// Espelha um bitboard de muro inteiro (todos os 64 bits) -- usado por
// selfplay.hpp pra gravar TrainingSample::wallsH/wallsV JÁ canônicos
// (mesma perspectiva do mover), em vez de crus. Motivo de gravar já
// espelhado em vez de espelhar depois em Python: o formato binário de
// self-play NÃO registra a identidade física (0/1) de quem jogou --
// só existe aqui, em selfplay.hpp, no momento da gravação (mesmo
// raciocínio já usado pra TrainingSample::ownDist/oppDist, ver comentário
// lá). Sem isso, seria impossível reconstruir no lado Python se uma
// amostra precisa ser espelhada ou não.
inline uint64_t mirrorWallBitboard(uint64_t bits, int perspective) {
    if (perspective == 0) return bits;
    uint64_t out = 0;
    for (int i = 0; i < WS * WS; i++) {
        if ((bits >> i) & 1ull) out |= (1ull << mirroredWallSlot(i, perspective));
    }
    return out;
}

// Espelha um Move inteiro (mesma perspectiva) -- usado por selfplay.hpp
// pra gravar TrainingSample::policyTarget já no mesmo referencial
// canônico de ownPawn/oppPawn/wallsH/wallsV, senão o alvo de policy
// ficaria dessincronizado da entrada (rótulo em coordenada crua, feature
// em coordenada espelhada) pra metade das amostras (as gravadas com
// mover=jogador 1). A orientação do muro (H/V) não muda no espelho --
// só a linha, nunca a coluna nem a orientação.
inline Move mirrorMoveForPerspective(const Move& m, int perspective) {
    if (perspective == 0) return m;
    if (!m.isWall) return Move::pawn(mirroredPawnCell(m.a, perspective));
    int mirroredSlot = mirroredWallSlot(slotIdx(m.b, m.c), perspective);
    return Move::wall(m.a, mirroredSlot / WS, mirroredSlot % WS);
}

// índices de feature: [0,81) peão próprio, [81,162) peão oponente,
// [162,226) muro H, [226,290) muro V, [290,311) bucket dist. própria,
// [311,332) bucket dist. oponente, [332,343) bucket muros restantes
// próprios, [343,354) bucket muros restantes do oponente.
//
// featOwnPawn/featOppPawn/featWallH/featWallV agora recebem `perspective`
// (0 ou 1, mesmo valor passado a buildAccumulator/buildAccumulatorQuant)
// e aplicam o espelho acima -- todo call site precisa passar a
// perspectiva correta (ver updateAccumulatorForMove(Quant): a perspectiva
// de um Accumulator não muda durante sua vida, então "viewerPlayer",
// calculado uma vez por lance a partir de viewerIsMover/mover, É a
// perspectiva).
inline int featOwnPawn(int cell, int perspective) { return mirroredPawnCell(cell, perspective); }
inline int featOppPawn(int cell, int perspective) { return N * N + mirroredPawnCell(cell, perspective); }
inline int featWallH(int slot, int perspective) { return N * N + N * N + mirroredWallSlot(slot, perspective); }
inline int featWallV(int slot, int perspective) { return N * N + N * N + WS * WS + mirroredWallSlot(slot, perspective); }
constexpr int DIST_FEAT_BASE = N * N + N * N + WS * WS * 2;              // 290
inline int featOwnDist(int bucket) { return DIST_FEAT_BASE + bucket; }
inline int featOppDist(int bucket) { return DIST_FEAT_BASE + DIST_BUCKETS + bucket; }
constexpr int WALLS_LEFT_FEAT_BASE = DIST_FEAT_BASE + 2 * DIST_BUCKETS;  // 332
inline int featOwnWallsLeft(int bucket) { return WALLS_LEFT_FEAT_BASE + bucket; }
inline int featOppWallsLeft(int bucket) { return WALLS_LEFT_FEAT_BASE + WALLS_LEFT_BUCKETS + bucket; }

// Duas cabeças de "valor" (mudança desta sessão, ver plano_quoridor.md):
// antes havia uma única saída escalar alimentando duas losses ao mesmo
// tempo (MSE contra a heurística E BCE contra o resultado — daí o
// "value" acabar preso perto de zero, ver comentário histórico em
// train_nnue.py). Agora são duas cabeças INDEPENDENTES, cada uma com seu
// próprio bottleneck 256->32->1, sem nada compartilhado entre elas além
// do acumulador (a):
//   - resultado (WL, sem empate): único logit, só BCE-with-logits contra
//     o resultado real da partida. É esta que deve dominar quando o
//     self-play passar a vir da própria NNUE (hoje ainda usa evalSimple).
//   - auxiliar de imitação da heurística: regressão MSE contra
//     search_score/VALUE_SCALE. Peso configurável via --w-score,
//     pensada para ir a 0 (ou a cabeça inteira ser removida) numa sessão
//     futura, quando o self-play deixar de depender do evalSimple.
// Duas cabeças independentes (em vez de uma cabeça com uma bifurcação só
// no último Linear) foi a escolha deliberada aqui: quando a cabeça
// auxiliar for removida numa sessão futura, basta apagar wv1_aux/bv1_aux/
// wv2_aux/bv2_aux (e o bloco correspondente no loop de treino) sem tocar
// em nada da cabeça de resultado. Custo: dobra o tamanho da "parte de
// valor" da rede (~8,4 mil parâmetros a mais) — desprezível frente aos
// ~85 mil de w1 (332*256).
struct NNUEWeights {
    // camada 1 (acumulador): pesos por feature esparsa -> HIDDEN
    std::vector<std::array<float, HIDDEN>> w1;   // [NUM_FEATURES][HIDDEN]
    std::array<float, HIDDEN> b1{};
    // cabeça de RESULTADO (WL): HIDDEN -> 32 -> 1 (logit único, sem empate)
    std::array<std::array<float, 32>, HIDDEN> wv1_wl;
    std::array<float, 32> bv1_wl{};
    std::array<float, 32> wv2_wl{};
    float bv2_wl = 0.f;
    // cabeça AUXILIAR (imitação da heurística evalSimple): HIDDEN -> 32 -> 1
    std::array<std::array<float, 32>, HIDDEN> wv1_aux;
    std::array<float, 32> bv1_aux{};
    std::array<float, 32> wv2_aux{};
    float bv2_aux = 0.f;
    // cabeça de política: HIDDEN -> POLICY_OUT
    std::vector<std::array<float, HIDDEN>> wp;   // [POLICY_OUT][HIDDEN] (transposto p/ dot direto)
    std::vector<float> bp;                        // [POLICY_OUT]

    NNUEWeights() { randomInit(12345); }

    void randomInit(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> d1(0.f, 0.05f);   // escala pequena: rede não treinada, só demo/benchmark
        w1.assign(NUM_FEATURES, {});
        for (auto& row : w1) for (auto& v : row) v = d1(rng);
        for (auto& v : b1) v = 0.f;
        for (auto& row : wv1_wl) for (auto& v : row) v = d1(rng);
        for (auto& v : wv2_wl) v = d1(rng);
        for (auto& row : wv1_aux) for (auto& v : row) v = d1(rng);
        for (auto& v : wv2_aux) v = d1(rng);
        wp.assign(POLICY_OUT, {});
        for (auto& row : wp) for (auto& v : row) v = d1(rng);
        bp.assign(POLICY_OUT, 0.f);
    }

    // Carrega pesos treinados (Fase 5, training/train_nnue.py) de um arquivo
    // binário cru de floats, na MESMA ordem em que os campos aparecem nesta
    // struct: w1 (332x256, ver NUM_FEATURES -- 290 originais + 42 de bucket
    // de distância BFS), b1 (256), wv1_wl (256x32), bv1_wl (32), wv2_wl (32),
    // bv2_wl (1), wv1_aux (256x32), bv1_aux (32), wv2_aux (32), bv2_aux (1),
    // wp (209x256), bp (209). Cada bloco é lido linha a linha (mesma ordem
    // de laço usada no export Python), sem cabeçalho. LAYOUT MUDOU nesta
    // sessão (cabeça de valor única -> duas cabeças) -- pesos exportados
    // antes da mudança não são compatíveis, precisam ser re-treinados.
    bool loadFromFile(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        bool ok = true;
        w1.assign(NUM_FEATURES, {});
        for (auto& row : w1) ok = ok && std::fread(row.data(), sizeof(float), HIDDEN, f) == (size_t)HIDDEN;
        ok = ok && std::fread(b1.data(), sizeof(float), HIDDEN, f) == (size_t)HIDDEN;
        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(bv1_wl.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(wv2_wl.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(&bv2_wl, sizeof(float), 1, f) == 1;
        for (auto& row : wv1_aux) ok = ok && std::fread(row.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(bv1_aux.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(wv2_aux.data(), sizeof(float), 32, f) == 32;
        ok = ok && std::fread(&bv2_aux, sizeof(float), 1, f) == 1;
        wp.assign(POLICY_OUT, {});
        for (auto& row : wp) ok = ok && std::fread(row.data(), sizeof(float), HIDDEN, f) == (size_t)HIDDEN;
        bp.assign(POLICY_OUT, 0.f);
        ok = ok && std::fread(bp.data(), sizeof(float), POLICY_OUT, f) == (size_t)POLICY_OUT;
        std::fclose(f);
        return ok;
    }

    // Grava no mesmo layout que loadFromFile espera (usado só em testes/
    // ferramentas C++; o export "de verdade" sai direto do PyTorch em
    // training/train_nnue.py, que já escreve exatamente esse layout).
    bool saveToFile(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        for (auto& row : w1) std::fwrite(row.data(), sizeof(float), HIDDEN, f);
        std::fwrite(b1.data(), sizeof(float), HIDDEN, f);
        for (auto& row : wv1_wl) std::fwrite(row.data(), sizeof(float), 32, f);
        std::fwrite(bv1_wl.data(), sizeof(float), 32, f);
        std::fwrite(wv2_wl.data(), sizeof(float), 32, f);
        std::fwrite(&bv2_wl, sizeof(float), 1, f);
        for (auto& row : wv1_aux) std::fwrite(row.data(), sizeof(float), 32, f);
        std::fwrite(bv1_aux.data(), sizeof(float), 32, f);
        std::fwrite(wv2_aux.data(), sizeof(float), 32, f);
        std::fwrite(&bv2_aux, sizeof(float), 1, f);
        for (auto& row : wp) std::fwrite(row.data(), sizeof(float), HIDDEN, f);
        std::fwrite(bp.data(), sizeof(float), POLICY_OUT, f);
        std::fclose(f);
        return true;
    }
};

inline NNUEWeights& weights() { static NNUEWeights w; return w; }

// Substitui os pesos do singleton pelos de um arquivo treinado (ver
// NNUEWeights::loadFromFile). Deve ser chamado antes de qualquer
// buildAccumulator/forward* se o objetivo é usar a rede treinada em vez
// dos pesos aleatórios de benchmark.
inline bool loadWeights(const std::string& path) { return weights().loadFromFile(path); }

struct Accumulator {
    std::array<float, HIDDEN> v{};
    // bucket de distância BFS atualmente somado no acumulador (own=quem é
    // "perspectiva", opp=o outro) -- cache que existe só pra evitar
    // recomputar a distância ANTES do lance via BFS dentro de
    // updateAccumulatorForMove: só a distância DEPOIS do lance precisa ser
    // calculada ali, porque a de antes já está guardada aqui desde a
    // última chamada. Sem esse cache, cada update de muro pagaria 4 BFS
    // (antes+depois x 2 jogadores) em vez de 2 (só depois), o que deixava
    // o update incremental de muro mais caro que um recompute completo --
    // ver benchAccumulatorUpdate em main.cpp.
    int ownDistBucket = 0;
    int oppDistBucket = 0;
    int ownWallsLeftBucket = 0;   // cache do bucket de muros restantes ativo -- mesmo padrão de ownDistBucket
    int oppWallsLeftBucket = 0;

    void addFeature(int featIdx) {
        auto& row = weights().w1[featIdx];
        for (int i = 0; i < HIDDEN; i++) v[i] += row[i];
    }
    void removeFeature(int featIdx) {
        auto& row = weights().w1[featIdx];
        for (int i = 0; i < HIDDEN; i++) v[i] -= row[i];
    }
};

// recomputa do zero -- só usada ao entrar numa posição "fria" (raiz da busca)
// Ponte pro cache de BFS entre nós (rules.hpp: PlayerPathCacheTable /
// computeDistCached), pra updateAccumulatorForMove(Quant) parar de pagar
// shortestPathLen CRU a cada nó. `xtable == nullptr` cai pro
// comportamento antigo (shortestPathLen direto) -- mantém tune_spsa.cpp,
// testes em teste/, etc. funcionando sem precisar passar uma tabela.
// Ganho duplo ao plugar aqui: (1) o mesmo (wallsH,wallsV,pawnCell,player)
// reaparece o tempo todo entre nós irmãos/transposições (exatamente o que
// PlayerPathCacheTable já explora pro heurístico); (2) dentro de UM SÓ
// lance, updateAccumulatorForMove(Quant) é chamada 2x (uma vez por
// acumulador/perspectiva) e as duas chamadas acabam pedindo as MESMAS duas
// distâncias (a do mover e a do oponente, só que em ordem own/opp
// trocada) -- com xtable, a segunda chamada acerta cache na hora (posta
// pela primeira), o que sozinho já elimina metade das BFS pagas aqui.
inline int distLenCached(uint64_t wallsH, uint64_t wallsV, int cell, int player, PlayerPathCacheTable* xtable) {
    if (!xtable) return shortestPathLen(wallsH, wallsV, cell, player);
    PlayerPathCache c;
    computeDistCached(wallsH, wallsV, cell, player, xtable, c);
    return cachedShortestPathLen(c);
}

inline Accumulator buildAccumulator(const State& s, int perspective, PlayerPathCacheTable* xtable = nullptr) {
    Accumulator acc;
    acc.v = weights().b1;
    int me = perspective, opp = 1 - perspective;
    acc.addFeature(featOwnPawn(s.pawn[me], perspective));
    acc.addFeature(featOppPawn(s.pawn[opp], perspective));
    for (int i = 0; i < WS * WS; i++) {
        if ((s.wallsH >> i) & 1ull) acc.addFeature(featWallH(i, perspective));
        if ((s.wallsV >> i) & 1ull) acc.addFeature(featWallV(i, perspective));
    }
    // features de distância BFS (ver nota em DIST_BUCKETS acima): 2 BFS,
    // O(81) cada, sem alocação (shortestPathLen usa arrays thread_local
    // fixos) -- barato frente ao custo de recompute total do acumulador,
    // que já percorre 128 slots de muro.
    acc.ownDistBucket = distBucket(distLenCached(s.wallsH, s.wallsV, s.pawn[me], me, xtable));
    acc.oppDistBucket = distBucket(distLenCached(s.wallsH, s.wallsV, s.pawn[opp], opp, xtable));
    acc.addFeature(featOwnDist(acc.ownDistBucket));
    acc.addFeature(featOppDist(acc.oppDistBucket));
    // muros restantes (feature nova, 2026-08 -- ver nota em WALLS_LEFT_BUCKETS acima)
    acc.ownWallsLeftBucket = wallsLeftBucket(s.wallsLeft[me]);
    acc.oppWallsLeftBucket = wallsLeftBucket(s.wallsLeft[opp]);
    acc.addFeature(featOwnWallsLeft(acc.ownWallsLeftBucket));
    acc.addFeature(featOppWallsLeft(acc.oppWallsLeftBucket));
    return acc;
}

// SCReLU simplificado (clip 0..1, ao quadrado) -- mesma família usada no Zchezz.
// Usado só no acumulador (soma de muitas features, precisa de não-linearidade
// "forte" pra não colapsar em um mapa linear das features de entrada).
inline float screlu(float x) {
    float c = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
    return c * c;
}

// Clipped ReLU (clip 0..1, sem elevar ao quadrado) -- usado nas camadas
// internas dos heads (value1->value2), padrão comum em NNUE pra camadas
// já quantizáveis em int8/16. Ver nota de correção logo abaixo.
inline float clippedRelu(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// CORREÇÃO (pós-treino, Fase 5, ainda válida pras duas cabeças atuais): a
// versão original desta função não tinha nenhuma não-linearidade entre as
// duas camadas lineares do value head (256->32 e 32->1) -- o bias bv1 era
// somado só depois de já multiplicar por wv2, sem ativação no meio. Como a
// composição de duas transformações lineares é ela mesma linear, o
// "bottleneck" de 32 neurônios não tinha efeito nenhum. Corrigido
// adicionando clippedRelu(h+bv1) antes do produto final com wv2.
//
// Duas cabeças (mudança desta sessão): a saída única "value" virou duas
// funções separadas, cada uma com seu próprio bottleneck 256->32->1 (ver
// comentário em NNUEWeights acima). forwardValueWL é a que a busca vai
// usar quando a Fase 6 ligar a NNUE (logit de resultado, sem empate);
// forwardValueAux existe só para treino/validação de paridade (imitação
// da heurística evalSimple) e não é chamada pelo motor de busca.
inline float forwardValueWL(const Accumulator& acc) {
    std::array<float, 32> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        float a = screlu(acc.v[i]);
        for (int j = 0; j < 32; j++) h[j] += a * W.wv1_wl[i][j];
    }
    float out = W.bv2_wl;
    for (int j = 0; j < 32; j++) {
        float hj = clippedRelu(h[j] + W.bv1_wl[j]);
        out += hj * W.wv2_wl[j];
    }
    return out;
}

inline float forwardValueAux(const Accumulator& acc) {
    std::array<float, 32> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        float a = screlu(acc.v[i]);
        for (int j = 0; j < 32; j++) h[j] += a * W.wv1_aux[i][j];
    }
    float out = W.bv2_aux;
    for (int j = 0; j < 32; j++) {
        float hj = clippedRelu(h[j] + W.bv1_aux[j]);
        out += hj * W.wv2_aux[j];
    }
    return out;
}

inline void forwardPolicy(const Accumulator& acc, std::array<float, POLICY_OUT>& out) {
    auto& W = weights();
    std::array<float, HIDDEN> a;
    for (int i = 0; i < HIDDEN; i++) a[i] = screlu(acc.v[i]);
    for (int o = 0; o < POLICY_OUT; o++) {
        float s = W.bp[o];
        auto& row = W.wp[o];
        for (int i = 0; i < HIDDEN; i++) s += a[i] * row[i];
        out[o] = s;
    }
}

// atualização incremental do acumulador para um lance -- peão e muro
// continuam O(HIDDEN) puro (2 ou 1 feature de tabuleiro trocam, nunca
// recompute completo). As features de distância BFS são a exceção: não
// dá pra "saber" se o bucket mudou sem recalcular a distância -- mas só a
// distância DEPOIS do lance precisa ser recalculada, porque a de ANTES já
// está em cache em acc.ownDistBucket/acc.oppDistBucket desde a última
// chamada (ou desde buildAccumulator). Isso mantém o custo em 1 BFS por
// jogador afetado (nunca 2), o que é essencial pro update de muro não
// ficar mais caro que um recompute completo (ver comentário no struct
// Accumulator e benchAccumulatorUpdate em main.cpp).
//
// Regra usada abaixo (consequência de shortestPathLen não depender da
// posição do peão adversário, só dos muros): um lance de PEÃO só pode
// mudar a distância de quem se moveu -- a distância do outro jogador é
// função só dos muros, que não mudam (1 BFS). Um lance de MURO pode mudar
// a distância dos DOIS jogadores (o muro pode alongar o caminho de
// qualquer um), então nesse caso recalculamos as duas depois do lance
// (2 BFS, contra as 4 que uma versão ingênua sem cache pagaria).
//
// `before` é o estado ANTES do lance (mesmo estado passado a
// legalMoves/applyMove pra gerar `m`); a função computa o estado depois
// internamente. Precisa ser chamada nas DUAS perspectivas (um Accumulator
// por lado), com viewerIsMover indicando se o acumulador é o de quem tem
// o lance em `before`. Pré-condição: acc.ownDistBucket/oppDistBucket
// precisam refletir corretamente `before` (garantido se todo Accumulator
// nasce de buildAccumulator e só é mutado por esta função).
inline void updateAccumulatorForMove(Accumulator& acc, bool viewerIsMover, const State& before, const Move& m,
                                      PlayerPathCacheTable* xtable = nullptr) {
    State after = applyMove(before, m);
    int mover = before.turn, opp = 1 - mover;
    if (!m.isWall) {
        int destCell = m.a;
        int moverCell = before.pawn[mover];
        int newBucket = distBucket(distLenCached(after.wallsH, after.wallsV, destCell, mover, xtable));
        int viewerPlayer = viewerIsMover ? mover : opp;
        if (viewerIsMover) {
            acc.removeFeature(featOwnPawn(moverCell, viewerPlayer));
            acc.addFeature(featOwnPawn(destCell, viewerPlayer));
            if (acc.ownDistBucket != newBucket) {
                acc.removeFeature(featOwnDist(acc.ownDistBucket));
                acc.addFeature(featOwnDist(newBucket));
                acc.ownDistBucket = newBucket;
            }
            // acc.oppDistBucket não muda: lance de peão não altera muro nenhum.
        } else {
            acc.removeFeature(featOppPawn(moverCell, viewerPlayer));
            acc.addFeature(featOppPawn(destCell, viewerPlayer));
            if (acc.oppDistBucket != newBucket) {
                acc.removeFeature(featOppDist(acc.oppDistBucket));
                acc.addFeature(featOppDist(newBucket));
                acc.oppDistBucket = newBucket;
            }
        }
        // lance de peão não muda wallsLeft de ninguém -- nada a atualizar aqui.
    } else {
        int slot = slotIdx(m.b, m.c);
        int viewerPlayer = viewerIsMover ? mover : opp;
        int otherPlayer = 1 - viewerPlayer;
        acc.addFeature(m.a == 0 ? featWallH(slot, viewerPlayer) : featWallV(slot, viewerPlayer));

        int newOwn = distBucket(distLenCached(after.wallsH, after.wallsV, after.pawn[viewerPlayer], viewerPlayer, xtable));
        if (acc.ownDistBucket != newOwn) {
            acc.removeFeature(featOwnDist(acc.ownDistBucket));
            acc.addFeature(featOwnDist(newOwn));
            acc.ownDistBucket = newOwn;
        }
        int newOpp = distBucket(distLenCached(after.wallsH, after.wallsV, after.pawn[otherPlayer], otherPlayer, xtable));
        if (acc.oppDistBucket != newOpp) {
            acc.removeFeature(featOppDist(acc.oppDistBucket));
            acc.addFeature(featOppDist(newOpp));
            acc.oppDistBucket = newOpp;
        }
        // muros restantes: SÓ quem jogou o lance perde 1 muro (regra em
        // rules.hpp: `ns.wallsLeft[player] -= 1`, nunca os dois). `mover`
        // é sempre quem jogou -- independente de quem seja o viewer.
        if (mover == viewerPlayer) {
            int newBucket = wallsLeftBucket(after.wallsLeft[viewerPlayer]);
            if (acc.ownWallsLeftBucket != newBucket) {
                acc.removeFeature(featOwnWallsLeft(acc.ownWallsLeftBucket));
                acc.addFeature(featOwnWallsLeft(newBucket));
                acc.ownWallsLeftBucket = newBucket;
            }
        } else {
            int newBucket = wallsLeftBucket(after.wallsLeft[mover]);
            if (acc.oppWallsLeftBucket != newBucket) {
                acc.removeFeature(featOppWallsLeft(acc.oppWallsLeftBucket));
                acc.addFeature(featOppWallsLeft(newBucket));
                acc.oppWallsLeftBucket = newBucket;
            }
        }
    }
}

// =========================================================================
// Quantização int8 -- pipeline estilo NNUE (Stockfish/Zchezz): camada do
// acumulador em int16 (escala QA), camadas de cabeça (value1/value2 das
// DUAS cabeças, e policy) em int8 (escala QB), acumulação intermediária em
// int32. Ver Seção 7.8 do plano para a derivação completa das escalas.
//
// MUDANÇA DESTA SESSÃO: QA e QB eram calculados DEPOIS do treino (QA fixo
// em 255, QB dinâmico a partir do maior peso encontrado). Isso virou
// quantization-aware training (QAT, nos moldes do nnue-pytorch do
// Stockfish): QA e QB agora são decididos ANTES do treino (constantes
// fixas, ver QA_DEFAULT/QB_DEFAULT abaixo) e um WeightClipper (ver
// train_nnue.py/train_nnue_numpy.py) trava os pesos dentro do range
// representável em int8/int16 A CADA PASSO do otimizador -- não só
// arredonda no final. quantize_nnue.py continua existindo (converte o
// .bin float32 final pro layout int8/int16 abaixo), mas não computa mais
// QB a partir do maior peso: ele recebe QA/QB já fixos e só arredonda
// pesos que, graças ao clipper, já deveriam caber no range (o aviso de
// saturação continua ali como rede de segurança para pesos vindos de um
// treino sem clipper).
//
// Camada 1 (acumulador): w1_i16 = round(w1_f32 * QA), b1_i16 = round(b1_f32 * QA).
//   acc_i32[h] = b1_i16[h] + soma incremental de w1_i16[feat][h]
//   (guardado em int32 para não arriscar overflow de int16 mesmo que o
//   treino produza pesos maiores no futuro; a MATRIZ de pesos em si é
//   int16, que é onde a quantização realmente importa para tamanho/banda).
//
// SCReLU inteira: clamp(acc_i32, 0, QA)^2 / QA -> inteiro em [0, QA],
// guardado em uint8. Isso é exatamente QA * screlu(acc_f32) a menos do
// arredondamento da divisão inteira.
//
// Camadas de cabeça (value1/value2 de CADA cabeça -- wl e aux -- mais
// policy 256->209): pesos e bias quantizados com uma ÚNICA escala QB fixa,
// compartilhada pelas cinco matrizes (wv1_wl, wv2_wl, wv1_aux, wv2_aux,
// wp). bv1/bv2/bp são quantizados em int32 com a escala combinada correta
// (QA*QB para bv1_wl/bv1_aux/bp, que somam diretamente ao produto
// ativação-uint8 x peso-int8; QA*QB*QB para bv2_wl/bv2_aux, que somam ao
// produto hj(escala QA*QB) x wv2(escala QB)).
constexpr int32_t QA_DEFAULT = 255;
// QB fixo (QAT): escolhido para caber com folga o range de pesos que o
// WeightClipper impõe durante o treino (|w| <= 127/QB_DEFAULT ~= 1.98).
// Antes da QAT desta sessão, QB era dinâmico e variava por treino (ex.
// QB=42 medido num treino anterior); agora é uma constante de compilação
// e de treino compartilhada -- se for alterado aqui, o clipper em
// train_nnue.py/train_nnue_numpy.py e o default de quantize_nnue.py
// precisam mudar junto (os três lados assumem o mesmo valor).
constexpr int32_t QB_DEFAULT = 64;

struct NNUEWeightsQuant {
    int32_t QA = QA_DEFAULT;
    int32_t QB = QB_DEFAULT;

    // true assim que loadFromFile termina com sucesso; falso no estado
    // recém-construído (pesos zerados, ver construtor abaixo) e falso de
    // novo se um loadFromFile subsequente falhar no meio da leitura (não
    // dá pra garantir que os vetores ficaram num estado consistente).
    // Consultável via nnueWeightsLoaded() -- usado pelos call-sites
    // (selfplay/arena/wasm) para decidir se é seguro ligar EvalMode::NNUE
    // ou se devem cair para o heurístico.
    bool loaded = false;

    std::vector<std::array<int16_t, HIDDEN>> w1;  // [NUM_FEATURES][HIDDEN], escala QA
    std::array<int16_t, HIDDEN> b1{};              // escala QA

    // cabeça de RESULTADO (WL)
    std::array<std::array<int8_t, 32>, HIDDEN> wv1_wl{}; // escala QB
    std::array<int32_t, 32> bv1_wl{};                      // escala QA*QB
    std::array<int8_t, 32> wv2_wl{};                       // escala QB
    int32_t bv2_wl = 0;                                    // escala QA*QB*QB

    // cabeça AUXILIAR (imitação da heurística)
    std::array<std::array<int8_t, 32>, HIDDEN> wv1_aux{}; // escala QB
    std::array<int32_t, 32> bv1_aux{};                      // escala QA*QB
    std::array<int8_t, 32> wv2_aux{};                       // escala QB
    int32_t bv2_aux = 0;                                    // escala QA*QB*QB

    std::vector<std::array<int8_t, HIDDEN>> wp;   // [POLICY_OUT][HIDDEN], escala QB
    std::vector<int32_t> bp;                       // escala QA*QB

    // CORREÇÃO: antes deste construtor, w1/wp nasciam como std::vector
    // vazio (tamanho 0) -- só ganhavam tamanho dentro de loadFromFile.
    // Enquanto todo call-site só ligava setEvalMode(NNUE) depois de
    // confirmar loadFromFile()==true, isso nunca mordia. Mas com NNUE
    // virando o default dos binários, existe agora um caminho onde
    // addFeature()/removeFeature() são chamados (via buildAccumulatorQuant)
    // ANTES ou SEM um load bem-sucedido -- w1[featIdx] num vetor vazio é
    // acesso fora dos limites (UB / corrupção de heap silenciosa), não um
    // valor "0" seguro. Pré-alocar tudo zerado aqui torna o estado
    // recém-construído seguro de usar (eval neutra, sempre 0) em vez de UB.
    NNUEWeightsQuant() {
        w1.assign(NUM_FEATURES, {});
        wp.assign(POLICY_OUT, {});
        bp.assign(POLICY_OUT, 0);
    }

    // Layout do arquivo: cabeçalho [QA:int32][QB:int32] (formato do
    // cabeçalho NÃO mudou com a QAT -- QA/QB continuam sendo lidos do
    // próprio arquivo, só que agora são sempre as mesmas duas constantes
    // fixas gravadas por quantize_nnue.py, em vez de um QB calculado por
    // arquivo), depois os blocos acima na mesma ordem de campos, sem
    // padding. Gerado por training/quantize_nnue.py a partir de um .bin
    // float32 já treinado (ver NNUEWeights::loadFromFile) -- nunca escrito
    // à mão em C++.
    bool loadFromFile(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;

        // Checagem explícita de tamanho ANTES de ler -- não confiar só no
        // fread() ir falhando por EOF no meio do caminho pra pegar um
        // arquivo de arquitetura diferente (ex.: pesos do layout antigo,
        // NUM_FEATURES=332, contra este binário compilado com
        // NUM_FEATURES=354, 2026-08). Descoberto porque `fread`-e-deixar-
        // falhar-no-meio É frágil por coincidência: se o total de bytes
        // dos dois formatos ficasse parecido o suficiente, um arquivo do
        // layout errado poderia ler campos inteiros com sucesso e só
        // desalinhar (sem nunca retornar false) -- mesma classe de bug já
        // encontrada e corrigida em quantize_nnue.py nesta sessão (lá era
        // uma checagem tautológica; aqui nem havia checagem de tamanho
        // nenhuma, só o encadeamento de `ok = ok && fread(...)`).
        std::fseek(f, 0, SEEK_END);
        long actualBytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        long expectedBytes =
            (long)sizeof(int32_t) * 2                                  // QA, QB
            + (long)NUM_FEATURES * HIDDEN * sizeof(int16_t)            // w1
            + (long)HIDDEN * sizeof(int16_t)                           // b1
            + 2 * ((long)HIDDEN * 32 * sizeof(int8_t)                  // wv1_wl/wv1_aux
                   + 32 * sizeof(int32_t)                              // bv1_wl/bv1_aux
                   + 32 * sizeof(int8_t)                               // wv2_wl/wv2_aux
                   + sizeof(int32_t))                                  // bv2_wl/bv2_aux
            + (long)POLICY_OUT * HIDDEN * sizeof(int8_t)               // wp
            + (long)POLICY_OUT * sizeof(int32_t);                      // bp
        if (actualBytes != expectedBytes) {
            std::fclose(f);
            std::fprintf(stderr,
                "[nnue] '%s' tem %ld bytes, esperado %ld para NUM_FEATURES=%d "
                "(arquivo de arquitetura antiga/diferente? precisa "
                "retreinar/re-quantizar) -- carregamento recusado.\n",
                path.c_str(), actualBytes, expectedBytes, NUM_FEATURES);
            loaded = false;
            return false;
        }

        bool ok = true;
        ok = ok && std::fread(&QA, sizeof(int32_t), 1, f) == 1;
        ok = ok && std::fread(&QB, sizeof(int32_t), 1, f) == 1;

        w1.assign(NUM_FEATURES, {});
        for (auto& row : w1) ok = ok && std::fread(row.data(), sizeof(int16_t), HIDDEN, f) == (size_t)HIDDEN;
        ok = ok && std::fread(b1.data(), sizeof(int16_t), HIDDEN, f) == (size_t)HIDDEN;

        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(int8_t), 32, f) == 32;
        ok = ok && std::fread(bv1_wl.data(), sizeof(int32_t), 32, f) == 32;
        ok = ok && std::fread(wv2_wl.data(), sizeof(int8_t), 32, f) == 32;
        ok = ok && std::fread(&bv2_wl, sizeof(int32_t), 1, f) == 1;

        for (auto& row : wv1_aux) ok = ok && std::fread(row.data(), sizeof(int8_t), 32, f) == 32;
        ok = ok && std::fread(bv1_aux.data(), sizeof(int32_t), 32, f) == 32;
        ok = ok && std::fread(wv2_aux.data(), sizeof(int8_t), 32, f) == 32;
        ok = ok && std::fread(&bv2_aux, sizeof(int32_t), 1, f) == 1;

        wp.assign(POLICY_OUT, {});
        for (auto& row : wp) ok = ok && std::fread(row.data(), sizeof(int8_t), HIDDEN, f) == (size_t)HIDDEN;
        bp.assign(POLICY_OUT, 0);
        ok = ok && std::fread(bp.data(), sizeof(int32_t), POLICY_OUT, f) == (size_t)POLICY_OUT;

        std::fclose(f);
        // Se a leitura falhou no meio (arquivo truncado/corrompido), os
        // vetores podem estar parcialmente preenchidos com lixo do arquivo
        // anterior -- não é seguro chamar isso de "carregado". `loaded`
        // fica false nesse caso, e o singleton continua com os pesos
        // zerados do construtor (ou os do último load bem-sucedido, se
        // havia um antes desta tentativa).
        loaded = ok;
        return ok;
    }
};

inline NNUEWeightsQuant& weightsQuant() { static NNUEWeightsQuant w; return w; }
inline bool loadWeightsQuant(const std::string& path) { return weightsQuant().loadFromFile(path); }

// true se os pesos do singleton vieram de um loadFromFile bem-sucedido
// (em vez do estado zerado do construtor). Usado pelos call-sites
// (selfplay/arena/wasm) para decidir se é seguro ligar
// Negamax::EvalMode::NNUE ou se devem cair para o heurístico.
inline bool nnueWeightsLoaded() { return weightsQuant().loaded; }

// Caminho padrão dos pesos quantizados, relativo à raiz do repositório
// (mesma convenção usada em readme.md/training/quantize_nnue.py e no
// nome do arquivo publicado por training/train_nnue.py). Usado por
// selfplay_main.cpp/arena.cpp/engine_wasm.cpp como default quando o
// caminho não é passado explicitamente -- NNUE é o default dos
// binários; este é o arquivo que eles tentam carregar automaticamente
// antes de decidir cair para o heurístico.
inline std::string defaultNnueWeightsPath() { return "data/nnue/nnue_weights_int8.bin"; }

struct AccumulatorQuant {
    std::array<int32_t, HIDDEN> v{};
    int ownDistBucket = 0;   // cache do bucket ativo -- mesma razão de ser do campo em Accumulator (float)
    int oppDistBucket = 0;
    int ownWallsLeftBucket = 0;
    int oppWallsLeftBucket = 0;

    void addFeature(int featIdx) {
        auto& row = weightsQuant().w1[featIdx];
        for (int i = 0; i < HIDDEN; i++) v[i] += row[i];
    }
    void removeFeature(int featIdx) {
        auto& row = weightsQuant().w1[featIdx];
        for (int i = 0; i < HIDDEN; i++) v[i] -= row[i];
    }
};

inline AccumulatorQuant buildAccumulatorQuant(const State& s, int perspective, PlayerPathCacheTable* xtable = nullptr) {
    AccumulatorQuant acc;
    auto& b1 = weightsQuant().b1;
    for (int i = 0; i < HIDDEN; i++) acc.v[i] = b1[i];
    int me = perspective, opp = 1 - perspective;
    acc.addFeature(featOwnPawn(s.pawn[me], perspective));
    acc.addFeature(featOppPawn(s.pawn[opp], perspective));
    for (int i = 0; i < WS * WS; i++) {
        if ((s.wallsH >> i) & 1ull) acc.addFeature(featWallH(i, perspective));
        if ((s.wallsV >> i) & 1ull) acc.addFeature(featWallV(i, perspective));
    }
    acc.ownDistBucket = distBucket(distLenCached(s.wallsH, s.wallsV, s.pawn[me], me, xtable));
    acc.oppDistBucket = distBucket(distLenCached(s.wallsH, s.wallsV, s.pawn[opp], opp, xtable));
    acc.addFeature(featOwnDist(acc.ownDistBucket));
    acc.addFeature(featOppDist(acc.oppDistBucket));
    acc.ownWallsLeftBucket = wallsLeftBucket(s.wallsLeft[me]);
    acc.oppWallsLeftBucket = wallsLeftBucket(s.wallsLeft[opp]);
    acc.addFeature(featOwnWallsLeft(acc.ownWallsLeftBucket));
    acc.addFeature(featOppWallsLeft(acc.oppWallsLeftBucket));
    return acc;
}

// mesma lógica incremental de updateAccumulatorForMove (ver comentário
// completo lá, incluindo o cache ownDistBucket/oppDistBucket que evita
// recalcular a distância ANTES do lance), só que sobre o acumulador
// quantizado -- mantida como função separada (em vez de template único)
// pra não esconder o tipo int32_t/float por trás de deducao automática
// nos pontos de chamada da busca.
inline void updateAccumulatorForMoveQuant(AccumulatorQuant& acc, bool viewerIsMover, const State& before, const Move& m,
                                           PlayerPathCacheTable* xtable = nullptr) {
    State after = applyMove(before, m);
    int mover = before.turn, opp = 1 - mover;
    if (!m.isWall) {
        int destCell = m.a;
        int moverCell = before.pawn[mover];
        int newBucket = distBucket(distLenCached(after.wallsH, after.wallsV, destCell, mover, xtable));
        int viewerPlayer = viewerIsMover ? mover : opp;
        if (viewerIsMover) {
            acc.removeFeature(featOwnPawn(moverCell, viewerPlayer));
            acc.addFeature(featOwnPawn(destCell, viewerPlayer));
            if (acc.ownDistBucket != newBucket) {
                acc.removeFeature(featOwnDist(acc.ownDistBucket));
                acc.addFeature(featOwnDist(newBucket));
                acc.ownDistBucket = newBucket;
            }
        } else {
            acc.removeFeature(featOppPawn(moverCell, viewerPlayer));
            acc.addFeature(featOppPawn(destCell, viewerPlayer));
            if (acc.oppDistBucket != newBucket) {
                acc.removeFeature(featOppDist(acc.oppDistBucket));
                acc.addFeature(featOppDist(newBucket));
                acc.oppDistBucket = newBucket;
            }
        }
        // lance de peão não muda wallsLeft de ninguém -- nada a atualizar aqui.
    } else {
        int slot = slotIdx(m.b, m.c);
        int viewerPlayer = viewerIsMover ? mover : opp;
        int otherPlayer = 1 - viewerPlayer;
        acc.addFeature(m.a == 0 ? featWallH(slot, viewerPlayer) : featWallV(slot, viewerPlayer));

        int newOwn = distBucket(distLenCached(after.wallsH, after.wallsV, after.pawn[viewerPlayer], viewerPlayer, xtable));
        if (acc.ownDistBucket != newOwn) {
            acc.removeFeature(featOwnDist(acc.ownDistBucket));
            acc.addFeature(featOwnDist(newOwn));
            acc.ownDistBucket = newOwn;
        }
        int newOpp = distBucket(distLenCached(after.wallsH, after.wallsV, after.pawn[otherPlayer], otherPlayer, xtable));
        if (acc.oppDistBucket != newOpp) {
            acc.removeFeature(featOppDist(acc.oppDistBucket));
            acc.addFeature(featOppDist(newOpp));
            acc.oppDistBucket = newOpp;
        }
        // muros restantes: SÓ quem jogou perde 1 muro -- ver comentário
        // equivalente em updateAccumulatorForMove (float).
        if (mover == viewerPlayer) {
            int newWlBucket = wallsLeftBucket(after.wallsLeft[viewerPlayer]);
            if (acc.ownWallsLeftBucket != newWlBucket) {
                acc.removeFeature(featOwnWallsLeft(acc.ownWallsLeftBucket));
                acc.addFeature(featOwnWallsLeft(newWlBucket));
                acc.ownWallsLeftBucket = newWlBucket;
            }
        } else {
            int newWlBucket = wallsLeftBucket(after.wallsLeft[mover]);
            if (acc.oppWallsLeftBucket != newWlBucket) {
                acc.removeFeature(featOppWallsLeft(acc.oppWallsLeftBucket));
                acc.addFeature(featOppWallsLeft(newWlBucket));
                acc.oppWallsLeftBucket = newWlBucket;
            }
        }
    }
}

// SCReLU inteira: clamp(x,0,QA)^2 / QA, resultado em [0,QA] -> cabe em uint8
// pra QA <= 255 (QA_DEFAULT). Divisão trunca em direção a zero, mas a
// entrada do quadrado já é não-negativa aqui, então trunc==floor neste caso
// específico (a distinção só importa nas divisões finais de dequantização,
// que podem ser negativas -- ver forwardValueQuant/forwardPolicyQuant).
inline uint8_t screluQuant(int32_t x, int32_t QA) {
    int32_t c = x < 0 ? 0 : (x > QA ? QA : x);
    int64_t sq = (int64_t)c * (int64_t)c;
    return (uint8_t)(sq / QA);
}

// Núcleo comum às duas cabeças quantizadas: só muda qual par (wv1,bv1,wv2,
// bv2) é usado. Mantido como função livre (não template) pelos mesmos
// motivos do par forwardValue*/forwardValue*Quant já discutidos no
// restante do arquivo -- tipos explícitos nos pontos de chamada.
inline float forwardValueHeadQuant(const AccumulatorQuant& acc,
                                    const std::array<std::array<int8_t, 32>, HIDDEN>& wv1,
                                    const std::array<int32_t, 32>& bv1,
                                    const std::array<int8_t, 32>& wv2,
                                    int32_t bv2) {
    auto& W = weightsQuant();
    std::array<uint8_t, HIDDEN> a{};
    for (int i = 0; i < HIDDEN; i++) a[i] = screluQuant(acc.v[i], W.QA);

    // value1 (256->32): escala QA*QB
    std::array<int32_t, 32> h{};
    for (int i = 0; i < HIDDEN; i++) {
        int32_t ai = a[i];
        if (ai == 0) continue;
        for (int j = 0; j < 32; j++) h[j] += ai * (int32_t)wv1[i][j];
    }
    // clippedRelu inteira: clamp(h+bv1, 0, QA*QB) -- mesma escala combinada
    int64_t QAQB = (int64_t)W.QA * (int64_t)W.QB;
    std::array<int32_t, 32> hj{};
    for (int j = 0; j < 32; j++) {
        int64_t hv = (int64_t)h[j] + (int64_t)bv1[j];
        if (hv < 0) hv = 0;
        if (hv > QAQB) hv = QAQB;
        hj[j] = (int32_t)hv;
    }
    // value2 (32->1): hj (escala QA*QB) x wv2 (escala QB) -> escala QA*QB*QB
    int64_t out = bv2;
    for (int j = 0; j < 32; j++) out += (int64_t)hj[j] * (int64_t)wv2[j];
    int64_t denom = QAQB * (int64_t)W.QB;
    // Des-escala final: divisão em PONTO FLUTUANTE, não inteira. Só a
    // divisão da SCReLU (não-negativa, acima) precisa ser inteira de
    // verdade -- é ela que fecha o loop de ida-e-volta pro domínio uint8
    // usado no próximo produto interno. Esta aqui é só a conversão do
    // resultado final pra um score comparável; truncar pra inteiro nesse
    // ponto jogaria fora toda a parte fracionária do valor (erro medido
    // de ~1 unidade em vez de ~0,01-0,03 -- bug pego na verificação de
    // paridade da sessão anterior, ver Seção 7.8 do plano).
    return (float)((double)out / (double)denom);
}

// forwardValueWL é a que a busca vai usar (Fase 6); forwardValueAuxQuant
// existe só para treino/validação, espelhando forwardValueAux (float).
inline float forwardValueWLQuant(const AccumulatorQuant& acc) {
    auto& W = weightsQuant();
    return forwardValueHeadQuant(acc, W.wv1_wl, W.bv1_wl, W.wv2_wl, W.bv2_wl);
}

inline float forwardValueAuxQuant(const AccumulatorQuant& acc) {
    auto& W = weightsQuant();
    return forwardValueHeadQuant(acc, W.wv1_aux, W.bv1_aux, W.wv2_aux, W.bv2_aux);
}

inline void forwardPolicyQuant(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out) {
    auto& W = weightsQuant();
    std::array<uint8_t, HIDDEN> a{};
    for (int i = 0; i < HIDDEN; i++) a[i] = screluQuant(acc.v[i], W.QA);

    int64_t QAQB = (int64_t)W.QA * (int64_t)W.QB;
    for (int o = 0; o < POLICY_OUT; o++) {
        int64_t s = W.bp[o];
        auto& row = W.wp[o];
        for (int i = 0; i < HIDDEN; i++) {
            int32_t ai = a[i];
            if (ai == 0) continue;
            s += (int64_t)ai * (int64_t)row[i];
        }
        out[o] = (float)((double)s / (double)QAQB);   // des-escala final em ponto flutuante, ver forwardValueQuant
    }
}

// =========================================================================
// AccPair + nnueEvalInt: helpers usados por search.hpp para manter dois
// acumuladores quantizados (um por perspectiva) na pilha de busca e avaliar
// folhas via NNUE sem recomputar do zero.
//
// NNUE_EVAL_SCALE: fator que mapeia o logit cru da cabeça WL (~[-3,3])
// para a mesma escala inteira que evalSimple usa (~[-600,600] em posições
// normais). Mesmo valor que VALUE_SCALE em train_nnue.py -- essencial para
// que aspiration windows e contempt calibrados para evalSimple continuem
// funcionando sem re-tuning quando a NNUE assume a avaliação de folha.
constexpr int NNUE_EVAL_SCALE = 200;

// Par de acumuladores quantizados -- um por perspectiva de jogador.
// acc[0] = perspectiva do jogador 0 (own=0, opp=1);
// acc[1] = perspectiva do jogador 1 (own=1, opp=0).
// Mantidos por search.hpp como pilha de pares (um por ply da busca),
// atualizados incrementalmente via updateAccumulatorForMoveQuant.
//
// Item 3 (update preguiçoso por perspectiva): a cada lance, search.hpp só
// PRECISA imediatamente da perspectiva de quem vai jogar no filho (é essa
// que nnueEvalInt lê se o filho for folha). A perspectiva de quem acabou
// de jogar (o "mover") não é lida no filho -- só volta a ser lida um ply
// depois, SE a busca chegar lá (nem sempre chega: cutoff de alpha-beta
// pode podar o resto da subárvore antes disso). pending[p] marca que
// acc[p] ainda não recebeu o update do último lance -- os dados pra
// aplicar esse update quando for preciso (pendBefore/pendMove/
// pendViewerIsMover) ficam guardados aqui. Só existe UM nível de posterga
// por perspectiva: toda vez que uma perspectiva pending é usada como base
// pra construir OUTRO AccPair (makeChildAccPair) ou lida (nnueEvalInt),
// ela é resolvida primeiro (ver resolvePending) -- nunca se acumulam dois
// lances pendentes na mesma perspectiva ao mesmo tempo.
//
// INVARIANTE mantida por makeChildAccPair/buildAccPairRoot (não é
// responsabilidade de quem chama garantir isso à mão): em qualquer AccPair
// que search.hpp esteja usando como `curAcc` de um nó com `side = s.turn`,
// acc[side] NUNCA está pending -- foi resolvida na criação do próprio nó
// (é a perspectiva "eager" de quem vai jogar ali). Só acc[1-side] (a de
// quem jogou o lance que levou a este nó) pode estar pending.
struct AccPair {
    AccumulatorQuant acc[2];
    bool pending[2] = {false, false};
    State pendBefore[2]{};
    Move pendMove[2]{Move::pawn(0), Move::pawn(0)};
    bool pendViewerIsMover[2] = {false, false};
};

// Resolve (se necessário) o update adiado da perspectiva `persp`. Idempotente
// -- chamar de novo com pending[persp] já false não faz nada. Deve ser
// chamada antes de: (1) ler acc[persp] para eval, (2) copiar acc[persp]
// como base de outro AccPair.
inline void resolvePending(AccPair& ap, int persp, PlayerPathCacheTable* xtable = nullptr) {
    if (!ap.pending[persp]) return;
    updateAccumulatorForMoveQuant(ap.acc[persp], ap.pendViewerIsMover[persp],
                                   ap.pendBefore[persp], ap.pendMove[persp], xtable);
    ap.pending[persp] = false;
}

// Constrói `child` a partir de `parent`, aplicando o lance `m` (jogado em
// `before`, com before.turn == mover do lance). Implementa o Item 3: só a
// perspectiva de quem vai jogar em `child` (1-mover) é atualizada agora; a
// do `mover` fica pending. `parent` pode ser mutado (resolvePending da sua
// própria perspectiva 1-mover, se estava pending -- precisa estar resolvida
// pra servir de base correta à cópia feita aqui).
inline void makeChildAccPair(AccPair& parent, AccPair& child, const State& before, const Move& m,
                              PlayerPathCacheTable* xtable = nullptr) {
    int mover = before.turn, opp = 1 - mover;
    // Perspectiva de quem joga em `child` -- precisa estar pronta já.
    resolvePending(parent, opp, xtable);
    child.acc[opp] = parent.acc[opp];
    child.pending[opp] = false;
    updateAccumulatorForMoveQuant(child.acc[opp], /*viewerIsMover=*/false, before, m, xtable);
    // Perspectiva de quem jogou -- adia. parent.acc[mover] já está
    // garantidamente resolvida (invariante da struct: é a perspectiva de
    // s.turn no nó de `parent`, sempre eager).
    child.acc[mover] = parent.acc[mover];
    child.pending[mover] = true;
    child.pendBefore[mover] = before;
    child.pendMove[mover] = m;
    child.pendViewerIsMover[mover] = true;
}

// Constrói um AccPair "frio" (recompute do zero nas duas perspectivas,
// sem nada pending) -- uso: raiz de cada busca/iteração de iterative
// deepening. CORREÇÃO: antes desta função, os 5 call-sites que
// reconstroem a raiz escreviam direto em nnueAccStack[0].acc[0]/acc[1] sem
// zerar pending[] -- como nnueAccStack é reaproveitado entre buscas, flags
// pending de uma busca anterior (junto com pendBefore/pendMove igualmente
// obsoletos) podiam sobrar no slot raiz e causar uma resolução espúria com
// dados de outra posição.
inline AccPair buildAccPairRoot(const State& s, PlayerPathCacheTable* xtable = nullptr) {
    AccPair ap;
    ap.acc[0] = buildAccumulatorQuant(s, 0, xtable);
    ap.acc[1] = buildAccumulatorQuant(s, 1, xtable);
    ap.pending[0] = false;
    ap.pending[1] = false;
    return ap;
}

// Avaliação NNUE do ponto de vista de `side` (quem vai jogar), em
// unidades inteiras (escala NNUE_EVAL_SCALE) -- compatível com o valor de
// retorno de evalSimple/evalSimpleW. Usa a cabeça quantizada de resultado
// (WL); a cabeça auxiliar (imitação de evalSimple) nunca é chamada pela
// busca.
inline int nnueEvalInt(const AccPair& ap, int side) {
    float logit = forwardValueWLQuant(ap.acc[side]);
    return (int)std::lround(logit * (float)NNUE_EVAL_SCALE);
}

} // namespace qr
