// dsu.hpp -- Union-Find (DSU) com rollback sobre o grafo dual de muros
// (Fase 4.2.1 do plano, item pendente). Técnica clássica de "barreira
// esquerda-direita": mesmo princípio usado para provar o teorema do Hex
// (dual planar do grafo de células -- uma cadeia de pedras conectando os
// dois lados "próprios" no dual é equivalente, por dualidade planar, a
// um corte separando os outros dois lados no primal).
//
// --- Por que barreira esquerda-direita, e não conectividade peão-meta ---
// As metas dos dois jogadores são FILEIRAS inteiras (linha 0 e linha
// N-1). Separar uma célula de uma fileira inteira do tabuleiro (que é um
// retângulo simplesmente conexo) só é possível, usando apenas muros
// internos (nunca a borda), com uma barreira que atravesse o tabuleiro
// de ponta a ponta conectando a borda ESQUERDA à borda DIREITA -- uma
// barreira que conectasse cima-a-baixo separaria esquerda/direita, não
// cima/baixo, e por isso jamais bloqueia uma meta-fileira (cada fileira
// tem células dos dois lados de uma barreira vertical). Esse é o mesmo
// argumento de dualidade planar do teorema do Hex.
//
// --- Geometria exata (derivada de rules.hpp: edgeBlocked/wallSlotAvailable) ---
// Um muro H(r,c) ocupa o segmento horizontal na linha de fronteira r+1,
// colunas [c, c+2] (dois "postes" de largura). Um muro V(r,c) ocupa o
// segmento vertical na coluna de fronteira c+1, linhas [r, r+2]. Dois
// segmentos se TOCAM (formam barreira contínua, sem brecha) sse
// compartilham pelo menos um ponto:
//   - mesma orientação, mesma linha/coluna, colineares ponta-a-ponta:
//     H(r,c) & H(r,c±2); V(r,c) & V(r±2,c). (distância 1 é sempre
//     ilegal por sobreposição -- ver wallSlotAvailable -- então só ±2
//     importa.)
//   - orientações cruzadas H(r,c) & V(r',c'): tocam sse
//     |r-r'| <= 1 E |c-c'| <= 1 (distância de Chebyshev <= 1 no grid de
//     slots), excluindo o mesmo poste exato (mutuamente exclusivo pela
//     regra de colocação). Prova: H(r,c) cobre pontos (r+1, y) para
//     y em [c,c+2]; V(r',c') cobre pontos (x, c'+1) para x em
//     [r',r'+2]. Interseção não-vazia sse c'+1 em [c,c+2] (isto é,
//     c' em {c-1,c,c+1}) E r+1 em [r',r'+2] (isto é, r' em
//     {r-1,r,r+1}).
//   - H(r,c) toca a borda ESQUERDA sse c==0; toca a borda DIREITA sse
//     c==WS-1. V(r,c) toca a borda SUPERIOR sse r==0; toca a borda
//     INFERIOR sse r==WS-1. (H nunca toca cima/baixo, V nunca toca
//     esquerda/direita -- cada orientação só alcança as 2 bordas
//     perpendiculares à direção em que ela se estende.)
//
// --- Quatro condições, não uma: por que só ESQUERDA-DIREITA não basta ---
// A primeira versão deste arquivo só modelava as bordas esquerda/
// direita e usava "ESQUERDA conectou com DIREITA" como único sinal de
// perigo. Isso é necessário mas NÃO suficiente: um peão pode ficar preso
// num "bolso de canto" -- cercado por duas bordas PERPENDICULARES (por
// exemplo SUPERIOR e DIREITA) -- sem que a barreira precise atravessar
// o tabuleiro de ponta a ponta. Caso comum logo no início da partida,
// já que os peões começam colados às fileiras 0 e 8: um muro perto de
// um canto, junto com um segundo muro formando o outro lado do bolso,
// tranca o peão contra duas bordas ao mesmo tempo.
// `testWallDsuRegression` pegou exatamente esse caso (peão em (0,5),
// bolso fechado contra as bordas SUPERIOR e DIREITA em (0,7)-(0,8)) com
// uma divergência real em jogo aleatório -- ver histórico do commit.
//
// A generalização correta tem duas partes:
//
// 1) Qualquer FECHAMENTO DE CICLO no grafo dual planar dos muros
//    corresponde a uma curva fechada real no tabuleiro (o grafo está
//    embutido no plano exatamente nas posições físicas dos muros), que
//    cerca alguma região não-vazia -- podendo conter um peão e/ou
//    excluir a fileira-meta de dentro dela, não importa se o ciclo toca
//    alguma borda ou é inteiramente interno. Em union-find, uma união
//    que liga dois nós JÁ no mesmo componente é exatamente essa situação
//    (fecha um ciclo).
//
// 2) Além de ciclos, existe uma segunda família de cortes que NÃO fecha
//    ciclo nenhum: conectar duas bordas DIFERENTES pela primeira vez.
//    ESQUERDA-DIREITA é uma delas (corta cima/baixo -- o caso
//    "clássico"). Mas os 4 cantos também contam: SUPERIOR-ESQUERDA,
//    SUPERIOR-DIREITA, INFERIOR-ESQUERDA e INFERIOR-DIREITA cada um
//    fecha um bolso de canto que exclui a fileira-meta de um dos dois
//    jogadores (SUPERIOR-* ameaça o jogador0, meta=INFERIOR; INFERIOR-*
//    ameaça o jogador1, meta=SUPERIOR). SUPERIOR-INFERIOR é a ÚNICA
//    combinação de bordas que fica de fora: ela separa esquerda/direita
//    do tabuleiro, não cima/baixo, e como as metas são fileiras
//    inteiras (não colunas), isso nunca bloqueia ninguém sozinho.
//
// A condição completa e segura para "não precisa de BFS, é PROVADO
// legal" é a NEGAÇÃO de:
//   (a) alguma união causada pelo candidato é redundante (ciclo), OU
//   (b) ESQUERDA e DIREITA acabam no mesmo componente, OU
//   (c) qualquer par de bordas PERPENDICULARES (SUPERIOR/INFERIOR ×
//       ESQUERDA/DIREITA) acaba no mesmo componente.
// Todas conservadoras (nunca dizem "legal" quando não é). Quando NENHUMA
// ocorre, é matematicamente garantido que a colocação não desconecta
// ninguém -- ver wallCandidateAmbiguous. O pior caso (candidato
// ambíguo) cai de volta no hasPathToGoal exato de rules.hpp, idêntico
// ao comportamento antes desta otimização -- sem risco de regressão por
// construção.
#pragma once
#include <array>
#include <cassert>
#include <cstdint>

namespace qr {

// União por tamanho, SEM compressão de caminho -- necessário para que o
// rollback (desfazer uniões em ordem LIFO) seja correto e barato: com
// compressão de caminho, desfazer uma união não basta para restaurar o
// estado anterior exato (o path compression pode ter alterado parents
// de nós que não foram tocados pela união sendo desfeita).
struct RollbackDSU {
    std::array<int, 132> parent;
    std::array<int, 132> sz;
    // child == -1 marca uma união que já era no-op/redundante (a e b já
    // no mesmo componente); caso contrário, `child` é o nó que era raiz
    // da sua própria árvore antes desta união e teve parent[child]
    // reatribuído (desfazer = parent[child]=child, sz[novo pai] -=
    // sz[child]).
    struct UndoEntry { int child; };
    // Capacidade calculada com folga: buildWallDSU processa até 20
    // muros já colocados (10 por jogador), cada um gerando até 12
    // chamadas a unite() (2 de borda + 2 colineares + até 8 cruzadas);
    // pior caso ~240. wallCandidateAmbiguous soma mais até 12 por cima
    // antes do rollback. 512 dá folga generosa sobre o pior caso teórico
    // (~252) -- um array pequeno demais aqui é overflow silencioso
    // (UB), não um erro visível, então a folga importa mais que o
    // desperdício de alguns bytes de stack.
    std::array<UndoEntry, 512> history;
    int histLen = 0;

    void init(int n) {
        for (int i = 0; i < n; i++) { parent[i] = i; sz[i] = 1; }
        histLen = 0;
    }

    int find(int x) const {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    // Une a e b; sempre grava exatamente uma entrada de histórico (no-op
    // incluído), pra que snapshot()/rollbackTo() contem operações 1:1.
    // Retorna true sse a união era REDUNDANTE (a e b já no mesmo
    // componente antes desta chamada -- fecha um ciclo).
    bool unite(int a, int b) {
        assert(histLen < (int)history.size() && "RollbackDSU::history overflow -- aumente a capacidade");
        int ra = find(a), rb = find(b);
        if (ra == rb) {
            history[histLen++] = {-1};
            return true;
        }
        if (sz[ra] < sz[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        sz[ra] += sz[rb];
        history[histLen++] = {rb};
        return false;
    }

    int snapshot() const { return histLen; }

    // Desfaz uniões em ordem LIFO até histLen==snap. Correto mesmo sem
    // compressão de caminho: como as uniões são desfeitas na ordem
    // inversa exata em que ocorreram, sz[child] ainda reflete o tamanho
    // do subgrupo no momento da união sendo desfeita (qualquer união
    // posterior que tocou esse subgrupo já foi desfeita primeiro).
    void rollbackTo(int snap) {
        while (histLen > snap) {
            int child = history[--histLen].child;
            if (child == -1) continue;
            int par = parent[child];
            sz[par] -= sz[child];
            parent[child] = child;
        }
    }
};

constexpr int WALL_DSU_H_BASE = 0;
constexpr int WALL_DSU_V_BASE = WS * WS;          // 64
constexpr int WALL_DSU_LEFT = 2 * WS * WS;        // 128
constexpr int WALL_DSU_RIGHT = 2 * WS * WS + 1;   // 129
constexpr int WALL_DSU_TOP = 2 * WS * WS + 2;     // 130
constexpr int WALL_DSU_BOTTOM = 2 * WS * WS + 3;  // 131
constexpr int WALL_DSU_SIZE = 2 * WS * WS + 4;    // 132

inline int wallDsuNodeH(int r, int c) { return WALL_DSU_H_BASE + r * WS + c; }
inline int wallDsuNodeV(int r, int c) { return WALL_DSU_V_BASE + r * WS + c; }

// Une o nó do muro (orientation,r,c) -- que pode ainda não estar
// fisicamente colocado em wallsH/wallsV (uso para candidato hipotético)
// -- com todos os nós de muros VIZINHOS já presentes em wallsH/wallsV
// (incluindo as 4 bordas), conforme a geometria de toque derivada
// acima. `wallsH`/`wallsV` devem refletir o estado ANTES desta
// colocação (o próprio muro sendo unido não deve estar setado neles).
//
// Retorna true se ALGUMA das uniões realizadas era redundante (fecha um
// ciclo -- ver nota de corretude no topo do arquivo).
inline bool unionWallNeighbors(RollbackDSU& dsu, uint64_t wallsH, uint64_t wallsV,
                                int orientation, int r, int c) {
    bool anyRedundant = false;
    auto uniteTrack = [&](int a, int b) { anyRedundant |= dsu.unite(a, b); };
    if (orientation == 0) {
        int me = wallDsuNodeH(r, c);
        if (c == 0) uniteTrack(me, WALL_DSU_LEFT);
        if (c == WS - 1) uniteTrack(me, WALL_DSU_RIGHT);
        if (c - 2 >= 0 && ((wallsH >> slotIdx(r, c - 2)) & 1ull)) uniteTrack(me, wallDsuNodeH(r, c - 2));
        if (c + 2 < WS && ((wallsH >> slotIdx(r, c + 2)) & 1ull)) uniteTrack(me, wallDsuNodeH(r, c + 2));
        for (int dr = -1; dr <= 1; dr++) {
            int rr = r + dr;
            if (rr < 0 || rr >= WS) continue;
            for (int dc = -1; dc <= 1; dc++) {
                int cc = c + dc;
                if (cc < 0 || cc >= WS) continue;
                if ((wallsV >> slotIdx(rr, cc)) & 1ull) uniteTrack(me, wallDsuNodeV(rr, cc));
            }
        }
    } else {
        int me = wallDsuNodeV(r, c);
        if (r == 0) uniteTrack(me, WALL_DSU_TOP);
        if (r == WS - 1) uniteTrack(me, WALL_DSU_BOTTOM);
        if (r - 2 >= 0 && ((wallsV >> slotIdx(r - 2, c)) & 1ull)) uniteTrack(me, wallDsuNodeV(r - 2, c));
        if (r + 2 < WS && ((wallsV >> slotIdx(r + 2, c)) & 1ull)) uniteTrack(me, wallDsuNodeV(r + 2, c));
        for (int dr = -1; dr <= 1; dr++) {
            int rr = r + dr;
            if (rr < 0 || rr >= WS) continue;
            for (int dc = -1; dc <= 1; dc++) {
                int cc = c + dc;
                if (cc < 0 || cc >= WS) continue;
                if ((wallsH >> slotIdx(rr, cc)) & 1ull) uniteTrack(me, wallDsuNodeH(rr, cc));
            }
        }
    }
    return anyRedundant;
}

// Constrói o DSU do zero a partir dos muros já colocados em `s`. Ordem
// de iteração não importa (união é comutativa/associativa) -- cada muro
// já colocado é unido com os vizinhos que enxerga no bitboard completo
// (idempotente: unir um par mais de uma vez é barato e sem efeito; a
// posição atual já é LEGAL por construção, incluindo qualquer ciclo que
// não tenha bloqueado ninguém quando se formou -- a redundância aqui é
// ignorada de propósito, só importa para o CANDIDATO em
// wallCandidateAmbiguous).
inline void buildWallDSU(RollbackDSU& dsu, uint64_t wallsH, uint64_t wallsV) {
    dsu.init(WALL_DSU_SIZE);
    uint64_t h = wallsH;
    while (h) {
        int slot = __builtin_ctzll(h);
        h &= h - 1;
        unionWallNeighbors(dsu, wallsH, wallsV, 0, slot / WS, slot % WS);
    }
    uint64_t v = wallsV;
    while (v) {
        int slot = __builtin_ctzll(v);
        v &= v - 1;
        unionWallNeighbors(dsu, wallsH, wallsV, 1, slot / WS, slot % WS);
    }
}

// Testa se o muro candidato (orientation,r,c) é AMBÍGUO quanto a
// legalidade -- isto é, se o DSU não consegue provar sozinho que ele é
// legal, exigindo queda para o BFS exato (hasPathToGoal). Dado um DSU
// já construído com os muros ATUAIS de `wallsH`/`wallsV` (sem o
// candidato), faz push + teste + rollback -- o `dsu` sai desta chamada
// exatamente como entrou (custo O(alfa), não O(células)).
//
// Ambíguo sse, após as uniões do candidato:
//   (a) alguma união é redundante (fecha um ciclo -- cobre cercados
//       inteiramente internos e "mesma borda duas vezes" que não
//       envolvem um segundo par de bordas diferentes), OU
//   (b) ESQUERDA e DIREITA acabam no mesmo componente (barreira
//       completa corta o tabuleiro em cima/baixo -- caso "clássico"),
//       OU
//   (c) qualquer CANTO se fecha: SUPERIOR com ESQUERDA, SUPERIOR com
//       DIREITA, INFERIOR com ESQUERDA, ou INFERIOR com DIREITA -- um
//       "bolso" de canto que cerca um peão contra duas bordas
//       perpendiculares sem precisar atravessar o tabuleiro inteiro
//       (ver nota de corretude no topo do arquivo; testWallDsuRegression
//       pegou justamente esse caso).
// SUPERIOR-INFERIOR (por si só, sem mais nada) fica de fora de
// propósito: separa esquerda/direita do tabuleiro, não cima/baixo, e as
// metas são fileiras inteiras -- não bloqueia ninguém sozinho (ver nota
// no topo do arquivo).
// Quando NENHUMA das condições ocorre, é provado que nenhum jogador
// fica sem caminho até sua meta com este muro -- legal, sem precisar de
// BFS.
inline bool wallCandidateAmbiguous(RollbackDSU& dsu, uint64_t wallsH, uint64_t wallsV,
                                    int orientation, int r, int c) {
    int snap = dsu.snapshot();
    bool anyRedundant = unionWallNeighbors(dsu, wallsH, wallsV, orientation, r, c);
    int rootLeft = dsu.find(WALL_DSU_LEFT);
    int rootRight = dsu.find(WALL_DSU_RIGHT);
    int rootTop = dsu.find(WALL_DSU_TOP);
    int rootBottom = dsu.find(WALL_DSU_BOTTOM);
    bool cornerOrSpan = (rootLeft == rootRight) ||
                         (rootTop == rootLeft) || (rootTop == rootRight) ||
                         (rootBottom == rootLeft) || (rootBottom == rootRight);
    dsu.rollbackTo(snap);
    return anyRedundant || cornerOrSpan;
}

} // namespace qr
