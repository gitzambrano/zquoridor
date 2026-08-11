// qtp_main.cpp -- interface texto (stdin/stdout, um comando por linha,
// no espírito UCI: um loop que le comandos e responde) para o zquoridor,
// usando o CONJUNTO DE COMANDOS do protocolo referenciado em
// https://github.com/pavlosdais/Quoridor (QTP -- Quoridor Text Protocol,
// especificação completa em http://quoridor.di.uoa.gr/qtp/qtp.html):
// name, known_command, list_commands, quit, boardsize, clear_board,
// walls, playmove, playwall, genmove, undo, winner, showboard.
//
// A NOTAÇÃO DE LANCE usada nos argumentos/respostas é A NOSSA (a mesma da
// GUI web, gui_web/app.js: pawnNotation/wallNotation), não a do QTP
// original (que usa "vertex" + "orientation" como tokens separados p/
// muro). Aqui um lance é sempre UM token só:
//   peão:  <coluna a-i><linha 1-9>            ex: e5
//   muro:  <coluna a-h><linha 1-8><h ou v>     ex: a3h, e5v
// Coluna sempre no referencial FIXO do tabuleiro (nunca espelhado -- ver
// nota detalhada em gui_web/app.js, função pawnNotation/wallNotation).
// playmove/playwall recebem <cor> <token> (2 args, em vez dos 2-3 tokens
// do QTP puro); genmove devolve o token direto (sem "vertex orientation"
// separados).
//
// Mapeamento de cor <-> jogador interno: jogador 0 = "black" (primeiro a
// mover -- State::turn começa em 0, ver rules.hpp::initialState, e o
// README do pavlosdais diz "the black player starts first"), jogador 1 =
// "white". Aceita os atalhos de 1 letra do QTP (b/w) também.
//
// LIMITAÇÃO DELIBERADA vs QTP puro: o QTP permite "playmove"/"playwall"
// fora de ordem (cor repetida não é erro de protocolo). O motor aqui tem
// UM State interno com alternância estrita (State::turn), então
// playmove/playwall/genmove com cor != mover atual respondem "illegal
// move" -- não dá pra representar posições fora da alternância normal do
// jogo sem reescrever o core (rules.hpp) pra isso.
//
// Extensões (fora do QTP original, mas o protocolo permite extensões
// privadas -- ver list_commands): "loadnnue <path>" carrega pesos NNUE
// quantizados (senão o motor roda no heurístico evalSimple, igual
// default de main.cpp/gui_web); "level <maxdepth> <time_ms>" ajusta
// orçamento de busca do genmove; "eval" mostra a avaliação da posição
// atual sem gastar orçamento de busca completo.
//
// Build: build/build_qtp.bat (Windows/MinGW) ou build/build_qtp.sh (Linux).
// Uso:   bin\qtp_engine.exe [pesos_nnue.bin] [maxdepth] [time_ms]
//        (todos os 3 argumentos de linha de comando são opcionais; sem
//        pesos, roda no heurístico; defaults de busca: depth 40, 200ms)
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "nnue.hpp"

using namespace qr;

namespace {

const char* ENGINE_NAME    = "zquoridor";
const char* ENGINE_VERSION = "1.0";

// --- notação de lance (mesma lógica de gui_web/app.js, portada p/ C++) ---
char colLetter(int c) { return (char)('a' + c); }

std::string pawnNotation(int cell) {
    int er = cell / N, ec = cell % N;
    return std::string(1, colLetter(ec)) + std::to_string(N - er);
}
std::string wallNotation(int orientation, int r, int c) {
    return std::string(1, colLetter(c)) + std::to_string(N - 1 - r) + (orientation == 0 ? "h" : "v");
}

bool parseUInt(const std::string& s, int& out) {
    if (s.empty()) return false;
    for (char ch : s) if (!std::isdigit((unsigned char)ch)) return false;
    try { out = std::stoi(s); } catch (...) { return false; }
    return true;
}

// "e5" -> cell. Falha (retorna false) se fora do tabuleiro ou mal formado.
bool parsePawnToken(const std::string& tokRaw, int& cell) {
    if (tokRaw.size() < 2) return false;
    char cch = (char)std::tolower((unsigned char)tokRaw[0]);
    if (cch < 'a' || cch >= 'a' + N) return false;
    int c = cch - 'a';
    int rowNum;
    if (!parseUInt(tokRaw.substr(1), rowNum)) return false;
    if (rowNum < 1 || rowNum > N) return false;
    cell = (N - rowNum) * N + c;
    return true;
}

// "a3h" / "e5v" -> orientação(0=H,1=V) + slot (r,c). Falha se mal formado.
bool parseWallToken(const std::string& tokRaw, int& orientation, int& r, int& c) {
    if (tokRaw.size() < 3) return false;
    char suffix = (char)std::tolower((unsigned char)tokRaw.back());
    if (suffix != 'h' && suffix != 'v') return false;
    orientation = (suffix == 'h') ? 0 : 1;
    std::string body = tokRaw.substr(0, tokRaw.size() - 1);
    if (body.size() < 2) return false;
    char cch = (char)std::tolower((unsigned char)body[0]);
    if (cch < 'a' || cch >= 'a' + WS) return false;
    c = cch - 'a';
    int rowNum;
    if (!parseUInt(body.substr(1), rowNum)) return false;
    if (rowNum < 1 || rowNum > WS) return false;
    r = WS - rowNum;   // inverso de wallNotation: label = N-1-r = WS-r (WS==N-1)
    return true;
}

// --- cores: jogador 0 = black, jogador 1 = white (ver nota no topo) ------
int parseColor(std::string tok) {
    for (auto& ch : tok) ch = (char)std::tolower((unsigned char)ch);
    if (tok == "black" || tok == "b") return 0;
    if (tok == "white" || tok == "w") return 1;
    return -1;
}
std::string colorName(int player) { return player == 0 ? "black" : "white"; }

// --- estado do motor -------------------------------------------------------
State g_state;
Negamax g_engine;
RepetitionTable g_reptbl;
std::vector<State> g_history;   // g_history.back() == g_state sempre; usado por undo
int g_maxDepth = 40;
int g_timeMs   = 200;

void resetGame() {
    g_state = initialState();
    g_reptbl.clear();
    g_reptbl.push(g_state.hash);
    g_history.clear();
    g_history.push_back(g_state);
}

const std::vector<std::string> COMMANDS = {
    "name", "known_command", "list_commands", "quit",
    "boardsize", "clear_board", "walls",
    "playmove", "playwall", "genmove", "undo", "winner", "showboard",
    "loadnnue", "level", "eval",
};

// --- respostas (secao 1.6/1.7 do QTP: "= resultado\n\n" ou "? erro\n\n") --
void respondOk(const std::string& body) {
    if (body.empty()) std::cout << "=\n\n";
    else std::cout << "= " << body << "\n\n";
    std::cout.flush();
}
void respondFail(const std::string& msg) {
    std::cout << "? " << msg << "\n\n";
    std::cout.flush();
}

// --- handlers de comando ---------------------------------------------------
void cmdName() { respondOk(std::string(ENGINE_NAME) + " " + ENGINE_VERSION); }

void cmdKnownCommand(const std::vector<std::string>& args) {
    if (args.empty()) { respondFail("syntax error"); return; }
    std::string q = args[0];
    for (auto& ch : q) ch = (char)std::tolower((unsigned char)ch);
    bool known = std::find(COMMANDS.begin(), COMMANDS.end(), q) != COMMANDS.end();
    respondOk(known ? "true" : "false");
}

void cmdListCommands() {
    std::string body;
    for (size_t i = 0; i < COMMANDS.size(); i++) {
        if (i) body += "\n";
        body += COMMANDS[i];
    }
    respondOk(body);
}

void cmdBoardsize(const std::vector<std::string>& args) {
    int n;
    if (args.empty() || !parseUInt(args[0], n)) { respondFail("syntax error"); return; }
    // N (rules.hpp) é fixo em tempo de compilação -- só 9 é aceito.
    if (n != N) { respondFail("unacceptable size"); return; }
    respondOk("");
}

void cmdClearBoard() { resetGame(); respondOk(""); }

void cmdWalls(const std::vector<std::string>& args) {
    int n;
    if (args.empty() || !parseUInt(args[0], n)) { respondFail("syntax error"); return; }
    g_state.wallsLeft[0] = (int8_t)n;
    g_state.wallsLeft[1] = (int8_t)n;
    g_history.back() = g_state;  // wallsLeft não entra no hash Zobrist (ok mutar direto), mas o topo do histórico precisa refletir
    respondOk("");
}

void cmdPlayMove(const std::vector<std::string>& args) {
    if (args.size() < 2) { respondFail("syntax error"); return; }
    int color = parseColor(args[0]);
    int cell;
    if (color < 0 || !parsePawnToken(args[1], cell)) { respondFail("syntax error"); return; }
    if (color != g_state.turn) { respondFail("illegal move"); return; }
    MoveList moves = legalMoves(g_state);
    for (size_t i = 0; i < moves.size(); i++) {
        if (!moves[i].isWall && moves[i].a == cell) {
            g_state = applyMove(g_state, moves[i]);
            g_reptbl.push(g_state.hash);
            g_history.push_back(g_state);
            respondOk("");
            return;
        }
    }
    respondFail("illegal move");
}

void cmdPlayWall(const std::vector<std::string>& args) {
    if (args.size() < 2) { respondFail("syntax error"); return; }
    int color = parseColor(args[0]);
    int orientation, r, c;
    if (color < 0 || !parseWallToken(args[1], orientation, r, c)) { respondFail("syntax error"); return; }
    if (color != g_state.turn) { respondFail("illegal move"); return; }
    MoveList moves = legalMoves(g_state);
    for (size_t i = 0; i < moves.size(); i++) {
        const Move& m = moves[i];
        if (m.isWall && m.a == orientation && m.b == r && m.c == c) {
            g_state = applyMove(g_state, m);
            g_reptbl.push(g_state.hash);
            g_history.push_back(g_state);
            respondOk("");
            return;
        }
    }
    respondFail("illegal move");
}

void cmdGenMove(const std::vector<std::string>& args) {
    if (args.empty()) { respondFail("syntax error"); return; }
    int color = parseColor(args[0]);
    if (color < 0) { respondFail("syntax error"); return; }
    if (color != g_state.turn) { respondFail("illegal move"); return; }
    if (winner(g_state) != -1) { respondFail("game over"); return; }
    SearchStats st;
    Move m = g_engine.chooseMove(g_state, g_maxDepth, g_timeMs, st, g_reptbl);
    g_state = applyMove(g_state, m);
    g_reptbl.push(g_state.hash);
    g_history.push_back(g_state);
    std::string token = m.isWall ? wallNotation(m.a, m.b, m.c) : pawnNotation(m.a);
    respondOk(token);
}

void cmdUndo(const std::vector<std::string>& args) {
    int times = 1;
    if (!args.empty() && !parseUInt(args[0], times)) { respondFail("syntax error"); return; }
    if (times <= 0 || (int)g_history.size() - 1 < times) { respondFail("cannot undo"); return; }
    for (int i = 0; i < times; i++) g_history.pop_back();
    g_state = g_history.back();
    g_reptbl.clear();
    for (const State& s : g_history) g_reptbl.push(s.hash);
    respondOk("");
}

void cmdWinner() {
    int w = winner(g_state);
    if (w == -1) respondOk("false");
    else respondOk(std::string("true ") + colorName(w));
}

// Desenho simples (QTP 4.5: "may draw as it likes", só p/ debug humano,
// nunca precisa ser parseável). Reaproveita edgeBlocked (rules.hpp) pra
// desenhar os muros -- mesma lógica que decide bloqueio de movimento, não
// uma reimplementação paralela sujeita a divergir.
void cmdShowBoard() {
    std::ostringstream oss;
    for (int er = 0; er < N; er++) {
        int label = N - er;
        oss << (label < 10 ? " " : "") << label << " ";
        for (int c = 0; c < N; c++) {
            int cell = cellIdx(er, c);
            char mark = '.';
            if (g_state.pawn[0] == cell) mark = 'b';
            if (g_state.pawn[1] == cell) mark = 'w';
            oss << mark;
            if (c < N - 1) {
                bool blocked = edgeBlocked(g_state.wallsH, g_state.wallsV, er, c, er, c + 1);
                oss << (blocked ? '|' : ' ');
            }
        }
        oss << "\n";
        if (er < N - 1) {
            oss << "   ";
            for (int c = 0; c < N; c++) {
                bool blocked = edgeBlocked(g_state.wallsH, g_state.wallsV, er, c, er + 1, c);
                oss << (blocked ? '-' : ' ');
                if (c < N - 1) oss << ' ';
            }
            oss << "\n";
        }
    }
    oss << "  ";
    for (int c = 0; c < N; c++) oss << " " << colLetter(c);
    oss << "\n";
    oss << "black(b) muros=" << (int)g_state.wallsLeft[0]
        << "  white(w) muros=" << (int)g_state.wallsLeft[1]
        << "  vez=" << colorName(g_state.turn) << "\n";
    respondOk(oss.str());
}

void cmdLoadNNUE(const std::vector<std::string>& args) {
    if (args.empty()) { respondFail("syntax error"); return; }
    if (!loadWeightsQuant(args[0])) { respondFail("could not load weights"); return; }
    g_engine.setEvalMode(Negamax::EvalMode::NNUE);
    g_engine.setPolicyOrderingEnabled(true);
    respondOk("");
}

void cmdLevel(const std::vector<std::string>& args) {
    int depth, ms;
    if (args.size() < 2 || !parseUInt(args[0], depth) || !parseUInt(args[1], ms)) {
        respondFail("syntax error");
        return;
    }
    g_maxDepth = depth;
    g_timeMs = ms;
    respondOk("");
}

// Avaliação instantânea (1 forward NNUE ou evalSimple, sem gastar
// orçamento de busca) -- só pra inspeção humana/depuração, não afeta
// genmove. Perspectiva de quem tem a vez: positivo = melhor pra quem vai
// jogar agora.
void cmdEval() {
    if (g_engine.getEvalMode() == Negamax::EvalMode::NNUE) {
        AccumulatorQuant acc = buildAccumulatorQuant(g_state, g_state.turn);
        double p = (double)nnueWinProbQuant(acc);  // 0..1
        respondOk(std::to_string((int)std::lround(p * 100.0)) + "% (NNUE, perspectiva de " + colorName(g_state.turn) + ")");
    } else {
        int sc = evalSimple(g_state, g_state.turn);
        respondOk(std::to_string(sc) + " (heuristica, perspectiva de " + colorName(g_state.turn) + ")");
    }
}

} // namespace

int main(int argc, char** argv) {
    resetGame();

    // uso opcional: qtp_engine.exe [pesos_nnue.bin] [maxdepth] [time_ms]
    if (argc > 1) {
        if (loadWeightsQuant(argv[1])) {
            g_engine.setEvalMode(Negamax::EvalMode::NNUE);
            g_engine.setPolicyOrderingEnabled(true);
            std::cerr << "[qtp] pesos NNUE carregados de '" << argv[1] << "'\n";
        } else {
            std::cerr << "[qtp] aviso: nao foi possivel carregar pesos de '" << argv[1]
                       << "', rodando no heuristico (evalSimple)\n";
        }
    }
    if (argc > 2) parseUInt(argv[2], g_maxDepth);
    if (argc > 3) parseUInt(argv[3], g_timeMs);
    std::cerr << "[qtp] " << ENGINE_NAME << " " << ENGINE_VERSION
              << " pronto (depth<=" << g_maxDepth << ", " << g_timeMs << "ms/lance). "
              << "Comandos: list_commands\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        // 1.4/2.1 do QTP: remove CR (linha vinda de Windows/telnet)
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        // 1.9/2.1: descarta comentario ('#' ate o fim da linha)
        auto hashPos = line.find('#');
        if (hashPos != std::string::npos) line = line.substr(0, hashPos);

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        if (tokens.empty()) continue;  // 1.10: linha vazia/so espaco -- ignora, sem resposta

        std::string cmd = tokens[0];
        for (auto& ch : cmd) ch = (char)std::tolower((unsigned char)ch);
        std::vector<std::string> args(tokens.begin() + 1, tokens.end());

        if (cmd == "name") cmdName();
        else if (cmd == "known_command") cmdKnownCommand(args);
        else if (cmd == "list_commands") cmdListCommands();
        else if (cmd == "quit") { respondOk(""); break; }
        else if (cmd == "boardsize") cmdBoardsize(args);
        else if (cmd == "clear_board") cmdClearBoard();
        else if (cmd == "walls") cmdWalls(args);
        else if (cmd == "playmove") cmdPlayMove(args);
        else if (cmd == "playwall") cmdPlayWall(args);
        else if (cmd == "genmove") cmdGenMove(args);
        else if (cmd == "undo") cmdUndo(args);
        else if (cmd == "winner") cmdWinner();
        else if (cmd == "showboard") cmdShowBoard();
        else if (cmd == "loadnnue") cmdLoadNNUE(args);
        else if (cmd == "level") cmdLevel(args);
        else if (cmd == "eval") cmdEval();
        else respondFail("unknown command");
    }
    return 0;
}
