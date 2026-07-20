// app.js -- GUI do Quoridor (v3): tabuleiro cabe na tela, flip automático
// (humano sempre joga de baixo pra cima), muros por "modo de colocação"
// (seleciona orientação, toca ou arrasta direto no tabuleiro), peão sempre
// clicável, barra de controles com 3 botões largos, histórico de lances
// com avaliação do motor, nova partida via modal (lado + força). Estado do
// jogo mora inteiro no módulo WASM; este arquivo só desenha e traduz
// gestos em chamadas ccall.

const N = 9, WS = 8;
let CELL = 48, GAP = 8;

let Q = null;
let humanSide = 1;
let flipped = true;       // true => humano sempre embaixo, subindo
let engineTimeMs = 500;
let theme = "chess";      // "chess" (madeira, padrão) ou "classic" (azul/vermelho)
let busy = false;
let gameGen = 0;           // incrementado a cada "nova partida"/troca de lado, pra
                            // cancelar uma jogada do motor que ficou pendente num
                            // setTimeout de uma partida anterior (ex: trocou de
                            // lado enquanto o motor ainda estava "pensando")
let moveLog = [];          // { mover: 0|1, text: "e3", eval?: number }

function setupWasmBindings(Module) {
    const w = (name, ret, args) => Module.cwrap(name, ret, args);
    const hasEval = typeof Module["_qr_last_move_eval"] === "function";
    return {
        newGame: w("qr_new_game", null, []),
        turn: w("qr_turn", "number", []),
        winner: w("qr_winner", "number", []),
        pawn: w("qr_pawn", "number", ["number"]),
        wallsLeft: w("qr_walls_left", "number", ["number"]),
        wallHBit: w("qr_wall_h_bit", "number", ["number"]),
        wallVBit: w("qr_wall_v_bit", "number", ["number"]),
        distToGoal: w("qr_dist_to_goal", "number", ["number"]),
        movesCount: w("qr_legal_moves_count", "number", []),
        moveIsWall: w("qr_legal_move_is_wall", "number", ["number"]),
        moveA: w("qr_legal_move_a", "number", ["number"]),
        moveB: w("qr_legal_move_b", "number", ["number"]),
        moveC: w("qr_legal_move_c", "number", ["number"]),
        applyPawn: w("qr_apply_pawn_move", "number", ["number"]),
        applyWall: w("qr_apply_wall_move", "number", ["number", "number", "number"]),
        engineMove: w("qr_engine_move", "number", ["number", "number"]),
        lastMoveIsWall: w("qr_last_move_is_wall", "number", []),
        lastMoveA: w("qr_last_move_a", "number", []),
        lastMoveB: w("qr_last_move_b", "number", []),
        lastMoveC: w("qr_last_move_c", "number", []),
        lastMoveEval: hasEval ? w("qr_last_move_eval", "number", []) : null,
    };
}

// --- notação algébrica -------------------------------------------------
// Sempre no referencial fixo do tabuleiro (lado das brancas / primeiro
// jogador), nunca no referencial visual espelhado (`flipped`): coluna
// a..i da esquerda pra direita, linha 1..9 de baixo pra cima -- ou seja,
// engine_row=8 (base do jogador 1, embaixo no layout padrão) é a linha 1,
// e engine_row=0 (base do jogador 0/primeiro jogador) é a linha 9.
function colLetter(c) { return String.fromCharCode(97 + c); }
function pawnNotation(cell) {
    const er = Math.floor(cell / N), ec = cell % N;
    return colLetter(ec) + (N - er);
}
function wallNotation(orientation, r, c) {
    // r é o índice do "corredor" entre engine_row r e r+1 (0..7). O lado
    // sul desse corredor (engine_row r+1, mais embaixo) vira a linha-âncora
    // da notação: (N-1-r) = 8-r.
    return colLetter(c) + (N - 1 - r) + (orientation === 0 ? "h" : "v");
}

function formatEval(v) {
    if (v === null || v === undefined) return "";
    const sign = v > 0 ? "+" : "";
    return sign + v;
}

// --- geometria: converte coordenadas de exibição <-> coordenadas do motor.
// flip só espelha linhas (eixo vertical); colunas nunca mudam. Assim um
// giro de 180° nunca troca H por V.
function dispRowToEngine(dr) { return flipped ? (N - 1 - dr) : dr; }
function dispWallRowToEngine(dr) { return flipped ? (WS - 1 - dr) : dr; }
function engineWallRowToDisp(er) { return flipped ? (WS - 1 - er) : er; }

function slotPos(el) {
    const dr = Number(el.dataset.dr), dc = Number(el.dataset.dc);
    const pad = 0, unit = CELL + GAP;
    if (el.classList.contains("wall-slot-h")) {
        el.style.left = (pad + dc * unit) + "px";
        el.style.top = (pad + dr * unit + CELL) + "px";
        el.style.width = (2 * CELL + GAP) + "px";
        el.style.height = GAP + "px";
    } else {
        el.style.left = (pad + dc * unit + CELL) + "px";
        el.style.top = (pad + dr * unit) + "px";
        el.style.width = GAP + "px";
        el.style.height = (2 * CELL + GAP) + "px";
    }
}

function buildBoardDom() {
    const board = document.getElementById("board");
    board.innerHTML = "";

    for (let dr = 0; dr < N; dr++) {
        for (let dc = 0; dc < N; dc++) {
            const er = dispRowToEngine(dr), ec = dc;
            const cell = document.createElement("div");
            cell.className = "cell";
            cell.dataset.cell = String(er * N + ec);
            cell.addEventListener("click", onCellClick);
            board.appendChild(cell);
        }
    }
    for (let dr = 0; dr < WS; dr++) {
        for (let dc = 0; dc < WS; dc++) {
            const er = dispWallRowToEngine(dr), ec = dc;
            const s = document.createElement("div");
            s.className = "wall-slot-h";
            s.dataset.dr = dr; s.dataset.dc = dc;
            s.dataset.er = er; s.dataset.ec = ec;
            board.appendChild(s);
        }
    }
    for (let dr = 0; dr < WS; dr++) {
        for (let dc = 0; dc < WS; dc++) {
            const er = dispWallRowToEngine(dr), ec = dc;
            const s = document.createElement("div");
            s.className = "wall-slot-v";
            s.dataset.dr = dr; s.dataset.dc = dc;
            s.dataset.er = er; s.dataset.ec = ec;
            board.appendChild(s);
        }
    }
    document.querySelectorAll(".wall-slot-h, .wall-slot-v").forEach(slotPos);
}

function fitBoard() {
    const wrap = document.getElementById("boardWrap");
    const availW = wrap.clientWidth;
    const availH = wrap.clientHeight;
    const side = Math.max(160, Math.min(availW, availH));
    GAP = Math.max(3, Math.round(side * 0.013));
    CELL = Math.max(14, Math.floor((side - GAP * 8) / 9));
    const board = document.getElementById("board");
    board.style.setProperty("--cell", CELL + "px");
    board.style.setProperty("--gap", GAP + "px");
    document.querySelectorAll(".wall-slot-h, .wall-slot-v").forEach(slotPos);
}

function currentLegalMoves() {
    const n = Q.movesCount();
    const moves = [];
    for (let i = 0; i < n; i++) {
        moves.push({ isWall: Q.moveIsWall(i) === 1, a: Q.moveA(i), b: Q.moveB(i), c: Q.moveC(i) });
    }
    return moves;
}

// --- histórico de lances -------------------------------------------------
function pushMove(mover, text, evalScore) {
    moveLog.push({ mover, text, eval: evalScore });
    renderMoveList();
}

function renderMoveList() {
    const el = document.getElementById("movelist");
    if (moveLog.length === 0) {
        el.innerHTML = '<div class="empty"></div>';
        return;
    }
    let html = "";
    for (let i = 0; i < moveLog.length; i += 2) {
        const num = i / 2 + 1;
        const m0 = moveLog[i], m1 = moveLog[i + 1];
        const isLastRow = i >= moveLog.length - 2;
        html += '<div class="row">';
        html += '<span class="num">' + num + '.</span>';
        html += moveSpan(m0, isLastRow && !m1);
        if (m1) html += moveSpan(m1, isLastRow);
        html += "</div>";
    }
    el.innerHTML = html;
    el.scrollTop = el.scrollHeight;
}

function moveSpan(m, isLast) {
    // avaliação do motor: só aparece nos lances do motor (Zquoridor), não
    // nos lances do humano -- é a leitura de posição dele após o próprio lance.
    const evalHtml = (m.eval !== undefined && m.eval !== null) ? '<span class="ev">' + formatEval(m.eval) + '</span>' : "";
    return '<span class="mv p' + m.mover + (isLast ? " last" : "") + '">' + m.text + '</span>' + evalHtml;
}

function render() {
    document.querySelectorAll(".pawn").forEach(el => el.remove());
    for (let p = 0; p < 2; p++) {
        const el = document.querySelector(`.cell[data-cell="${Q.pawn(p)}"]`);
        if (el) {
            const pawnEl = document.createElement("div");
            pawnEl.className = "pawn p" + p;
            el.appendChild(pawnEl);
        }
    }
    document.querySelectorAll(".cell").forEach(el => el.classList.remove("dest-hint", "p0", "p1"));
    document.querySelectorAll(".wall-slot-h").forEach(el => {
        el.classList.toggle("placed", Q.wallHBit(Number(el.dataset.er) * WS + Number(el.dataset.ec)) === 1);
    });
    document.querySelectorAll(".wall-slot-v").forEach(el => {
        el.classList.toggle("placed", Q.wallVBit(Number(el.dataset.er) * WS + Number(el.dataset.ec)) === 1);
    });

    document.getElementById("walls0").textContent = Q.wallsLeft(0) + "/10";
    document.getElementById("walls1").textContent = Q.wallsLeft(1) + "/10";
    renderWallBars("barsP0", Q.wallsLeft(0));
    renderWallBars("barsP1", Q.wallsLeft(1));
    document.getElementById("dist0").textContent = Q.distToGoal(0);
    document.getElementById("dist1").textContent = Q.distToGoal(1);
    document.getElementById("nameP0").textContent = humanSide === 0 ? "Human" : "Zquoridor";
    document.getElementById("nameP1").textContent = humanSide === 1 ? "Human" : "Zquoridor";

    const w = Q.winner();
    const cardP0 = document.getElementById("cardP0");
    const cardP1 = document.getElementById("cardP1");
    cardP0.classList.remove("winner");
    cardP1.classList.remove("winner");

    const wallH = document.getElementById("wallH");
    const wallV = document.getElementById("wallV");

    if (w !== -1) {
        cardP0.classList.remove("active", "thinking");
        cardP1.classList.remove("active", "thinking");
        (w === 0 ? cardP0 : cardP1).classList.add("winner");
        wallH.classList.add("disabled");
        wallV.classList.add("disabled");
        setSelectedWallOrientation(null);
        return;
    }

    const isHumanTurn = Q.turn() === humanSide;
    const moves = currentLegalMoves();

    cardP0.classList.toggle("active", Q.turn() === 0);
    cardP1.classList.toggle("active", Q.turn() === 1);
    cardP0.classList.toggle("thinking", Q.turn() === 0 && !isHumanTurn);
    cardP1.classList.toggle("thinking", Q.turn() === 1 && !isHumanTurn);

    if (isHumanTurn && !busy) {
        moves.filter(m => !m.isWall).forEach(m => {
            const el = document.querySelector(`.cell[data-cell="${m.a}"]`);
            if (el) el.classList.add("dest-hint", "p" + humanSide);
        });
        wallH.classList.remove("disabled");
        wallV.classList.remove("disabled");
        if (selectedWallOrientation !== null) highlightLegalSlots(selectedWallOrientation);
        else clearSlotHighlights();
    } else {
        wallH.classList.add("disabled");
        wallV.classList.add("disabled");
        setSelectedWallOrientation(null);
    }

    if (!isHumanTurn && !busy) {
        busy = true;
        const myGen = gameGen;
        setTimeout(() => {
            if (myGen !== gameGen) { busy = false; return; } // partida trocou enquanto pensava: descarta
            const engineSide = Q.turn();
            Q.engineMove(40, engineTimeMs);
            busy = false;
            if (myGen === gameGen) {
                const text = Q.lastMoveIsWall() === 1
                    ? wallNotation(Q.lastMoveA(), Q.lastMoveB(), Q.lastMoveC())
                    : pawnNotation(Q.lastMoveA());
                // qr_last_move_eval() vem do ponto de vista de quem jogou
                // (convenção negamax padrão do motor -- não mexemos nisso).
                // Pro display, sempre convertemos pra perspectiva do
                // primeiro jogador (brancas): positivo = brancas melhor,
                // negativo = segundo jogador melhor.
                const rawEval = Q.lastMoveEval ? Q.lastMoveEval() : undefined;
                const evalScore = (rawEval === undefined) ? undefined : (engineSide === 0 ? rawEval : -rawEval);
                pushMove(engineSide, text, evalScore);
                render();
            }
        }, 30);
    }
}

function renderWallBars(id, left) {
    const el = document.getElementById(id);
    let html = "";
    for (let i = 0; i < 10; i++) {
        html += '<i class="' + (i < left ? "on" : "") + '"></i>';
    }
    el.innerHTML = html;
}

let suppressNextCellClick = false;

function onCellClick(ev) {
    if (suppressNextCellClick) return;
    if (busy || Q.winner() !== -1 || Q.turn() !== humanSide) return;
    const dest = Number(ev.currentTarget.dataset.cell);
    const mover = Q.turn();
    if (Q.applyPawn(dest) === 1) {
        pushMove(mover, pawnNotation(dest));
        render();
    }
}

// --- colocação de muros -------------------------------------------------
// Duas formas de colocar um muro, ambas sempre disponíveis:
//  1) "clique e jogue": toca em Horizontal/Vertical pra entrar no "modo de
//     colocação" (o botão acende + todos os slots legais ficam com um tom
//     dourado); a partir daí um toque simples em qualquer lugar do
//     tabuleiro coloca o muro no slot mais próximo daquele toque.
//  2) "arrastar e soltar": pressiona o dedo/mouse diretamente no botão
//     Horizontal/Vertical e arrasta sem soltar até o tabuleiro -- durante o
//     arraste, o próprio slot que vai receber o muro se ilumina como uma
//     "barreira fantasma" translúcida (verde = válido, vermelho =
//     inválido) exatamente onde ela vai parar.
// As duas formas usam o mesmo código: um toque que nunca chega a entrar no
// tabuleiro vira (1); um toque que se move até o tabuleiro vira (2).
let selectedWallOrientation = null;
let dragOrientation = null;
let dragActive = false;
let dragLegalSet = null;   // Set "er,ec" dos muros legais na orientação atual
let dragTarget = null;     // { er, ec } ou null
let dragFromButton = null; // orientação, se o gesto começou no botão H/V (não no tabuleiro)
let dragEnteredBoard = false;

function clearSlotHighlights() {
    document.querySelectorAll(".wall-slot-h.legal, .wall-slot-v.legal").forEach(el => el.classList.remove("legal"));
}

function highlightLegalSlots(orientation) {
    clearSlotHighlights();
    if (busy || !Q || Q.winner() !== -1 || Q.turn() !== humanSide) return;
    const sel = orientation === 0 ? ".wall-slot-h" : ".wall-slot-v";
    currentLegalMoves().filter(m => m.isWall && m.a === orientation).forEach(m => {
        const el = document.querySelector(`${sel}[data-er="${m.b}"][data-ec="${m.c}"]`);
        if (el) el.classList.add("legal");
    });
}

function setSelectedWallOrientation(o) {
    selectedWallOrientation = o;
    document.getElementById("wallH").classList.toggle("selected", o === 0);
    document.getElementById("wallV").classList.toggle("selected", o === 1);
    document.getElementById("board").classList.toggle("wall-mode", o !== null);
    if (o !== null) highlightLegalSlots(o); else clearSlotHighlights();
}

function beginWallDrag(orientation, ev, fromButton) {
    if (busy || !Q || Q.winner() !== -1 || Q.turn() !== humanSide) return;
    ev.preventDefault();
    dragOrientation = orientation;
    dragActive = true;
    dragTarget = null;
    dragFromButton = fromButton ? orientation : null;
    dragEnteredBoard = false;
    dragLegalSet = new Set(
        currentLegalMoves().filter(m => m.isWall && m.a === orientation).map(m => m.b + "," + m.c)
    );
    highlightLegalSlots(orientation);

    try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch (e) { /* ambientes sem suporte: ignora */ }

    updateDrag(ev.clientX, ev.clientY);

    window.addEventListener("pointermove", onWindowPointerMove, { passive: false });
    window.addEventListener("pointerup", onWindowPointerUp, { passive: false });
    window.addEventListener("pointercancel", onWindowPointerUp, { passive: false });
}

// pressionar direto no tabuleiro quando já há uma orientação selecionada
// (fluxo "clique e jogue"):
function onBoardPointerDown(ev) {
    if (selectedWallOrientation === null) return;
    beginWallDrag(selectedWallOrientation, ev, false);
}

// pressionar o próprio botão Horizontal/Vertical -- pode virar um arraste
// de verdade se o dedo se mover até o tabuleiro, ou um toque simples que
// alterna o "modo de colocação" se soltar sem sair do botão:
function onWallBtnPointerDown(orientation, ev) {
    if (busy || !Q || Q.winner() !== -1 || Q.turn() !== humanSide) return;
    beginWallDrag(orientation, ev, true);
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function updateDrag(clientX, clientY) {
    document.querySelectorAll(".wall-slot-h, .wall-slot-v").forEach(el => {
        el.classList.remove("ghost-ok", "ghost-bad");
    });

    const board = document.getElementById("board");
    const rect = board.getBoundingClientRect();
    // margem de tolerância: permite iniciar/continuar o arraste um pouco
    // fora da borda física do tabuleiro sem cancelar o gesto.
    const margin = CELL * 0.6;
    const inside = clientX >= rect.left - margin && clientX <= rect.right + margin &&
                   clientY >= rect.top - margin && clientY <= rect.bottom + margin;
    if (!inside) { dragTarget = null; return; }
    dragEnteredBoard = true;

    const x = clientX - rect.left, y = clientY - rect.top;
    const pad = 0, unit = CELL + GAP;
    let dr, dc;
    if (dragOrientation === 0) {
        // Horizontal: a coluna é a "casa" que você clicou -- o muro sempre
        // ocupa essa casa e a da direita. Usa floor (não round) pra que
        // clicar em QUALQUER ponto dentro da casa (não só a metade
        // esquerda) escolha essa casa como âncora.
        dc = clamp(Math.floor((x - pad) / unit), 0, WS - 1);
        dr = clamp(Math.round((y - pad - CELL) / unit), 0, WS - 1);
    } else {
        // Vertical: a linha clicada é sempre a casa "de baixo" do muro
        // (o muro se forma entre ela e a casa acima) -- independe de flip,
        // porque dr aqui já está em coordenadas de exibição; a conversão
        // pra engine (com o espelhamento) acontece só depois, abaixo.
        dc = clamp(Math.round((x - pad - CELL) / unit), 0, WS - 1);
        const clickedRow = Math.floor((y - pad) / unit);
        dr = clamp(clickedRow - 1, 0, WS - 1);
    }

    const er = dispWallRowToEngine(dr), ec = dc;
    const sel = dragOrientation === 0 ? ".wall-slot-h" : ".wall-slot-v";
    const target = document.querySelector(`${sel}[data-er="${er}"][data-ec="${ec}"]`);
    const legal = dragLegalSet.has(er + "," + ec);
    if (target) target.classList.add(legal ? "ghost-ok" : "ghost-bad");
    dragTarget = legal ? { er, ec } : null;
}

function onWindowPointerMove(ev) {
    if (!dragActive) return;
    ev.preventDefault();
    updateDrag(ev.clientX, ev.clientY);
}

function onWindowPointerUp(ev) {
    if (!dragActive) return;
    dragActive = false;
    window.removeEventListener("pointermove", onWindowPointerMove);
    window.removeEventListener("pointerup", onWindowPointerUp);
    window.removeEventListener("pointercancel", onWindowPointerUp);

    document.querySelectorAll(".wall-slot-h, .wall-slot-v").forEach(el => {
        el.classList.remove("ghost-ok", "ghost-bad");
    });

    // uma pressão sem arraste real (toque simples pra colocar o muro) termina
    // com pointerdown e pointerup no mesmo elemento -- o navegador dispara um
    // "click" sintético logo em seguida, que tentaria mover o peão da casa
    // tocada. Suprime esse único click subsequente.
    suppressNextCellClick = true;
    setTimeout(() => { suppressNextCellClick = false; }, 0);

    let applied = false;
    if (dragTarget) {
        const mover = Q.turn();
        const orientation = dragOrientation;
        if (Q.applyWall(orientation, dragTarget.er, dragTarget.ec) === 1) {
            pushMove(mover, wallNotation(orientation, dragTarget.er, dragTarget.ec));
            applied = true;
        }
    }

    if (!applied && dragFromButton !== null && !dragEnteredBoard) {
        // toque simples no botão, sem nunca entrar no tabuleiro -> alterna
        // o "modo de colocação" (clique-e-jogue).
        setSelectedWallOrientation(selectedWallOrientation === dragFromButton ? null : dragFromButton);
    }

    dragTarget = null;
    dragOrientation = null;
    dragFromButton = null;
    dragEnteredBoard = false;

    if (applied) render();
}

// --- modal: nova partida -------------------------------------------------
let modalSide = humanSide;
let modalStrength = engineTimeMs;
let modalTheme = theme;

const THEME_LABELS = {
    chess: ["clara", "escura"],
    classic: ["azul", "vermelho"],
};

function applyTheme(t) {
    theme = t;
    document.documentElement.dataset.theme = t;
}

function openNewGameModal() {
    modalSide = humanSide;
    modalStrength = engineTimeMs;
    modalTheme = theme;
    syncModalSelection();
    document.getElementById("newGameModal").hidden = false;
}

function closeNewGameModal() {
    document.getElementById("newGameModal").hidden = true;
}

function syncModalSelection() {
    document.querySelectorAll("#sideChoices .choice-btn").forEach(btn => {
        btn.classList.toggle("selected", Number(btn.dataset.side) === modalSide);
    });
    document.querySelectorAll("#strengthChoices .choice-btn").forEach(btn => {
        btn.classList.toggle("selected", Number(btn.dataset.ms) === modalStrength);
    });
    document.querySelectorAll("#themeChoices .choice-btn").forEach(btn => {
        btn.classList.toggle("selected", btn.dataset.theme === modalTheme);
    });
    const labels = THEME_LABELS[modalTheme] || THEME_LABELS.chess;
    document.getElementById("sideSubP0").textContent = labels[0];
    document.getElementById("sideSubP1").textContent = labels[1];
}

function startNewGame(side, strength) {
    humanSide = side;
    engineTimeMs = strength;
    flipped = (humanSide === 0);
    gameGen++;
    moveLog = [];
    setSelectedWallOrientation(null);
    Q.newGame();
    busy = false;
    buildBoardDom();
    fitBoard();
    renderMoveList();
    render();
}

function wireControls() {
    // botões Horizontal/Vertical funcionam nos dois modos: um toque simples
    // liga/desliga o "modo de colocação" (clique-e-jogue no tabuleiro
    // depois), e um arraste de verdade a partir do próprio botão já solta
    // o muro direto onde o dedo soltar -- ver beginWallDrag/onWindowPointerUp.
    document.getElementById("wallH").addEventListener("pointerdown", (ev) => onWallBtnPointerDown(0, ev));
    document.getElementById("wallV").addEventListener("pointerdown", (ev) => onWallBtnPointerDown(1, ev));
    document.getElementById("board").addEventListener("pointerdown", onBoardPointerDown);

    document.getElementById("flipBtn").addEventListener("click", () => {
        flipped = !flipped;
        buildBoardDom();
        fitBoard();
        render();
    });

    document.getElementById("newGame").addEventListener("click", () => {
        openNewGameModal();
    });

    document.getElementById("newGameModal").addEventListener("click", (ev) => {
        if (ev.target.id === "newGameModal") closeNewGameModal();
    });
    document.querySelectorAll("#sideChoices .choice-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            modalSide = Number(btn.dataset.side);
            syncModalSelection();
        });
    });
    document.querySelectorAll("#strengthChoices .choice-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            modalStrength = Number(btn.dataset.ms);
            syncModalSelection();
        });
    });
    document.querySelectorAll("#themeChoices .choice-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            modalTheme = btn.dataset.theme;
            syncModalSelection();
        });
    });
    document.getElementById("modalConfirm").addEventListener("click", () => {
        closeNewGameModal();
        applyTheme(modalTheme);
        startNewGame(modalSide, modalStrength);
    });

    window.addEventListener("resize", fitBoard);
    window.addEventListener("orientationchange", fitBoard);
    if (window.ResizeObserver) {
        new ResizeObserver(fitBoard).observe(document.getElementById("boardWrap"));
    }
}

QuoridorModule().then((Module) => {
    Q = setupWasmBindings(Module);
    applyTheme(theme);
    flipped = (humanSide === 0);
    buildBoardDom();
    wireControls();
    Q.newGame();
    fitBoard();
    render();
}).catch((err) => {
    console.error("Erro carregando o WASM:", err);
    alert("Erro carregando o motor (WASM): " + err);
});
