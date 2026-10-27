const isLocal =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:";
const API_BASE = isLocal
  ? "http://localhost:8000"
  : "https://vishnuraju06-chess-bot.hf.space";

const PIECE_MAP = {
  p: "bP", r: "bR", n: "bN", b: "bB", q: "bQ", k: "bK",
  P: "wP", R: "wR", N: "wN", B: "wB", Q: "wQ", K: "wK",
};

const START_COUNTS = { P: 8, N: 2, B: 2, R: 2, Q: 1, p: 8, n: 2, b: 2, r: 2, q: 1 };
const CAPTURE_ORDER = ["q", "r", "b", "n", "p"];

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const statusKickerEl = document.getElementById("status-kicker");
const statusCopyEl = document.getElementById("status-copy");
const statusCardEl = document.getElementById("status-card");
const thinkingEl = document.getElementById("thinking");
const colorModalEl = document.getElementById("color-modal");
const promoModalEl = document.getElementById("promo-modal");
const promoChoicesEl = document.getElementById("promo-choices");
const moveListEl = document.getElementById("move-list");
const plyCountEl = document.getElementById("ply-count");
const capturedWhiteEl = document.getElementById("captured-white");
const capturedBlackEl = document.getElementById("captured-black");
const youColorEl = document.getElementById("you-color");
const engineColorEl = document.getElementById("engine-color");
const rankLabelsEl = document.getElementById("rank-labels");
const fileLabelsEl = document.getElementById("file-labels");

let selectedSquare = null;
let legalMoves = {};
let boardPieces = {};
let busy = false;
let pendingPromotion = null;
let gameId = null;
let humanColor = "white";

function destSquares(from) {
  const entries = legalMoves[from] || [];
  return [...new Set(entries.map((entry) => (typeof entry === "string" ? entry : entry.to)))];
}

function promotionChoices(from, to) {
  const entries = legalMoves[from] || [];
  return entries
    .filter((entry) => typeof entry !== "string" && entry.to === to && entry.promotion)
    .map((entry) => entry.promotion);
}

function fenToBoard(fen) {
  const board = {};
  const rows = fen.split(" ")[0].split("/");
  let rank = 8;
  for (const row of rows) {
    let file = 0;
    for (const c of row) {
      if (Number.isNaN(Number(c))) {
        board["abcdefgh"[file] + rank] = c;
        file += 1;
      } else {
        file += Number(c);
      }
    }
    rank -= 1;
  }
  return board;
}

function squareName(r, c) {
  return "abcdefgh"[c] + (8 - r);
}

function isWhitePiece(letter) {
  return letter === letter.toUpperCase();
}

function isHumanPiece(letter) {
  return letter && isWhitePiece(letter) === (humanColor === "white");
}

function lastMoveSquares(history) {
  if (!history || history.length === 0) return { from: null, to: null };
  const last = history[history.length - 1];
  const uci = typeof last === "string" ? last : last.uci;
  return { from: uci.slice(0, 2), to: uci.slice(2, 4) };
}

function kingSquare(board, turn) {
  const needle = turn === "black" ? "k" : "K";
  return Object.keys(board).find((sq) => board[sq] === needle) || null;
}

function resultLabel(result, termination) {
  if (result === "1-0") return "White wins";
  if (result === "0-1") return "Black wins";
  if (result === "1/2-1/2") return "Draw";
  return termination ? termination.replaceAll("_", " ").toLowerCase() : "Game over";
}

function capturedPieces(fen) {
  const counts = { P: 0, N: 0, B: 0, R: 0, Q: 0, p: 0, n: 0, b: 0, r: 0, q: 0 };
  for (const ch of fen.split(" ")[0]) {
    if (counts[ch] !== undefined) counts[ch] += 1;
  }
  const whiteTook = [];
  const blackTook = [];
  for (const kind of CAPTURE_ORDER) {
    const missingBlack = START_COUNTS[kind] - counts[kind];
    const missingWhite = START_COUNTS[kind.toUpperCase()] - counts[kind.toUpperCase()];
    for (let i = 0; i < missingBlack; i += 1) whiteTook.push(kind);
    for (let i = 0; i < missingWhite; i += 1) blackTook.push(kind.toUpperCase());
  }
  return { whiteTook, blackTook };
}

function renderCaptured(el, pieces) {
  el.innerHTML = pieces
    .map((letter) => `<img src="/pieces/${PIECE_MAP[letter]}.png" alt="${letter}" />`)
    .join("");
}

function renderMoves(history) {
  moveListEl.innerHTML = "";
  const rows = [];
  for (let i = 0; i < history.length; i += 2) {
    const white = history[i];
    const black = history[i + 1];
    const whiteSan = typeof white === "string" ? white : white.san;
    const blackSan = black ? (typeof black === "string" ? black : black.san) : "";
    rows.push(`<li><span class="n">${i / 2 + 1}.</span><span>${whiteSan}</span><span>${blackSan}</span></li>`);
  }
  moveListEl.innerHTML = rows.join("") || `<li><span class="n">—</span><span>No moves yet</span><span></span></li>`;
  plyCountEl.textContent = `${history.length} ply`;
  moveListEl.scrollTop = moveListEl.scrollHeight;
}

function setBusy(value) {
  busy = value;
  thinkingEl.classList.toggle("hidden", !value);
  boardEl.classList.toggle("board-busy", value);
}

async function api(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Request failed");
  }
  return res.json();
}

function showColorChoice() {
  closePromotion();
  selectedSquare = null;
  colorModalEl.classList.remove("hidden");
}

async function startGame(color) {
  colorModalEl.classList.add("hidden");
  closePromotion();
  selectedSquare = null;
  gameId = null;
  setBusy(true);
  try {
    const state = await api("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ color }),
    });
    gameId = state.game_id;
    renderState(state);
  } catch (error) {
    statusEl.textContent = "Could not start game";
    statusCopyEl.textContent = error.message;
    colorModalEl.classList.remove("hidden");
  } finally {
    setBusy(false);
  }
}

async function playMove(uci) {
  setBusy(true);
  try {
    const state = await api("/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, move: uci }),
    });
    selectedSquare = null;
    renderState(state);
  } catch (error) {
    selectedSquare = null;
    try {
      await render();
    } catch {
      // Keep the original move error visible if state refresh also fails.
    }
    statusCopyEl.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function closePromotion() {
  pendingPromotion = null;
  promoModalEl.classList.add("hidden");
  promoChoicesEl.innerHTML = "";
}

function openPromotion(from, to, choices, colorPrefix) {
  pendingPromotion = { from, to };
  promoChoicesEl.innerHTML = "";
  const unique = [...new Set(choices)];
  for (const promo of unique) {
    const letter = colorPrefix === "b" ? promo : promo.toUpperCase();
    const btn = document.createElement("button");
    btn.type = "button";
    btn.title = promo;
    btn.innerHTML = `<img src="/pieces/${PIECE_MAP[letter]}.png" alt="${promo}" />`;
    btn.addEventListener("click", () => {
      const move = (legalMoves[from] || []).find(
        (entry) => typeof entry !== "string" && entry.to === to && entry.promotion === promo,
      );
      const uci = move?.uci || from + to + promo;
      closePromotion();
      playMove(uci);
    });
    promoChoicesEl.appendChild(btn);
  }
  promoModalEl.classList.remove("hidden");
}

function tryMove(from, to) {
  const dests = destSquares(from);
  if (!dests.includes(to)) {
    if (isHumanPiece(boardPieces[to])) {
      selectedSquare = to;
      renderBoardOnly();
      return;
    }
    selectedSquare = null;
    renderBoardOnly();
    return;
  }

  const promos = promotionChoices(from, to);
  if (promos.length > 0) {
    const piece = boardPieces[from];
    openPromotion(from, to, promos, piece && piece === piece.toLowerCase() ? "b" : "w");
    return;
  }

  const move = (legalMoves[from] || []).find(
    (entry) => typeof entry !== "string" && entry.to === to && !entry.promotion,
  );
  playMove(move?.uci || from + to);
}

function updateStatus(state) {
  humanColor = state.human_color || "white";
  youColorEl.textContent = humanColor[0].toUpperCase() + humanColor.slice(1);
  const aiColor = state.ai_color || (humanColor === "white" ? "black" : "white");
  engineColorEl.textContent = aiColor[0].toUpperCase() + aiColor.slice(1);
  statusCardEl.classList.remove("is-check", "is-over");
  if (state.game_over) {
    statusCardEl.classList.add("is-over");
    statusKickerEl.textContent = state.termination
      ? state.termination.replaceAll("_", " ")
      : "Finished";
    statusEl.textContent = resultLabel(state.result, state.termination);
    statusCopyEl.textContent = `Final score ${state.result}. Start a new game to play again.`;
    return;
  }
  if (state.in_check) {
    statusCardEl.classList.add("is-check");
    statusKickerEl.textContent = state.turn === "white" ? "White is in check" : "Black is in check";
    statusEl.textContent = "Check";
    statusCopyEl.textContent = "Your king is attacked. Move it to safety or block the check.";
    return;
  }
  const humanTurn = state.turn === humanColor;
  statusKickerEl.textContent = humanTurn
    ? `${humanColor[0].toUpperCase() + humanColor.slice(1)} to move`
    : "Engine thinking";
  statusEl.textContent = humanTurn ? "Your turn" : "Engine turn";
  statusCopyEl.textContent = `You play ${humanColor}. The neural net plays ${aiColor}.`;
}

function renderBoardOnly() {
  const squares = boardEl.querySelectorAll(".square");
  squares.forEach((sq) => {
    sq.classList.remove("selected", "move-hint", "has-piece");
    const name = sq.dataset.square;
    if (selectedSquare === name) sq.classList.add("selected");
    if (selectedSquare && destSquares(selectedSquare).includes(name)) {
      sq.classList.add("move-hint");
      if (boardPieces[name]) sq.classList.add("has-piece");
    }
  });
}

function renderBoard(state) {
  boardPieces = fenToBoard(state.fen);
  legalMoves = state.legal_moves || {};
  humanColor = state.human_color || "white";
  const flipped = humanColor === "black";
  const files = "abcdefgh".split("");
  const ranks = "87654321".split("");
  if (flipped) {
    files.reverse();
    ranks.reverse();
  }
  [...fileLabelsEl.children].forEach((label, index) => { label.textContent = files[index]; });
  [...rankLabelsEl.children].forEach((label, index) => { label.textContent = ranks[index]; });
  const last = lastMoveSquares(state.move_history);
  const checkedKing = state.in_check ? kingSquare(boardPieces, state.turn) : null;

  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r += 1) {
    for (let c = 0; c < 8; c += 1) {
      const name = flipped
        ? "abcdefgh"[7 - c] + (r + 1)
        : squareName(r, c);
      const sq = document.createElement("div");
      const isLight = (r + c) % 2 === 0;
      sq.className = `square ${isLight ? "light" : "dark"}`;
      sq.dataset.square = name;

      if (name === last.from || name === last.to) sq.classList.add("last-move");
      if (name === checkedKing) sq.classList.add("in-check");
      if (selectedSquare === name) sq.classList.add("selected");
      if (selectedSquare && destSquares(selectedSquare).includes(name)) {
        sq.classList.add("move-hint");
        if (boardPieces[name]) sq.classList.add("has-piece");
      }

      if (boardPieces[name]) {
        const img = document.createElement("img");
        img.src = `/pieces/${PIECE_MAP[boardPieces[name]]}.png`;
        img.alt = boardPieces[name];
        img.className = "piece";
        img.draggable = false;
        sq.appendChild(img);
      }

      sq.addEventListener("pointerdown", (event) => onPointerDown(event, name));
      boardEl.appendChild(sq);
    }
  }
}

async function render() {
  if (!gameId) return;
  const state = await api(`/state?game_id=${encodeURIComponent(gameId)}`);
  renderState(state);
}

function renderState(state) {
  renderBoard(state);
  renderMoves(state.move_history || []);
  const captured = capturedPieces(state.fen);
  renderCaptured(capturedBlackEl, captured.whiteTook);
  renderCaptured(capturedWhiteEl, captured.blackTook);
  updateStatus(state);
}

function activateSquare(square) {
  if (busy || !promoModalEl.classList.contains("hidden")) return;

  if (!selectedSquare) {
    if (isHumanPiece(boardPieces[square]) && destSquares(square).length) {
      selectedSquare = square;
      renderBoardOnly();
    }
    return;
  }

  if (square === selectedSquare) {
    selectedSquare = null;
    renderBoardOnly();
    return;
  }

  tryMove(selectedSquare, square);
}

function onPointerDown(event, square) {
  if (busy || event.button !== 0 || !promoModalEl.classList.contains("hidden")) return;

  const canDrag =
    boardPieces[square] &&
    isHumanPiece(boardPieces[square]) &&
    destSquares(square).length > 0;

  if (canDrag) {
    selectedSquare = square;
    renderBoardOnly();
  }

  const startX = event.clientX;
  const startY = event.clientY;
  let dragging = false;
  const ghost = canDrag ? boardEl.querySelector(`[data-square="${square}"] img`) : null;

  const onMove = (moveEvent) => {
    if (Math.hypot(moveEvent.clientX - startX, moveEvent.clientY - startY) < 8 && !dragging) {
      return;
    }
    if (!canDrag) return;
    dragging = true;
    if (ghost) {
      ghost.style.transform = `translate(${moveEvent.clientX - startX}px, ${moveEvent.clientY - startY}px)`;
      ghost.style.zIndex = "5";
    }
  };

  const onUp = (upEvent) => {
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", onUp);
    if (ghost) ghost.style.transform = "";

    const target = document.elementFromPoint(upEvent.clientX, upEvent.clientY);
    const dropSquare = target?.closest(".square")?.dataset.square;

    if (dragging && canDrag && dropSquare) {
      tryMove(square, dropSquare);
      return;
    }

    if (!dragging && dropSquare) {
      if (canDrag && dropSquare === square) return;
      activateSquare(dropSquare);
    }
  };

  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", onUp);
}

document.getElementById("restart-btn").addEventListener("click", showColorChoice);
document.getElementById("play-white").addEventListener("click", () => startGame("white"));
document.getElementById("play-black").addEventListener("click", () => startGame("black"));
document.getElementById("promo-cancel").addEventListener("click", () => {
  closePromotion();
  selectedSquare = null;
  renderBoardOnly();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closePromotion();
    selectedSquare = null;
    renderBoardOnly();
  }
});

window.onload = showColorChoice;
