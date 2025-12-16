let selectedSquare = null;
let legalMoves = {};

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");

// Map FEN piece letters to sprite filenames on disk.
const PIECE_MAP = {
  p: "bP", r: "bR", n: "bN", b: "bB", q: "bQ", k: "bK",
  P: "wP", R: "wR", N: "wN", B: "wB", Q: "wQ", K: "wK"
};

// ------------------ API ------------------
async function getState() {
  const res = await fetch("/state");
  return res.json();
}

async function startGame() {
  await fetch("/start", { method: "POST" });
  render();
}

async function sendMove(move) {
  const res = await fetch("/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ move })
  });
  if (!res.ok) throw "illegal";
  return res.json();
}

// ------------------ UTIL ------------------
function fenToBoard(fen) {
  const board = {};
  const rows = fen.split(" ")[0].split("/");
  let rank = 8;

  for (const row of rows) {
    let file = 0;
    for (const c of row) {
      if (isNaN(c)) {
        const sq = "abcdefgh"[file] + rank;
        board[sq] = c;
        file++;
      } else {
        file += parseInt(c);
      }
    }
    rank--;
  }
  return board;
}

function squareName(r, c) {
  return "abcdefgh"[c] + (8 - r);
}

// ------------------ RENDER ------------------
async function render() {
  const state = await getState();
  const boardState = fenToBoard(state.fen);
  legalMoves = state.legal_moves || {};

  boardEl.innerHTML = "";

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sqName = squareName(r, c);
      const sq = document.createElement("div");

      sq.className = "square " + ((r + c) % 2 === 0 ? "light" : "dark");
      sq.dataset.square = sqName;

      if (selectedSquare && (legalMoves[selectedSquare] || []).includes(sqName)) {
        sq.classList.add("move-hint");
      }

      if (boardState[sqName]) {
        const img = document.createElement("img");
        img.src = `/pieces/${PIECE_MAP[boardState[sqName]]}.png`;
        img.className = "piece";
        sq.appendChild(img);
      }

      sq.onclick = () => onSquareClick(sqName);
      boardEl.appendChild(sq);
    }
  }

  statusEl.textContent = state.game_over
    ? "Game Over: " + state.result
    : "Your turn (White)";
}

// ------------------ CLICK ------------------
async function onSquareClick(square) {
  if (!selectedSquare) {
    selectedSquare = square;
    highlight(square);
    highlightMoves(square);
    return;
  }

  const move = selectedSquare + square;
  selectedSquare = null;
  clearHighlight();
  clearMoveHints();

  try {
    await sendMove(move);
    render();
  } catch {
    render();
  }
}

// ------------------ UI ------------------
function highlight(square) {
  document.querySelectorAll(".square").forEach(s => {
    if (s.dataset.square === square) s.classList.add("selected");
  });
}

function highlightMoves(square) {
  clearMoveHints();
  document.querySelectorAll(".square").forEach(s => {
    const sqName = s.dataset.square;
    if ((legalMoves[square] || []).includes(sqName)) {
      s.classList.add("move-hint");
    }
  });
}

function clearHighlight() {
  document.querySelectorAll(".square").forEach(s =>
    s.classList.remove("selected")
  );
}

function clearMoveHints() {
  document.querySelectorAll(".square").forEach(s =>
    s.classList.remove("move-hint")
  );
}

// ------------------ INIT ------------------
window.onload = startGame;
