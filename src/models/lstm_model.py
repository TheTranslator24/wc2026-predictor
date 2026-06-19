# ==============================================================
# FILE: src/models/lstm_model.py
# PURPOSE: PyTorch LSTM that reads a team's recent-match SEQUENCE.
#
# WHY AN LSTM ON TOP OF XGBOOST:
#   Averages hide momentum. "W W W W L" and "L W W W W" share an 80% win
#   rate, but one team is fading and the other surging. An LSTM processes
#   matches in order, so it can feel that trajectory. It sees the game
#   differently from XGBoost, which is exactly why blending them helps.
#
# DEVICE (read this): trains on CPU by DEFAULT, even on Apple Silicon.
#   PyTorch's MPS (Metal) backend is unreliable for RNNs — it can hang or
#   miscompute LSTM ops. This model is small; CPU trains it in a few
#   minutes and is correct. Opt into MPS via MODEL_CONFIG["lstm"]["device"].
#
# THE MEMORY FIX: health is judged by RSS (real RAM), NOT VMS. On macOS,
#   PyTorch reserves huge VMS by design while RSS stays small — a normal
#   state that the old watchdog mislabeled a "leak", causing false kills.
# ==============================================================

import gc
import os
import logging
import signal
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple

# ── macOS CPU-training deadlock fix ───────────────────────────
# These MUST be set BEFORE torch is imported. PyTorch and numpy can each load
# their own OpenMP runtime; on Apple Silicon the two can deadlock at the first
# training step, hanging the process at 0% CPU. Forcing a single OpenMP thread
# (and tolerating duplicate runtimes) eliminates the hang. The model is small,
# so single-threaded CPU is plenty fast.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import MODEL_CONFIG, MODELS_DIR, ELO_CONFIG

logger = logging.getLogger("wc2026.models.lstm")
_SEED = 42


# ==============================================================
# MEMORY MONITORING
# ==============================================================
def _get_memory_usage() -> dict:
    """Return this process's memory. RSS = real RAM (what matters);
    VMS = reserved virtual address space (normally huge on macOS — ignore it)."""
    info = psutil.Process(os.getpid()).memory_info()
    return {
        "rss_mb": info.rss / 1024 / 1024,
        "vms_mb": info.vms / 1024 / 1024,
        "percent": psutil.Process(os.getpid()).memory_percent(),
    }


# ==============================================================
# DEVICE SELECTION
# ==============================================================
def get_device(prefer: str = "cpu") -> torch.device:
    """
    Pick a compute device. Default "cpu" is deliberate on Apple Silicon
    (MPS is unreliable for RNNs). "mps" opts in WITH a CPU fallback for any
    unimplemented op so it degrades gracefully instead of crashing.
    """
    prefer = (prefer or "cpu").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        logger.info("CUDA detected — using CUDA")
        return torch.device("cuda")
    if prefer == "mps":
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            logger.warning("MPS requested for LSTM — not recommended for RNNs; "
                           "CPU fallback enabled for missing ops.")
            return torch.device("mps")
        logger.warning("MPS requested but unavailable — using CPU")
    logger.info("Using CPU for LSTM training (recommended on M1)")
    return torch.device("cpu")


# ==============================================================
# DATASET
# ==============================================================
class SequenceDataset(Dataset):
    """Wraps numpy sequence/label arrays as a PyTorch Dataset."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.from_numpy(np.ascontiguousarray(sequences, dtype=np.float32))
        self.labels    = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ==============================================================
# NETWORK
# ==============================================================
class LSTMNet(nn.Module):
    """
    2-layer stacked LSTM -> dropout -> batch-norm -> linear(3 logits).
    Only the final timestep's hidden state feeds the classifier.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.dropout    = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc         = nn.Linear(hidden_size, output_size)
        logger.info(f"LSTMNet | input={input_size} | hidden={hidden_size} | "
                    f"layers={num_layers} | dropout={dropout} | output={output_size}")

    def forward(self, x):
        lstm_out, _ = self.lstm(x)        # (batch, seq, hidden)
        last = lstm_out[:, -1, :]         # final timestep (batch, hidden)
        out  = self.dropout(last)
        # BatchNorm1d needs >1 sample in train mode; guard the rare size-1 batch.
        if not (self.training and out.shape[0] == 1):
            out = self.batch_norm(out)
        return self.fc(out)               # (batch, 3) logits


# ==============================================================
# TRAINER
# ==============================================================
class LSTMTrainer:
    """Builds sequences, trains with class-weighted loss + early stopping, predicts, persists."""

    # 8 features describing one match FROM a given team's perspective.
    MATCH_FEATURES = [
        "result", "goals_scored", "goals_conceded", "opponent_elo",
        "is_home", "goal_diff", "days_since_prev", "tournament_weight",
    ]
    N_FEATURES = len(MATCH_FEATURES)      # 8; a sample concatenates home+away -> 16

    def __init__(self):
        self.config     = MODEL_CONFIG["lstm"]
        self.device     = get_device(self.config.get("device", "cpu"))
        self.model: Optional[LSTMNet] = None
        self.is_trained = False
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    # ── per-match feature vector (shared by training + inference) ──
    def _tournament_weight(self, tournament) -> float:
        """Tournament importance in [0,1], from the Elo multiplier table /4."""
        t = str(tournament).lower()
        for key, val in ELO_CONFIG["tournament_multipliers"].items():
            if key.lower() in t:
                return float(min(val / 4.0, 1.0))
        return float(min(1.0 / 4.0, 1.0))   # unknown -> friendly

    def _match_features(self, m, team: str, days_since_prev: Optional[float] = None) -> np.ndarray:
        """
        8-dim vector for one match from `team`'s view. LEAKAGE-FREE: opponent
        strength is the PRE-match Elo recorded on the row (home/away_elo_before),
        i.e. the strength as it stood on match day — never post-history ratings.
        """
        is_home  = m["home_team"] == team
        scored   = m["home_score"] if is_home else m["away_score"]
        conceded = m["away_score"] if is_home else m["home_score"]
        res = 1.0 if scored > conceded else (0.5 if scored == conceded else 0.0)
        gd  = float(np.clip((scored - conceded) / 5.0, -1.0, 1.0))

        opp_elo = (m["away_elo_before"] if is_home else m["home_elo_before"]) \
            if ("away_elo_before" in m and "home_elo_before" in m) else 1500.0
        opp_elo_norm = (float(opp_elo) - 1000.0) / 1000.0

        days_norm = 0.5 if days_since_prev is None else float(min(max(days_since_prev, 0.0) / 365.0, 1.0))

        return np.array([
            res,
            min(float(scored) / 5.0, 1.0),
            min(float(conceded) / 5.0, 1.0),
            opp_elo_norm,
            1.0 if is_home else 0.0,
            gd,
            days_norm,
            self._tournament_weight(m.get("tournament", "Friendly")),
        ], dtype=np.float32)

    def _stack_recent(self, hist_oldest_first: list, n: int) -> np.ndarray:
        """(n, 8) array, MOST RECENT at row 0, zero-padded at the end."""
        seq = np.zeros((n, self.N_FEATURES), dtype=np.float32)
        for i, fv in enumerate(hist_oldest_first[-n:][::-1]):
            seq[i] = fv
        return seq

    def _team_sequence(self, df: pd.DataFrame, team: str, before_date) -> np.ndarray:
        """Inference-time: (seq_len, 8) for one team's last matches before a date."""
        n = self.config["sequence_length"]
        seq = np.zeros((n, self.N_FEATURES), dtype=np.float32)
        mask = (((df["home_team"] == team) | (df["away_team"] == team))
                & (df["date"] < before_date) & (df["home_score"].notna()))
        past = df[mask].sort_values("date", ascending=False).head(n).reset_index(drop=True)
        for i in range(len(past)):
            gap = (past.iloc[i]["date"] - past.iloc[i + 1]["date"]).days if i + 1 < len(past) else None
            seq[i] = self._match_features(past.iloc[i], team, gap)
        return seq

    def build_match_sequence(self, df: pd.DataFrame, home: str, away: str, before_date) -> np.ndarray:
        """Inference-time: full (seq_len, 16) sample = home seq concat away seq."""
        h = self._team_sequence(df, home, before_date)
        a = self._team_sequence(df, away, before_date)
        return np.concatenate([h, a], axis=-1)

    # ── bulk sequence building (O(n) rolling histories) ───────
    def build_sequences(self, results_df: pd.DataFrame, elo_calc=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (n_matches, seq_len, 16) training sequences in ONE chronological
        pass. Each team keeps a rolling deque of its last `seq_len` match
        vectors; histories update only AFTER all of a date's matches are read,
        so same-day games never leak into each other. O(n), seconds not minutes.
        """
        n  = self.config["sequence_length"]
        df = results_df.sort_values("date").reset_index(drop=True)
        logger.info(f"Building LSTM sequences | ~{int(df['home_score'].notna().sum()):,} matches...")

        history:   dict[str, deque] = defaultdict(lambda: deque(maxlen=n))
        last_date: dict[str, object] = {}
        sequences: list[np.ndarray] = []
        labels:    list[int] = []
        date_count = 0

        for match_date, day_group in df.groupby("date", sort=True):
            day_updates: list[tuple[str, np.ndarray]] = []
            for _, row in day_group.iterrows():
                if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                    continue
                home, away = row["home_team"], row["away_team"]
                # Sample = each team's history BEFORE this match (leakage-free).
                h_seq = self._stack_recent(list(history[home]), n)
                a_seq = self._stack_recent(list(history[away]), n)
                sequences.append(np.concatenate([h_seq, a_seq], axis=-1))
                if   row["home_score"] > row["away_score"]: labels.append(2)
                elif row["home_score"] < row["away_score"]: labels.append(0)
                else:                                       labels.append(1)
                h_gap = (match_date - last_date[home]).days if home in last_date else None
                a_gap = (match_date - last_date[away]).days if away in last_date else None
                day_updates.append((home, self._match_features(row, home, h_gap)))
                day_updates.append((away, self._match_features(row, away, a_gap)))

            for team, fv in day_updates:      # apply updates AFTER the whole date
                history[team].append(fv)
                last_date[team] = match_date

            date_count += 1
            if date_count % 100 == 0:
                gc.collect()

        # np.stack on equal-shaped 2D arrays -> one contiguous 3D array.
        arr_X = np.stack(sequences, axis=0).astype(np.float32)
        arr_y = np.array(labels, dtype=np.int64)
        logger.info(f"Sequences built | shape={arr_X.shape} | labels={arr_y.shape} | "
                    f"memory={arr_X.nbytes / 1e6:.1f}MB")
        return arr_X, arr_y

    # ── training loop ─────────────────────────────────────────
    def train(self, sequences: np.ndarray, labels: np.ndarray) -> "LSTMTrainer":
        """Class-weighted training with early stopping, LR scheduling, graceful Ctrl+C."""
        torch.manual_seed(_SEED)
        # Runtime guard against the same OpenMP deadlock (works alongside the
        # env vars set at import). Single-threaded is fine for this model size.
        torch.set_num_threads(1)
        rng = np.random.default_rng(_SEED)

        input_size = sequences.shape[2]   # 16
        self.model = LSTMNet(
            input_size=input_size, hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"], dropout=self.config["dropout"],
        ).to(self.device)

        n = len(labels)
        n_val = int(n * 0.2)
        perm = rng.permutation(n)
        v_idx, t_idx = perm[:n_val], perm[n_val:]
        use_cuda = (self.device.type == "cuda")

        train_dl = DataLoader(SequenceDataset(sequences[t_idx], labels[t_idx]),
                              batch_size=self.config["batch_size"], shuffle=True,
                              drop_last=True, pin_memory=use_cuda)
        val_dl   = DataLoader(SequenceDataset(sequences[v_idx], labels[v_idx]),
                              batch_size=self.config["batch_size"], pin_memory=use_cuda)

        # Class-weighted loss: counter home-win majority so draws aren't ignored.
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = (counts.sum() / (3.0 * counts)).astype(np.float32)
        logger.info(f"Class weights (away/draw/home) = {weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}")
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val = float("inf")
        patience_count = 0
        patience = self.config["patience"]
        best_ckpt = MODELS_DIR / "lstm_best_checkpoint.pt"
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)

        # Graceful Ctrl+C: flip a flag, finish cleanly, keep the best checkpoint.
        interrupted = False
        def handle_sigint(signum, frame):
            nonlocal interrupted
            interrupted = True
            logger.warning("Ctrl+C received — finishing current epoch and stopping...")
        original = signal.signal(signal.SIGINT, handle_sigint)

        # Hang guard: if an epoch makes no progress for 10 min, stop.
        last_progress = time.time()
        timed_out = False
        def watch_timeout():
            nonlocal interrupted, timed_out
            while not interrupted and not timed_out:
                if time.time() - last_progress > 600:
                    logger.critical("TIMEOUT: no epoch progress for 600s — stopping.")
                    interrupted = timed_out = True
                    break
                time.sleep(10)
        threading.Thread(target=watch_timeout, daemon=True).start()

        logger.info(f"LSTM training start | device={self.device} | train={len(t_idx):,} | "
                    f"val={len(v_idx):,} | epochs={self.config['epochs']} | patience={patience}")
        try:
            for epoch in range(self.config["epochs"]):
                if interrupted:
                    logger.warning(f"Stopping at epoch {epoch}.")
                    break
                last_progress = time.time()

                # Memory check — judged by RSS (real RAM), not VMS.
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    mem = _get_memory_usage()
                    logger.info(f"[Memory] RSS={mem['rss_mb']:.0f}MB (real) | "
                                f"VMS={mem['vms_mb']:.0f}MB (reserved; large is normal on macOS)")
                    if mem["rss_mb"] > 4000:      # ~4GB real RAM on a 16GB machine = worth noting
                        logger.warning(f"High RSS={mem['rss_mb']:.0f}MB — real memory use is elevated.")

                self.model.train()
                train_loss = 0.0
                for Xb, yb in train_dl:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(self.model(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_dl:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        val_loss += criterion(self.model(Xb), yb).item()

                avg_train = train_loss / max(len(train_dl), 1)
                avg_val   = val_loss   / max(len(val_dl), 1)
                self.train_losses.append(avg_train)
                self.val_losses.append(avg_val)
                scheduler.step(avg_val)

                if epoch == 0 or (epoch + 1) % 5 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.info(f"Epoch {epoch+1:3d} | train_loss={avg_train:.4f} | "
                                f"val_loss={avg_val:.4f} | lr={lr_now:.2e}")

                if avg_val < best_val - 1e-5:
                    best_val = avg_val
                    patience_count = 0
                    torch.save(self.model.state_dict(), best_ckpt)  # nosec B614 - writes our own checkpoint; saving is not a deserialization path
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                        break
        finally:
            signal.signal(signal.SIGINT, original)

        if best_ckpt.exists():
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device, weights_only=True))  # nosec B614 - weights_only=True; loads only our own local checkpoint
            self.is_trained = True
            logger.info(f"LSTM training complete | best_val_loss={best_val:.4f}")
        else:
            # No epoch ever finished (e.g. an early interrupt/timeout). Do NOT
            # mark this as trained — a random LSTM would poison the ensemble.
            # The ensemble will simply use XGBoost only.
            self.is_trained = False
            logger.warning("LSTM produced no checkpoint — left UNTRAINED; ensemble uses XGBoost only.")
        return self

    # ── inference ─────────────────────────────────────────────
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """[away, draw, home] probabilities for one (seq_len, 16) sequence."""
        if not self.is_trained:
            raise RuntimeError("LSTM not trained. Call train() or load() first.")
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]
        self.model.eval()
        x = torch.from_numpy(np.ascontiguousarray(sequence, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=-1)
        return probs.cpu().numpy()[0]

    # ── persistence ───────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODELS_DIR / "lstm_wc2026.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({  # nosec B614 - writes our own checkpoint; saving is not a deserialization path
            "model_state": self.model.state_dict(), "config": self.config,
            "train_losses": self.train_losses, "val_losses": self.val_losses,
            "input_size": self.N_FEATURES * 2,
        }, path)
        logger.info(f"LSTM saved | {path.name}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LSTMTrainer":
        if path is None:
            path = MODELS_DIR / "lstm_wc2026.pt"
        if not path.exists():
            raise FileNotFoundError(f"LSTM model not found: {path}")
        inst = cls()
        # weights_only=False is required here because the checkpoint also holds
        # config + loss lists (not just tensors). RISK ASSESSMENT: this loads ONLY
        # our own locally-produced model file — gitignored, never committed, and
        # published only as an integrity-verified release artifact (see
        # make_release.sh / integrity_check.py). It is never fed untrusted input,
        # so the deserialization risk bandit flags does not apply here.
        ckpt = torch.load(path, map_location=inst.device, weights_only=False)  # nosec B614 - own local file only; never untrusted input
        inst.config       = ckpt["config"]
        inst.train_losses = ckpt.get("train_losses", [])
        inst.val_losses   = ckpt.get("val_losses", [])
        inst.model = LSTMNet(
            input_size=ckpt.get("input_size", inst.N_FEATURES * 2),
            hidden_size=inst.config["hidden_size"], num_layers=inst.config["num_layers"],
            dropout=inst.config["dropout"],
        ).to(inst.device)
        inst.model.load_state_dict(ckpt["model_state"])
        inst.is_trained = True
        logger.info(f"LSTM loaded | {path.name}")
        return inst
