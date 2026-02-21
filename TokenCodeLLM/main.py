#!/usr/bin/env python3
"""
TokenCode AI v3.0 — The "Power" Edition
Features:
- PyTorch GPT-style Architecture
- BPE Tokenizer (tiktoken cl100k_base like GPT-4)
- Flash Attention (scaled_dot_product_attention)
- Dropout & Model Scaling
"""

import argparse, os, sys, time, random, sqlite3, threading, itertools
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError:
    print("ERROR: PyTorch is required. Install: pip install torch")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken is required for BPE. Install: pip install tiktoken")
    sys.exit(1)

TRY_RICH = False
try:
    from rich.console import Console
    from rich.panel import Panel
    TRY_RICH = True
    console = Console()
except ImportError:
    console = None

# ====== Config & Device ======
DB_PATH = "tokencode_corpus.db"
MODEL_PATH = "tokencode_power_model.pt"
APP_NAME = "TokenCode AI v3.0 (Power + BPE)"
COPYRIGHT_LINE = "© 2026 TokenCode"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ====== UI Helpers ======
_spinner_cycle = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])

def banner(device_label=DEVICE):
    txt = f"""
████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗   {APP_NAME}
╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║   {COPYRIGHT_LINE}
   ██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║
   ██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║
   ██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝
Hardware Backend: [ {device_label.upper()} ] | Tokenizer: BPE (cl100k_base)
"""
    if TRY_RICH:
        console.print(Panel(txt, title="System Init", style="bold magenta"))
    else:
        print(txt)

class AnimatedSpinner:
    def __init__(self, text="Working"):
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            s = next(_spinner_cycle)
            sys.stdout.write(f"\r\033[95m{s}\033[0m {self.text} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.text) + 20) + "\r")
        sys.stdout.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

# ====== DB Layer ======
class CorpusDB:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
        """)
        self.conn.commit()

    def insert_texts(self, texts: List[str]):
        now = int(time.time())
        self.conn.executemany("INSERT INTO texts (text, created_at) VALUES (?, ?);", [(t, now) for t in texts])
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM texts;").fetchone()[0]

    def sample_texts(self, limit: int = 100) -> List[str]:
        cur = self.conn.execute("SELECT text FROM texts ORDER BY RANDOM() LIMIT ?", (limit,))
        return [r[0] for r in cur.fetchall()]

    def search(self, query: str, limit: int = 10) -> List[Tuple[int,str]]:
        like = f"%{query}%"
        cur = self.conn.execute("SELECT id, text FROM texts WHERE text LIKE ? LIMIT ?", (like, limit))
        return cur.fetchall()

# ====== Synthetic Data ======
class SyntheticGenerator:
    def __init__(self):
        self.names = ["Alex", "AI", "TokenCode", "The System", "User", "The Architect"]
        self.actions = ["optimizes", "generates", "calculates", "dreams of", "analyzes", "refines"]
        self.concepts = ["neural networks", "quantum states", "data streams", "logic gates", "infinite loops"]

    def build(self, n: int) -> List[str]:
        return [f"{random.choice(self.names)} {random.choice(self.actions)} {random.choice(self.concepts)}. " +
                random.choice(["It works perfectly.", "The output is massive.", "Learning in progress...", "Fascinating results."]) 
                for _ in range(n)]

# ====== BPE Tokenizer (tiktoken) ======
class BPETokenizer:
    def __init__(self):
        # Используем токенизатор от GPT-4
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str) -> List[int]:
        return self.enc.encode(s, allowed_special="all")
        
    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

# ====== PyTorch Model (Flash Attention) ======
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention with Flash Attention implementation
        B, T, C = x.shape
        x_norm = self.ln_1(x)
        
        qkv = self.c_attn(x_norm)
        q, k, v = qkv.split(C, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # PyTorch 2.0+ Flash Attention (Super fast)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.resid_dropout(self.c_proj(y))
        
        # MLP
        x = x + self.mlp(self.ln_2(x))
        return x

class PowerGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, block_size=128, n_head=8, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval() # Turn off dropout for generation
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# ====== Batch Helper ======
def get_batch(tokenizer, texts, block_size, batch_size):
    # Улучшенный сбор батчей под BPE
    data = []
    for _ in range(batch_size):
        t = random.choice(texts)
        ids = tokenizer.encode(t + " <|endoftext|>") # Добавляем токен конца
        if len(ids) > block_size + 1:
            start = random.randint(0, len(ids) - block_size - 1)
            data.append(ids[start:start+block_size+1])
        else:
            padded = ids + [tokenizer.encode(" ")[0]] * (block_size + 1 - len(ids))
            data.append(padded[:block_size+1])
            
    data_tensor = torch.tensor(data, dtype=torch.long)
    x = data_tensor[:, :-1].to(DEVICE)
    y = data_tensor[:, 1:].to(DEVICE)
    return x, y

# ====== CLI Commands ======
def cmd_init_db(args):
    db = CorpusDB()
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    db = CorpusDB()
    gen = SyntheticGenerator()
    spinner = AnimatedSpinner("Populating Database...")
    spinner.start()
    db.insert_texts(gen.build(args.corpus_size))
    spinner.stop()
    print(f"[OK] DB initialized with {db.count()} texts.")

def cmd_train(args):
    db = CorpusDB()
    tokenizer = BPETokenizer()
    
    model = PowerGPT(
        vocab_size=tokenizer.vocab_size, 
        block_size=args.seq_len, 
        n_embd=args.emb_dim,
        n_layer=args.layers,
        n_head=args.heads
    ).to(DEVICE)
    
    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=DEVICE))
        print(f"[INFO] Loaded existing model from {args.save_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    spinner = AnimatedSpinner("Training...")
    spinner.start()
    start_time = time.time()
    
    model.train()
    for epoch in range(args.epochs):
        texts = db.sample_texts(1000)
        for step in range(args.steps_per_epoch):
            xb, yb = get_batch(tokenizer, texts, model.block_size, args.batch_size)
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Градиентный клиппинг (защита от взрыва лосса)
            optimizer.step()
            
            if step % 10 == 0:
                spinner.text = f"Epoch {epoch+1}/{args.epochs} | Step {step} | Loss {loss.item():.4f}"
                
    torch.save(model.state_dict(), args.save_model)
    spinner.stop()
    print(f"\n[OK] Training complete in {time.time() - start_time:.1f}s. Saved to {args.save_model}.")

def cmd_serve(args):
    db = CorpusDB()
    tokenizer = BPETokenizer()
    
    model = PowerGPT(
        vocab_size=tokenizer.vocab_size, 
        block_size=args.seq_len, 
        n_embd=args.emb_dim,
        n_layer=args.layers,
        n_head=args.heads
    ).to(DEVICE)
    
    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=DEVICE))
    model.eval()

    print("\n[Serve Mode] Type 'help' for commands, 'exit' to quit.")
    chat_context = []

    while True:
        try:
            user_input = input("\n\033[92mUser>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input: continue
        if user_input.lower() in ["exit", "quit"]: break
        if user_input.lower() == "help":
            print(" Commands: search <query>, clear (clears memory), exit")
            continue
        if user_input.lower() == "clear":
            chat_context = []
            print("[System] Memory cleared.")
            continue
        if user_input.lower().startswith("search "):
            q = user_input[7:]
            for rid, txt in db.search(q): print(f"  [{rid}] {txt}")
            continue

        chat_context.append(f"User: {user_input}")
        prompt = "\n".join(chat_context[-3:]) + "\nAI: "
        
        spinner = AnimatedSpinner("Thinking...")
        spinner.start()
        
        context_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
        
        # Защита от переполнения контекста
        if context_ids.shape[1] > model.block_size:
            context_ids = context_ids[:, -model.block_size:]
            
        out_ids = model.generate(context_ids, max_new_tokens=args.generate_len)[0]
        response = tokenizer.decode(out_ids.tolist())
        
        ai_response = response[len(prompt):].split("User:")[0].split("<|endoftext|>")[0].strip()
        chat_context.append(f"AI: {ai_response}")
        
        spinner.stop()
        print(f"\033[93mAI>\033[0m {ai_response}")

# ====== Main ======
def main():
    p = argparse.ArgumentParser(description="TokenCode AI v3 Power Edition")
    sub = p.add_subparsers(dest="command", required=True)

    a_init = sub.add_parser("init-db")
    a_init.add_argument("--corpus-size", type=int, default=10000)

    # Увеличенные дефолтные параметры для "Мощи"
    a_train = sub.add_parser("train")
    a_train.add_argument("--epochs", type=int, default=3)
    a_train.add_argument("--steps-per-epoch", type=int, default=200)
    a_train.add_argument("--batch-size", type=int, default=32)
    a_train.add_argument("--seq-len", type=int, default=128)
    a_train.add_argument("--emb-dim", type=int, default=256)
    a_train.add_argument("--layers", type=int, default=4)
    a_train.add_argument("--heads", type=int, default=8)
    a_train.add_argument("--lr", type=float, default=5e-4)
    a_train.add_argument("--save-model", default=MODEL_PATH)

    a_serv = sub.add_parser("serve")
    a_serv.add_argument("--seq-len", type=int, default=128)
    a_serv.add_argument("--emb-dim", type=int, default=256)
    a_serv.add_argument("--layers", type=int, default=4)
    a_serv.add_argument("--heads", type=int, default=8)
    a_serv.add_argument("--generate-len", type=int, default=100)
    a_serv.add_argument("--save-model", default=MODEL_PATH)

    args = p.parse_args()
    banner()

    if args.command == "init-db": cmd_init_db(args)
    elif args.command == "train": cmd_train(args)
    elif args.command == "serve": cmd_serve(args)

if __name__ == "__main__":
    main()
