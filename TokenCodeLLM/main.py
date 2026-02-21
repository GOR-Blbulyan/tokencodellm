#!/usr/bin/env python3
"""TokenCode AI v5.1: terminal chat + CPU Torch + Gemini teacher learning."""

import argparse
import importlib
import os
import random
from typing import List

from training.self_training import SyntheticGenerator, score_text_quality
from utils.config import APP_NAME, COPYRIGHT_LINE, DB_PATH, DEFAULTS, MODEL_PATH
from utils.knowledge import CorpusDB, external_search
from utils.ui import AnimatedSpinner, banner

TORCH_IMPORT_ERROR = None
TIKTOKEN_IMPORT_ERROR = None
GEMINI_IMPORT_ERROR = None

# lazy runtime modules
torch = None
nn = None
F = None
tiktoken = None
genai = None
POWER_GPT_CLASS = None
ML_RUNTIME_STATUS_PRINTED = False


# CPU-only mode requested by user.
def get_device() -> str:
    return "cpu"


def mask_secret(value: str | None) -> str:
    if not value:
        return "not-set"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}***{value[-4:]}"


def ensure_ml_runtime() -> bool:
    """Load torch+tiktoken lazily and handle Windows DLL/runtime issues gracefully."""
    global torch, nn, F, tiktoken, TORCH_IMPORT_ERROR, TIKTOKEN_IMPORT_ERROR, ML_RUNTIME_STATUS_PRINTED

    if torch is None:
        try:
            torch = importlib.import_module("torch")
            nn = importlib.import_module("torch.nn")
            F = importlib.import_module("torch.nn.functional")
            TORCH_IMPORT_ERROR = None
        except Exception as exc:
            TORCH_IMPORT_ERROR = exc

    if tiktoken is None:
        try:
            tiktoken = importlib.import_module("tiktoken")
            TIKTOKEN_IMPORT_ERROR = None
        except Exception as exc:
            TIKTOKEN_IMPORT_ERROR = exc

    if TORCH_IMPORT_ERROR is not None:
        if not ML_RUNTIME_STATUS_PRINTED:
            print("[ERROR] PyTorch runtime is unavailable.")
            print(f"        Import error: {TORCH_IMPORT_ERROR}")
            print("        For Windows DLL issues (WinError 1114):")
            print("        1) Install CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("        2) Install Microsoft Visual C++ Redistributable")
            print("        3) Reopen terminal and retry")
            ML_RUNTIME_STATUS_PRINTED = True
        return False

    if TIKTOKEN_IMPORT_ERROR is not None:
        if not ML_RUNTIME_STATUS_PRINTED:
            print("[ERROR] tiktoken runtime is unavailable.")
            print(f"        Import error: {TIKTOKEN_IMPORT_ERROR}")
            print("        Install with: pip install tiktoken")
            ML_RUNTIME_STATUS_PRINTED = True
        return False

    ML_RUNTIME_STATUS_PRINTED = False
    return True


def get_gemini_client():
    """Safe Gemini client bootstrap via env var only (no key in code/db/git)."""
    global genai, GEMINI_IMPORT_ERROR

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY is not set"

    if genai is None:
        try:
            from google import genai as imported_genai

            genai = imported_genai
            GEMINI_IMPORT_ERROR = None
        except Exception as exc:
            GEMINI_IMPORT_ERROR = exc
            return None, f"google-genai import failed: {exc}"

    try:
        return genai.Client(api_key=api_key), None
    except Exception as exc:
        return None, f"Gemini client init failed: {exc}"


def generate_with_gemini(prompt: str, model_name: str) -> str:
    client, err = get_gemini_client()
    if err:
        raise RuntimeError(err)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return (response.text or "").strip()


class BPETokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str) -> List[int]:
        return self.enc.encode(s, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)


def get_power_gpt_class():
    global POWER_GPT_CLASS
    if POWER_GPT_CLASS is not None:
        return POWER_GPT_CLASS

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
            bsz, seq_len, channels = x.shape
            x_norm = self.ln_1(x)
            qkv = self.c_attn(x_norm)
            q, k, v = qkv.split(channels, dim=2)
            k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
            q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
            v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
            x = x + self.resid_dropout(self.c_proj(y))
            return x + self.mlp(self.ln_2(x))

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
            self.token_embedding_table.weight = self.lm_head.weight

        def forward(self, idx, targets=None):
            bsz, seq_len = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(seq_len, device=get_device()))
            x = self.drop(tok_emb + pos_emb)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(bsz * seq_len, -1), targets.view(bsz * seq_len))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
            self.eval()
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            self.train()
            return idx

    POWER_GPT_CLASS = PowerGPT
    return POWER_GPT_CLASS


def get_batch(tokenizer, texts, block_size, batch_size):
    data = []
    pad_id = tokenizer.encode(" ")[0]
    for _ in range(batch_size):
        text = random.choice(texts)
        ids = tokenizer.encode(text + " <|endoftext|>")
        if len(ids) > block_size + 1:
            start = random.randint(0, len(ids) - block_size - 1)
            data.append(ids[start : start + block_size + 1])
        else:
            data.append((ids + [pad_id] * (block_size + 1 - len(ids)))[: block_size + 1])
    data_tensor = torch.tensor(data, dtype=torch.long)
    device = get_device()
    return data_tensor[:, :-1].to(device), data_tensor[:, 1:].to(device)


def load_model(args, tokenizer):
    model_cls = get_power_gpt_class()
    model = model_cls(
        vocab_size=tokenizer.vocab_size,
        block_size=args.seq_len,
        n_embd=args.emb_dim,
        n_layer=args.layers,
        n_head=args.heads,
    ).to(get_device())
    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=get_device()))
    return model


def generate_with_model(model, tokenizer, prompt, generate_len):
    with torch.no_grad():
        context_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=get_device())
        if context_ids.shape[1] > model.block_size:
            context_ids = context_ids[:, -model.block_size :]
        out_ids = model.generate(context_ids, max_new_tokens=generate_len)[0]
        decoded = tokenizer.decode(out_ids.tolist())
        return decoded[len(prompt) :].split("<|endoftext|>")[0].strip()


def fallback_response(db: CorpusDB, user_text: str) -> str:
    local = db.search(user_text, limit=1)
    if local:
        return f"Я пока без ML-ядра, но нашёл в базе: {local[0][1]}"
    random_sample = db.sample_texts(1)
    if random_sample:
        return f"ML-режим недоступен. Ближайшая подсказка из базы: {random_sample[0]}"
    return "ML-режим недоступен. Запусти `init-db`, а затем `train`, когда установишь PyTorch CPU build."


def learn_from_teacher(db: CorpusDB, prompt: str, teacher_answer: str):
    """Save teacher answers into local corpus for future model training."""
    db.insert_texts([f"User: {prompt}\nTeacher: {teacher_answer}"])
    db.save_generation(prompt, teacher_answer, score_text_quality(teacher_answer))


def cmd_init_db(args):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    db = CorpusDB(DB_PATH)
    spinner = AnimatedSpinner("Populating database")
    spinner.start()
    db.insert_texts(SyntheticGenerator().build(args.corpus_size))
    spinner.stop()
    print(f"[OK] Database initialized with {db.count_texts()} texts")


def cmd_train(args, mode="train"):
    if not ensure_ml_runtime():
        return
    db = CorpusDB(DB_PATH)
    tokenizer = BPETokenizer()
    model = load_model(args, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    spinner = AnimatedSpinner("Training")
    spinner.start()
    losses = []
    for epoch in range(args.epochs):
        texts = db.sample_texts(1000)
        for step in range(args.steps_per_epoch):
            xb, yb = get_batch(tokenizer, texts, model.block_size, args.batch_size)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 10 == 0:
                spinner.text = f"Epoch {epoch + 1}/{args.epochs} Step {step} Loss {loss.item():.4f}"
    torch.save(model.state_dict(), args.save_model)
    spinner.stop()
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    db.log_training_run(mode=mode, epochs=args.epochs, steps=args.steps_per_epoch, avg_loss=avg_loss)
    print(f"[OK] Training complete. avg_loss={avg_loss:.4f} model={args.save_model}")


def cmd_generate(args):
    if not ensure_ml_runtime():
        return
    db = CorpusDB(DB_PATH)
    tokenizer = BPETokenizer()
    model = load_model(args, tokenizer)

    spinner = AnimatedSpinner("Generating")
    spinner.start()
    response = generate_with_model(model, tokenizer, args.prompt, args.generate_len)
    spinner.stop()

    quality = score_text_quality(response)
    db.save_generation(args.prompt, response, quality)
    print(f"\nAI> {response}\n[quality={quality:.2f} saved-to-db]")


def cmd_teacher(args):
    db = CorpusDB(DB_PATH)
    spinner = AnimatedSpinner("Gemini teacher")
    spinner.start()
    try:
        teacher_answer = generate_with_gemini(args.prompt, args.gemini_model)
    except Exception as exc:
        spinner.stop()
        print(f"[ERROR] Gemini request failed: {exc}")
        return
    spinner.stop()

    learn_from_teacher(db, args.prompt, teacher_answer)
    print(f"Teacher> {teacher_answer}")
    print("[OK] Teacher response saved for future training")


def cmd_search(args):
    db = CorpusDB(DB_PATH)
    local = db.search(args.query, limit=args.limit)
    if local:
        print("[Local knowledge]")
        for rid, text in local:
            print(f" - ({rid}) {text}")
    elif args.external:
        try:
            print("[External knowledge]")
            print(external_search(args.query))
        except Exception as exc:
            print(f"[WARN] External search failed: {exc}")
    else:
        print("No local matches. Use --external to query web summary source.")


def cmd_doctor(_args):
    print("[Doctor] Runtime diagnostics")
    ok = ensure_ml_runtime()
    print(f" - ml_runtime: {'ok' if ok else 'unavailable'}")
    print(" - torch_device_mode: cpu-forced")
    print(f" - torch_error: {TORCH_IMPORT_ERROR if TORCH_IMPORT_ERROR else 'none'}")
    print(f" - tiktoken_error: {TIKTOKEN_IMPORT_ERROR if TIKTOKEN_IMPORT_ERROR else 'none'}")
    gemini_key = os.getenv("GEMINI_API_KEY")
    print(f" - gemini_key: {mask_secret(gemini_key)}")
    client, err = get_gemini_client()
    print(f" - gemini_client: {'ok' if client else f'unavailable ({err})'}")
    print(" - recommendation: keep GEMINI_API_KEY only in environment variables")


def cmd_stats(_args):
    db = CorpusDB(DB_PATH)
    stats = db.stats()
    print("[Stats]")
    print(f" - texts: {stats['texts']}")
    print(f" - generations: {stats['generations']}")
    print(f" - training_runs: {stats['training_runs']}")
    print(f" - avg_generation_quality: {stats['avg_quality']:.3f}")
    print(f" - last_loss: {stats['last_loss'] if stats['last_loss'] is not None else 'n/a'}")


def cmd_self_train(args):
    if not ensure_ml_runtime():
        return
    db = CorpusDB(DB_PATH)
    synth = SyntheticGenerator().build(args.synthetic_count)
    scored = [(t, score_text_quality(t)) for t in synth]
    selected = [t for t, q in scored if q >= args.min_quality]
    db.insert_texts(selected)
    print(f"[OK] self-training accepted {len(selected)}/{len(synth)} synthetic samples")
    cmd_train(args, mode="self-train")


def print_chat_help():
    print("\n[Commands]")
    print(" /help                 - список команд")
    print(" /stats                - статистика")
    print(" /search <text>        - поиск по локальной БЗ")
    print(" /search-web <text>    - внешний поиск")
    print(" /teacher <text>       - ответ от Gemini + сохранение в обучающую базу")
    print(" /train                - запуск обучения с дефолтами")
    print(" /self-train           - self-training")
    print(" /clear                - очистить контекст диалога")
    print(" /exit                 - выход")
    print(" Любой другой текст = сообщение ИИ\n")


def cmd_chat(args):
    db = CorpusDB(DB_PATH)
    chat_context = []
    model = None
    tokenizer = None

    print("\n✨ Интерактивный режим запущен. Пиши сообщение или /help")

    while True:
        try:
            user_input = input("\033[92mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("[bye]")
            break
        if user_input.lower() == "/help":
            print_chat_help()
            continue
        if user_input.lower() == "/clear":
            chat_context = []
            print("[OK] Контекст очищен")
            continue
        if user_input.lower() == "/stats":
            cmd_stats(args)
            continue
        if user_input.lower().startswith("/search-web "):
            cmd_search(argparse.Namespace(query=user_input[12:].strip(), limit=5, external=True))
            continue
        if user_input.lower().startswith("/search "):
            cmd_search(argparse.Namespace(query=user_input[8:].strip(), limit=5, external=False))
            continue
        if user_input.lower().startswith("/teacher "):
            prompt = user_input[9:].strip()
            if not prompt:
                print("[WARN] Пустой prompt")
                continue
            cmd_teacher(argparse.Namespace(prompt=prompt, gemini_model=args.gemini_model))
            continue
        if user_input.lower() == "/train":
            cmd_train(args)
            continue
        if user_input.lower() == "/self-train":
            cmd_self_train(args)
            continue

        chat_context.append(f"User: {user_input}")
        prompt = "\n".join(chat_context[-4:]) + "\nAI: "

        spinner = AnimatedSpinner("Thinking")
        spinner.start()

        response = None
        if args.teacher_gemini:
            try:
                response = generate_with_gemini(prompt, args.gemini_model)
                learn_from_teacher(db, user_input, response)
            except Exception:
                response = None

        if response is None and ensure_ml_runtime():
            if model is None or tokenizer is None:
                tokenizer = BPETokenizer()
                model = load_model(args, tokenizer)
                model.eval()
            response = generate_with_model(model, tokenizer, prompt, args.generate_len)

        if response is None:
            response = fallback_response(db, user_input)

        spinner.stop()
        chat_context.append(f"AI: {response}")
        db.save_generation(user_input, response, score_text_quality(response))
        print(f"\033[93mAI>\033[0m {response}")


def build_parser():
    p = argparse.ArgumentParser(description="TokenCode AI v5.1")
    sub = p.add_subparsers(dest="command", required=False)

    a_chat = sub.add_parser("chat")
    a_chat.add_argument("--generate-len", type=int, default=DEFAULTS["generate_len"])
    a_chat.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    a_chat.add_argument("--emb-dim", type=int, default=DEFAULTS["emb_dim"])
    a_chat.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    a_chat.add_argument("--heads", type=int, default=DEFAULTS["heads"])
    a_chat.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    a_chat.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    a_chat.add_argument("--steps-per-epoch", type=int, default=DEFAULTS["steps_per_epoch"])
    a_chat.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    a_chat.add_argument("--synthetic-count", type=int, default=400)
    a_chat.add_argument("--min-quality", type=float, default=0.55)
    a_chat.add_argument("--save-model", default=MODEL_PATH)
    a_chat.add_argument("--teacher-gemini", action="store_true")
    a_chat.add_argument("--gemini-model", default="gemini-2.5-flash")

    a_init = sub.add_parser("init-db")
    a_init.add_argument("--corpus-size", type=int, default=10000)

    a_train = sub.add_parser("train")
    a_train.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    a_train.add_argument("--steps-per-epoch", type=int, default=DEFAULTS["steps_per_epoch"])
    a_train.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    a_train.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    a_train.add_argument("--emb-dim", type=int, default=DEFAULTS["emb_dim"])
    a_train.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    a_train.add_argument("--heads", type=int, default=DEFAULTS["heads"])
    a_train.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    a_train.add_argument("--save-model", default=MODEL_PATH)

    a_gen = sub.add_parser("generate")
    a_gen.add_argument("prompt")
    a_gen.add_argument("--generate-len", type=int, default=DEFAULTS["generate_len"])
    a_gen.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    a_gen.add_argument("--emb-dim", type=int, default=DEFAULTS["emb_dim"])
    a_gen.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    a_gen.add_argument("--heads", type=int, default=DEFAULTS["heads"])
    a_gen.add_argument("--save-model", default=MODEL_PATH)

    a_teacher = sub.add_parser("teacher")
    a_teacher.add_argument("prompt")
    a_teacher.add_argument("--gemini-model", default="gemini-2.5-flash")

    a_search = sub.add_parser("search")
    a_search.add_argument("query")
    a_search.add_argument("--limit", type=int, default=5)
    a_search.add_argument("--external", action="store_true")

    sub.add_parser("stats")
    sub.add_parser("doctor")

    a_self = sub.add_parser("self-train")
    a_self.add_argument("--synthetic-count", type=int, default=400)
    a_self.add_argument("--min-quality", type=float, default=0.55)
    a_self.add_argument("--epochs", type=int, default=1)
    a_self.add_argument("--steps-per-epoch", type=int, default=80)
    a_self.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    a_self.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    a_self.add_argument("--emb-dim", type=int, default=DEFAULTS["emb_dim"])
    a_self.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    a_self.add_argument("--heads", type=int, default=DEFAULTS["heads"])
    a_self.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    a_self.add_argument("--save-model", default=MODEL_PATH)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["chat"])

    banner(APP_NAME, COPYRIGHT_LINE, get_device())

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "init-db":
        cmd_init_db(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "teacher":
        cmd_teacher(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "self-train":
        cmd_self_train(args)


if __name__ == "__main__":
    main()
