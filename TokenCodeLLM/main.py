#!/usr/bin/env python3
"""TokenCode AI v5.9: Gemini 2 Flash chat with stronger context and train-on-command GPU mode."""

import argparse
import importlib
import os
import random
import re
from typing import List

from training.self_training import SyntheticGenerator, score_text_quality
from utils.config import APP_NAME, COPYRIGHT_LINE, DB_PATH, DEFAULTS, MODEL_PATH
from utils.knowledge import CorpusDB, external_search
from utils.ui import AnimatedSpinner, banner, print_ai, print_help_panel, print_system

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
GEMINI_STATUS_PRINTED = False

CHAT_PERSONA_BASE = (
    "Ты TokenCode AI: дружелюбный, разговорчивый и внимательный собеседник. "
    "Отвечай естественно и по делу, но не сухо. Поддерживай диалог вопросами, "
    "помни факты из недавнего контекста, не противоречь уже сказанному. "
    "Если пользователь сообщает личный факт (например имя), используй его в следующих ответах. "
    "Отвечай на языке пользователя."
)

CHAT_STYLE_GUIDE = {
    "balanced": "Делай ответ средней длины: 3-6 предложений.",
    "talkative": "Будь очень болтливым: 8-14 предложений, добавляй примеры, короткие рассуждения и 1-2 уточняющих вопроса в конце.",
}


MAX_HISTORY = 30
LEARN_EVERY = 5
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def get_device() -> str:
    """Preferred PyTorch device for local model training/generation."""
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def mask_secret(value: str | None) -> str:
    if not value:
        return "not-set"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}***{value[-4:]}"




def load_local_env_key():
    """Load GEMINI_API_KEY from local ignored files if env var is absent."""
    if os.getenv("GEMINI_API_KEY"):
        return
    for file_name in (".env.local", "secrets.env", ".env"):
        if not os.path.exists(file_name):
            continue
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() == "GEMINI_API_KEY" and value.strip():
                        os.environ["GEMINI_API_KEY"] = value.strip().strip('"').strip("'")
                        return
        except OSError:
            continue

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
    """Safe Gemini client bootstrap via env var/local ignored files only."""
    global genai, GEMINI_IMPORT_ERROR

    load_local_env_key()
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


def generate_with_local_model(prompt: str, args) -> str | None:
    if not ensure_ml_runtime():
        return None
    if not os.path.exists(args.save_model):
        return None
    try:
        tokenizer = BPETokenizer()
        model = load_model(args, tokenizer)
        return generate_with_model(model, tokenizer, prompt, args.generate_len)
    except Exception:
        return None


def extract_user_name(chat_context: list[str]) -> str | None:
    patterns = [
        r"my name is\s+([A-Za-zА-Яа-я0-9_-]{2,30})",
        r"меня зовут\s+([A-Za-zА-Яа-я0-9_-]{2,30})",
        r"я\s+([A-Za-zА-Яа-я0-9_-]{2,30})",
    ]
    for message in reversed(chat_context):
        if not message.startswith("User: "):
            continue
        text = message[6:].strip()
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
    return None


def build_gemini_chat_prompt(chat_context: list[str], user_text: str, style: str, memory: dict[str, str] | None = None) -> str:
    recent = "\n".join(chat_context[-MAX_HISTORY:])
    style_instruction = CHAT_STYLE_GUIDE.get(style, CHAT_STYLE_GUIDE["balanced"])
    memory_hint = ""
    if memory:
        mem_items = ", ".join(f"{k}={v}" for k, v in sorted(memory.items()))
        memory_hint = f"\nПамять о пользователе: {mem_items}\n"
    return (
        f"{CHAT_PERSONA_BASE}\n"
        f"{style_instruction}\n"
        "Ты умный ассистент: делай выводы по контексту, не теряй нить диалога и не противоречь прошлым ответам.\n"
        "Если пользователь просит кратко — сокращай ответ, иначе отвечай развёрнуто и живо.\n"
        f"{memory_hint}"
        "Ниже история диалога. Сначала тихо учти контекст, затем дай цельный ответ без служебных пометок.\n\n"
        f"История:\n{recent}\n"
        f"User: {user_text}\n"
        "AI:"
    )


def extract_memory_facts(db: CorpusDB, user_msg: str):
    msg_lower = user_msg.lower()
    name_match = re.search(r"(?:my name is|меня зовут)\s+([A-Za-zА-Яа-я0-9_-]{2,30})", user_msg, re.IGNORECASE)
    if name_match:
        db.set_memory("name", name_match.group(1))
    if "python" in msg_lower:
        db.set_memory("likes_python", "yes")
    if "rust" in msg_lower:
        db.set_memory("knows_rust", "yes")


def fallback_response(db: CorpusDB, user_text: str, chat_context: list[str], style: str = "balanced") -> str:
    text = user_text.strip().lower()
    user_name = extract_user_name(chat_context)
    if any(g in text for g in ["hi", "hello", "привет", "здар", "добрый"]):
        if user_name:
            return (
                f"Привет, {user_name}! Я сейчас в локальном offline-режиме, но всё равно могу болтать и помогать. "
                "Расскажи, что тебе интересно: идеи, код, обучение ИИ или просто разговор?"
            )
        return (
            "Привет! Я сейчас в локальном offline-режиме, но всё равно могу поддержать разговор. "
            "Можешь спросить что угодно: от ИИ и кода до обычной болтовни."
        )
    if "name" in text or "зовут" in text or "имя" in text:
        if user_name:
            return f"Приятно познакомиться, {user_name}! Я TokenCode AI и запомнил твоё имя в текущем чате."
        return "Приятно познакомиться! Я TokenCode AI (offline-режим). Можешь написать: 'Меня зовут ...'"

    local = db.search(user_text, limit=1)
    if local:
        return local[0][1]

    sample = db.sample_texts(1)
    if sample:
        return (
            "Gemini сейчас недоступен, поэтому отвечаю из локальной базы. "
            f"Вот близкий пример: {sample[0]}"
        )

    return "Я в offline-режиме. Выполни init-db, чтобы наполнить базу, или настрой GEMINI_API_KEY для ответов Gemini."


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
        return False
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
    return True


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
        print_system(f"Gemini request failed: {exc}")
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
    print(f" - torch_device_mode: {get_device()}")
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
    print(f" - conversations: {stats['conversations']}")
    print(f" - memory_keys: {stats['memory_keys']}")


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
    help_text = """/train                  GPU/CPU-обучение (3 эпохи × 200 шагов)
/train fast             быстрое обучение (1 эпоха × 100 шагов)
/train deep             глубокое обучение (10 эпох × 300 шагов)
/search <text>          поиск по локальной БЗ
/db stats               статистика базы данных
/db clear               очистить всю базу
/history                показать историю диалога
/memory                 показать, что ИИ запомнил о тебе
/model info             информация о модели и устройстве
/model <name>           поменять Gemini-модель в рантайме
/mode <balanced|talkative> стиль ответа
/context                показать текущий контекст
/key                    статус API ключа (masked)
/init-db                создать локальную базу
/clear                  очистить контекст диалога
/exit                   выход"""
    print_help_panel(help_text)


def cmd_chat(args):
    global GEMINI_STATUS_PRINTED
    db = CorpusDB(DB_PATH)
    chat_context = [f"{role}: {content}" for role, content in db.recent_conversations(limit=MAX_HISTORY)]
    active_gemini_model = args.gemini_model
    chat_style = "talkative"
    turn_count = 0

    print_system("Интерактивный режим запущен. Сначала пробую локальную модель, при недоступности — Gemini.")
    print_system("Стиль: talkative (болтливый). Можно сменить: /mode balanced")
    print_system("Напиши /help чтобы увидеть все команды")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in {"/exit", "exit", "quit", "/quit"}:
            print("[bye]")
            break
        if cmd == "/help":
            print_chat_help()
            continue
        if cmd == "/clear":
            chat_context = []
            print_system("Контекст очищен")
            continue
        if cmd == "/history":
            if not chat_context:
                print_system("История пуста")
            else:
                for line in chat_context[-MAX_HISTORY:]:
                    print(line)
            continue
        if cmd == "/memory":
            memory = db.get_memory()
            if not memory:
                print_system("Память пока пуста")
            else:
                for key, value in memory.items():
                    print(f" - {key}: {value}")
            continue
        if cmd == "/db stats":
            cmd_stats(args)
            continue
        if cmd == "/db clear":
            db.clear_all()
            chat_context = []
            print_system("База и контекст очищены")
            continue
        if cmd == "/model info":
            ml_runtime = "ok" if ensure_ml_runtime() else "unavailable"
            print_system(
                f"Gemini={active_gemini_model} | Device={get_device().upper()} | ML runtime={ml_runtime}"
            )
            continue
        if cmd == "/key":
            print_system(f"GEMINI_API_KEY: {mask_secret(os.getenv('GEMINI_API_KEY'))}")
            continue
        if cmd.startswith("/model "):
            active_gemini_model = user_input[7:].strip() or active_gemini_model
            print_system(f"Gemini model switched to: {active_gemini_model}")
            continue
        if cmd.startswith("/mode "):
            mode = user_input[6:].strip().lower()
            if mode not in CHAT_STYLE_GUIDE:
                print_system("Неверный режим. Используй: /mode balanced или /mode talkative")
                continue
            chat_style = mode
            print_system(f"Режим диалога: {chat_style}")
            continue
        if cmd == "/context":
            if not chat_context:
                print_system("Контекст пока пуст")
            else:
                print_system("Последние реплики контекста:")
                for line in chat_context[-8:]:
                    print(line)
            continue
        if cmd.startswith("/search-web "):
            cmd_search(argparse.Namespace(query=user_input[12:].strip(), limit=5, external=True))
            continue
        if cmd.startswith("/search "):
            cmd_search(argparse.Namespace(query=user_input[8:].strip(), limit=8, external=False))
            continue
        if cmd.startswith("/teacher "):
            prompt = user_input[9:].strip()
            if not prompt:
                print_system("Пустой prompt")
                continue
            cmd_teacher(argparse.Namespace(prompt=prompt, gemini_model=active_gemini_model))
            continue
        if cmd == "/init-db":
            cmd_init_db(argparse.Namespace(corpus_size=10000))
            continue
        if cmd.startswith("/train"):
            train_args = argparse.Namespace(**vars(args))
            train_args.epochs = 3
            train_args.steps_per_epoch = 200
            if "fast" in cmd:
                train_args.epochs = 1
                train_args.steps_per_epoch = 100
            elif "deep" in cmd:
                train_args.epochs = 10
                train_args.steps_per_epoch = 300

            if not ensure_ml_runtime():
                print_system(
                    "Локальное Torch-обучение недоступно в этом окружении. "
                    "Диалоги продолжают сохраняться в базу автоматически; после фикса PyTorch запусти /train."
                )
                continue

            print_system(
                f"Запуск обучения на устройстве: {get_device().upper()} "
                f"| epochs={train_args.epochs} steps={train_args.steps_per_epoch}"
            )
            cmd_train(train_args)
            continue
        if cmd == "/self-train":
            cmd_self_train(args)
            continue

        memory_context = db.get_memory()
        prompt = build_gemini_chat_prompt(chat_context, user_input, chat_style, memory_context if memory_context else None)

        spinner = AnimatedSpinner("Thinking")
        spinner.start()

        response = generate_with_local_model(prompt, args)

        if response is None:
            try:
                response = generate_with_gemini(prompt, active_gemini_model)
                GEMINI_STATUS_PRINTED = False
                learn_from_teacher(db, user_input, response)
            except Exception as exc:
                if not GEMINI_STATUS_PRINTED:
                    print_system(f"Gemini временно недоступен: {exc}")
                    GEMINI_STATUS_PRINTED = True
                response = None

        if response is None:
            response = fallback_response(db, user_input, chat_context, style=chat_style)

        spinner.stop()
        chat_context.append(f"User: {user_input}")
        chat_context.append(f"AI: {response}")
        chat_context = chat_context[-MAX_HISTORY:]
        db.save_turn("User", user_input)
        db.save_turn("AI", response)
        db.save_generation(user_input, response, score_text_quality(response))
        extract_memory_facts(db, user_input)
        print_ai(response)

        turn_count += 1
        if turn_count % LEARN_EVERY == 0:
            print_system(f"Диалог автоматически сохранён в базу: {db.stats()['texts']} текстов. Для дообучения локальной модели запусти /train")


def build_parser():
    p = argparse.ArgumentParser(description="TokenCode AI v5.9")
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
    a_chat.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)

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
    a_teacher.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)

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
