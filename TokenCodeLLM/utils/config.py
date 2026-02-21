"""Application-wide configuration defaults for TokenCode AI."""

DB_PATH = "tokencode_corpus.db"
MODEL_PATH = "tokencode_power_model.pt"
APP_NAME = "TokenCode AI v5.4"
COPYRIGHT_LINE = "© 2026 TokenCode"

DEFAULTS = {
    "epochs": 2,
    "steps_per_epoch": 120,
    "batch_size": 24,
    "seq_len": 128,
    "emb_dim": 256,
    "layers": 4,
    "heads": 8,
    "lr": 5e-4,
    "generate_len": 120,
}
