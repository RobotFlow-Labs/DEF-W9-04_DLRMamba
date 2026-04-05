from anima_dlrmamba.config import load_config


def test_load_debug_config() -> None:
    cfg = load_config("configs/debug.toml")
    assert cfg.model.embed_dim == 32
    assert cfg.train.batch_size == 2
    assert cfg.infer.topk == 50
