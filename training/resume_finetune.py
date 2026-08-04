"""Run the official Octo fine-tuner with full TrainState resume support.

The upstream script currently has no resume flag. This wrapper applies two
small, asserted source transformations in memory; it never edits the upstream
Octo checkout.
"""
from __future__ import annotations

import os
from pathlib import Path


OFFICIAL_SCRIPT=Path(os.environ.get("OCTO_FINETUNE_SCRIPT","/home/sixr/Desktop/octo/scripts/finetune.py"))
RESUME_STATE=Path(os.environ["OCTO_RESUME_STATE"]).resolve()


def patched_source(source: str) -> str:
    create_block="""    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )
"""
    restore_block=create_block+f"""    import orbax.checkpoint as _orbax_checkpoint
    _resume_dir = {str(RESUME_STATE / 'default')!r}
    train_state = _orbax_checkpoint.PyTreeCheckpointer().restore(
        _resume_dir, item=train_state
    )
    logging.info("Resumed full TrainState from %s at step %s", _resume_dir, train_state.step)
"""
    if source.count(create_block)!=1:
        raise RuntimeError("upstream TrainState creation block changed; refusing unsafe patch")
    source=source.replace(create_block,restore_block)
    old="""        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
"""
    new="""        range(int(train_state.step), int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps) - int(train_state.step),
"""
    if source.count(old)!=1:
        raise RuntimeError("upstream training loop changed; refusing unsafe patch")
    return source.replace(old,new)


def main() -> None:
    if not (RESUME_STATE/"default/checkpoint").is_file():
        raise FileNotFoundError(f"full TrainState checkpoint not found: {RESUME_STATE}")
    source=OFFICIAL_SCRIPT.read_text(encoding="utf-8")
    code=compile(patched_source(source),str(OFFICIAL_SCRIPT),"exec")
    globals_dict={"__name__":"__main__","__file__":str(OFFICIAL_SCRIPT)}
    exec(code,globals_dict)


if __name__=="__main__":main()
