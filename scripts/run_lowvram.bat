@echo off
rem Example: 24 GB card, keep ~4 GB of weights resident (headroom = 24 - 4). Change HEADROOM to fit your card/target.
set HEADROOM=20
python main.py --vram-headroom %HEADROOM% --disable-pinned-memory %*
