@echo off
rem For layerstream_h3_15s_1mp.json on a 24 GB card
python main.py --vram-headroom 10 --disable-pinned-memory %*
