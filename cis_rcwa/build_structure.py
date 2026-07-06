#!/usr/bin/env python3
"""구조 생성(npy). 예: python3 build_structure.py -c conf/qcell_config.yaml -o out"""
import runpy, os
os.environ.setdefault("PYTHONPATH", os.path.dirname(__file__))
from src.structure import builder
import sys
sys.argv = sys.argv  # pass-through
builder.main()
