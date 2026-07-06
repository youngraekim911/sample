#!/usr/bin/env python3
"""RCWA 엔진 검증 (해석해 비교)."""
import runpy, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
runpy.run_path(os.path.join(os.path.dirname(__file__), "src", "rcwa", "tests", "validate.py"), run_name="__main__")
