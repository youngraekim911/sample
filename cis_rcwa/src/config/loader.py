# -*- coding: utf-8 -*-
"""YAML 설정 로드/파싱."""
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
