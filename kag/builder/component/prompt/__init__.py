#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt组件模块

包含各种提示生成器，用于NER、实体标准化、关系抽取等任务。
"""

from .schema_aware_prompt import SchemaAwarePromptBase
from .tunnel_engineering_ner import TunnelEngineeringNERPrompt
from .tunnel_engineering_std import TunnelEngineeringStandardizationPrompt
from .tunnel_engineering_triple import TunnelEngineeringTriplePrompt

__all__ = [
    "SchemaAwarePromptBase",
    "TunnelEngineeringNERPrompt",
    "TunnelEngineeringStandardizationPrompt",
    "TunnelEngineeringTriplePrompt"
]