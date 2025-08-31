#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token感知的Schema实体标准化提示生成器

解决大规模Schema实体定义导致标准化提示词过长的问题。
采用智能分批、相关性过滤、动态裁剪等策略优化提示长度。
"""

from typing import Dict, List, Any, Tuple, Optional, Set
from kag.common.registry import Registrable
from .schema_aware_prompt import SchemaAwarePromptBase
import json
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@Registrable.register("tunnel_engineering_std")
class TunnelEngineeringStandardizationPrompt(SchemaAwarePromptBase):
    """
    隧道工程专业实体标准化提示生成器
    
    特点：
    1. 基于输入实体的相关性过滤
    2. 智能同义词映射选择
    3. 动态标准化规则生成
    4. Token使用量优化
    """
    
    # 定义标准化专用模板
    template_zh = {
        "instruction": "请将以下隧道工程术语标准化为规范的专业术语。",
        "standard_terms": "$standard_terms",
        "standardization_rules": "$standardization_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "请按照JSON格式输出标准化结果，格式为：{\"原术语\": \"标准术语\"}"
    }
    
    template_en = {
        "instruction": "Please standardize the following tunnel engineering terms to standard professional terminology.",
        "standard_terms": "$standard_terms",
        "standardization_rules": "$standardization_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "Please output the standardization results in JSON format: {\"Original Term\": \"Standard Term\"}"
    }
    
    @property
    def template_variables(self) -> List[str]:
        """返回模板变量列表"""
        return ["standard_terms", "standardization_rules", "examples", "input"]
    
    def __init__(self, 
                 project_id: str = None,
                 host_addr: str = None,
                 schema=None,
                 max_prompt_tokens: int = 2500,
                 max_synonyms_per_entity: int = 5,
                 similarity_threshold: float = 0.3,
                 **kwargs):
        """
        初始化Token感知标准化提示生成器
        
        Args:
            project_id: 项目ID
            host_addr: OpenSPG服务器地址
            schema: 已加载的Schema对象
            max_prompt_tokens: 最大提示token数
            max_synonyms_per_entity: 每个实体最大同义词数
            similarity_threshold: 相似度阈值
        """
        super().__init__(project_id=project_id, host_addr=host_addr, schema=schema, **kwargs)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_synonyms_per_entity = max_synonyms_per_entity
        self.similarity_threshold = similarity_threshold
        
        # Token计算相关
        self.avg_chars_per_token = 2.5
        self.base_prompt_tokens = 300
        
        logger.info(f"Token感知标准化初始化: max_tokens={max_prompt_tokens}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(len(word) for word in re.findall(r'[a-zA-Z]+', text))
        
        estimated_tokens = int(
            chinese_chars / self.avg_chars_per_token +
            english_words * 1.3 +
            other_chars / 4
        )
        
        return max(estimated_tokens, len(text) // 4)
    
    def calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """
        计算实体相似度
        
        Args:
            entity1: 实体1
            entity2: 实体2
            
        Returns:
            相似度分数 (0-1)
        """
        # 简单的字符级相似度计算
        set1 = set(entity1)
        set2 = set(entity2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # 考虑长度相似性
        length_similarity = 1 - abs(len(entity1) - len(entity2)) / max(len(entity1), len(entity2))
        
        # 综合相似度
        return (jaccard * 0.7 + length_similarity * 0.3)
    
    def select_relevant_entities(self, input_entities: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        选择与输入实体相关的Schema实体
        
        Args:
            input_entities: 输入的实体列表
            
        Returns:
            相关实体及其信息
        """
        relevant_entities = {}
        
        for input_entity in input_entities:
            # 为每个输入实体找到最相关的Schema实体
            candidates = []
            
            for schema_entity, entity_info in self.schema_entities.items():
                # 计算与Schema实体的相似度
                similarity = self.calculate_entity_similarity(input_entity, schema_entity)
                
                # 也检查与中文名称的相似度
                chinese_name = entity_info.get('chinese_name', schema_entity)
                chinese_similarity = self.calculate_entity_similarity(input_entity, chinese_name)
                
                max_similarity = max(similarity, chinese_similarity)
                
                if max_similarity >= self.similarity_threshold:
                    candidates.append((schema_entity, entity_info, max_similarity))
            
            # 按相似度排序，选择最相关的
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # 选择前几个最相关的实体
            for schema_entity, entity_info, similarity in candidates[:3]:
                if schema_entity not in relevant_entities:
                    relevant_entities[schema_entity] = {
                        **entity_info,
                        'similarity_scores': [],
                        'related_inputs': []
                    }
                
                relevant_entities[schema_entity]['similarity_scores'].append(similarity)
                relevant_entities[schema_entity]['related_inputs'].append(input_entity)
        
        return relevant_entities
    
    def build_optimized_synonym_mapping(self, relevant_entities: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        构建优化的同义词映射
        
        Args:
            relevant_entities: 相关实体信息
            
        Returns:
            优化的同义词映射
        """
        synonym_mapping = {}
        current_tokens = 0
        target_tokens = self.max_prompt_tokens - self.base_prompt_tokens
        
        # 按相关性排序实体
        sorted_entities = sorted(
            relevant_entities.items(),
            key=lambda x: max(x[1].get('similarity_scores', [0])),
            reverse=True
        )
        
        for entity_name, entity_info in sorted_entities:
            # 构建该实体的同义词列表
            synonyms = []
            
            # 添加中文名称
            chinese_name = entity_info.get('chinese_name', '')
            if chinese_name and chinese_name != entity_name:
                synonyms.append(chinese_name)
            
            # 添加相关输入实体
            related_inputs = entity_info.get('related_inputs', [])
            for input_entity in related_inputs:
                if input_entity not in synonyms:
                    synonyms.append(input_entity)
            
            # 添加预定义同义词
            predefined_synonyms = entity_info.get('synonyms', [])
            for synonym in predefined_synonyms:
                if synonym not in synonyms and len(synonyms) < self.max_synonyms_per_entity:
                    synonyms.append(synonym)
            
            # 限制同义词数量
            synonyms = synonyms[:self.max_synonyms_per_entity]
            
            # 检查token使用量
            entry_text = f"{entity_name}: {', '.join(synonyms)}"
            entry_tokens = self.estimate_tokens(entry_text)
            
            if current_tokens + entry_tokens <= target_tokens:
                synonym_mapping[entity_name] = synonyms
                current_tokens += entry_tokens
            else:
                # Token已满，停止添加
                break
        
        logger.info(f"构建同义词映射: {len(synonym_mapping)}个实体，预估{current_tokens}个token")
        return synonym_mapping
    
    def get_standardization_prompt_zh(self, input_entities: List[str]) -> Dict[str, Any]:
        """
        获取Token感知的中文标准化提示
        
        Args:
            input_entities: 需要标准化的实体列表
            
        Returns:
            优化后的标准化提示
        """
        # 选择相关实体
        relevant_entities = self.select_relevant_entities(input_entities)
        
        # 构建优化的同义词映射
        synonym_mapping = self.build_optimized_synonym_mapping(relevant_entities)
        
        # 构建单位标准化规则（简化版）
        unit_rules = {
            "长度单位": ["米(m)", "厘米(cm)", "毫米(mm)", "千米(km)"],
            "面积单位": ["平方米(m²)", "平方厘米(cm²)"],
            "体积单位": ["立方米(m³)", "升(L)"],
            "重量单位": ["千克(kg)", "吨(t)", "克(g)"],
            "压力单位": ["兆帕(MPa)", "千帕(kPa)"]
        }
        
        return {
            "instruction": f"请将以下{len(input_entities)}个隧道工程实体标准化为Schema中定义的标准形式。参考{len(synonym_mapping)}个相关实体的标准化规则：",
            "input_entities": input_entities,
            "standardization_mapping": synonym_mapping,
            "unit_standardization": unit_rules,
            "standardization_rules": {
                "实体名称": "使用Schema中定义的标准名称",
                "单位统一": "统一使用国际标准单位",
                "术语规范": "使用行业标准术语",
                "格式统一": "保持命名格式的一致性",
                "相关性": "优先使用与输入实体最相关的标准形式"
            },
            "examples": self._generate_standardization_examples(input_entities, synonym_mapping),
            "output_format": {
                "description": "输出格式为JSON映射，原实体名 -> 标准实体名",
                "schema": {"原实体名": "标准实体名"}
            },
            "optimization_info": {
                "input_count": len(input_entities),
                "relevant_entities": len(relevant_entities),
                "selected_mappings": len(synonym_mapping),
                "token_optimized": True
            }
        }
    
    def _generate_standardization_examples(self, input_entities: List[str], 
                                         synonym_mapping: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        生成标准化示例
        """
        examples = []
        
        # 基于实际的同义词映射生成示例
        example_mappings = []
        for standard_name, synonyms in list(synonym_mapping.items())[:3]:  # 最多3个示例
            if synonyms:
                example_mappings.append((synonyms[0], standard_name))
        
        if example_mappings:
            input_example = [mapping[0] for mapping in example_mappings]
            output_example = {mapping[0]: mapping[1] for mapping in example_mappings}
            
            examples.append({
                "input": input_example,
                "output": output_example
            })
        
        # 添加通用示例
        examples.append({
            "input": ["C30砼", "钢筋混凝土管", "防水卷材"],
            "output": {
                "C30砼": "C30混凝土",
                "钢筋混凝土管": "钢筋混凝土排水管",
                "防水卷材": "防水材料"
            }
        })
        
        return examples
    
    def standardize_entities(self, entities: List[str]) -> Dict[str, str]:
        """
        标准化实体列表
        
        Args:
            entities: 待标准化的实体列表
            
        Returns:
            标准化映射
        """
        # 获取标准化提示
        prompt_data = self.get_standardization_prompt_zh(entities)
        
        # 这里应该调用LLM进行标准化，暂时返回示例结果
        standardized = {}
        
        for entity in entities:
            # 尝试在同义词映射中找到标准形式
            found_standard = None
            for standard_name, synonyms in prompt_data["standardization_mapping"].items():
                if entity in synonyms or entity == standard_name:
                    found_standard = standard_name
                    break
            
            if found_standard:
                standardized[entity] = found_standard
            else:
                # 如果没找到，保持原样
                standardized[entity] = entity
        
        return standardized
    
    def build_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        构建标准化提示（兼容接口）
        
        Args:
            inputs: 输入参数字典，包含 'input' 和 'named_entities' 键
            
        Returns:
            格式化的提示字符串
        """
        # 从 named_entities 中提取实体名称
        named_entities = inputs.get('named_entities', [])
        if isinstance(named_entities, list) and named_entities:
            if isinstance(named_entities[0], dict):
                # 如果是字典格式，提取name字段
                entities = [entity.get('name', str(entity)) for entity in named_entities]
            else:
                # 如果是字符串列表
                entities = named_entities
        else:
            entities = []
        
        return self.format_prompt(entities)
    
    def format_prompt(self, entities: List[str]) -> str:
        """
        格式化标准化提示
        
        Args:
            entities: 待标准化的实体列表
            
        Returns:
            格式化的提示字符串
        """
        prompt_data = self.get_standardization_prompt_zh(entities)
        
        prompt_parts = [
            prompt_data["instruction"],
            f"\n待标准化实体：{', '.join(entities)}",
            "\n\n标准化映射参考："
        ]
        
        # 添加同义词映射
        for standard_name, synonyms in prompt_data["standardization_mapping"].items():
            if synonyms:
                prompt_parts.append(f"- {standard_name}: {', '.join(synonyms)}")
        
        # 添加单位标准化规则
        prompt_parts.append("\n\n单位标准化规则：")
        for unit_type, units in prompt_data["unit_standardization"].items():
            prompt_parts.append(f"- {unit_type}: {', '.join(units)}")
        
        # 添加标准化规则
        prompt_parts.append("\n\n标准化规则：")
        for rule, description in prompt_data["standardization_rules"].items():
            prompt_parts.append(f"- {rule}: {description}")
        
        # 添加示例
        if prompt_data["examples"]:
            prompt_parts.append("\n\n示例：")
            for i, example in enumerate(prompt_data["examples"], 1):
                prompt_parts.append(f"\n示例{i}：")
                prompt_parts.append(f"输入：{example['input']}")
                prompt_parts.append(f"输出：{json.dumps(example['output'], ensure_ascii=False)}")
        
        # 添加输出格式
        prompt_parts.append("\n\n输出格式：")
        prompt_parts.append(prompt_data["output_format"]["description"])
        
        final_prompt = "\n".join(prompt_parts)
        
        # 检查token数量
        estimated_tokens = self.estimate_tokens(final_prompt)
        if estimated_tokens > self.max_prompt_tokens:
            logger.warning(f"标准化提示token数量({estimated_tokens})超过限制({self.max_prompt_tokens})")
        
        return final_prompt
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        获取token统计信息
        """
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "max_synonyms_per_entity": self.max_synonyms_per_entity,
            "similarity_threshold": self.similarity_threshold,
            "total_entities": len(self.schema_entities),
            "avg_chars_per_token": self.avg_chars_per_token
        }


if __name__ == "__main__":
    # 测试Token感知标准化
    std_prompt = TunnelEngineeringStandardizationPrompt("Tunnelknowledge")
    
    test_entities = ["C30砼", "钢筋混凝土管", "防水卷材", "隧道衬砌", "排水系统"]
    
    print("=== Token感知标准化测试 ===")
    print(f"Token统计: {std_prompt.get_token_statistics()}")
    
    # 测试标准化提示
    print("\n=== 标准化提示测试 ===")
    prompt = std_prompt.format_prompt(test_entities)
    print(f"提示长度: {len(prompt)} 字符")
    print(f"估算token: {std_prompt.estimate_tokens(prompt)}")
    
    # 测试build_prompt方法
    print("\n=== build_prompt方法测试 ===")
    test_inputs = {
        'input': '测试输入',
        'named_entities': [{'name': 'C30砼'}, {'name': '钢筋混凝土管'}]
    }
    build_prompt_result = std_prompt.build_prompt(test_inputs)
    print(f"build_prompt结果长度: {len(build_prompt_result)} 字符")
    
    # 测试标准化结果
    print("\n=== 标准化结果测试 ===")
    result = std_prompt.standardize_entities(test_entities)
    print(f"标准化结果: {result}")