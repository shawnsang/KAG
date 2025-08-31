#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token感知的Schema关系抽取提示生成器

解决大规模Schema关系定义导致三元组抽取提示词过长的问题。
采用实体相关性过滤、关系优先级排序、动态关系选择等策略优化提示长度。
"""

from typing import Dict, List, Any, Tuple, Optional, Set
from kag.common.registry import Registrable
from .schema_aware_prompt import SchemaAwarePromptBase
import json
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@Registrable.register("tunnel_engineering_triple")
class TunnelEngineeringTriplePrompt(SchemaAwarePromptBase):
    """
    隧道工程专业关系抽取提示生成器
    
    特点：
    1. 基于识别实体的相关关系过滤
    2. 关系重要性优先级排序
    3. 动态关系约束生成
    4. Token使用量优化
    5. 多轮关系抽取支持
    """
    
    # 定义三元组抽取专用模板
    template_zh = {
        "instruction": "请从以下隧道工程文本中抽取实体间的关系，形成三元组。",
        "entity_types": "$entity_types",
        "relation_types": "$relation_types",
        "extraction_rules": "$extraction_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "请按照JSON格式输出三元组，格式为：[{\"subject\": \"主体\", \"predicate\": \"关系\", \"object\": \"客体\"}]"
    }
    
    template_en = {
        "instruction": "Please extract relationships between entities from the following tunnel engineering text to form triples.",
        "entity_types": "$entity_types",
        "relation_types": "$relation_types",
        "extraction_rules": "$extraction_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "Please output triples in JSON format: [{\"subject\": \"Subject\", \"predicate\": \"Relation\", \"object\": \"Object\"}]"
    }
    
    @property
    def template_variables(self) -> List[str]:
        """返回模板变量列表"""
        return ["entity_types", "relation_types", "extraction_rules", "examples", "input"]
    
    def __init__(self, 
                 project_id: str = None,
                 host_addr: str = None,
                 schema=None,
                 max_prompt_tokens: int = 3500,
                 max_relations_per_batch: int = 30,
                 min_relations_per_type: int = 2,
                 enable_multi_round: bool = True,
                 **kwargs):
        """
        初始化Token感知关系抽取提示生成器
        
        Args:
            project_id: 项目ID
            host_addr: OpenSPG服务器地址
            schema: 已加载的Schema对象
            max_prompt_tokens: 最大提示token数
            max_relations_per_batch: 每批最大关系数
            min_relations_per_type: 每种类型最少关系数
            enable_multi_round: 是否启用多轮抽取
        """
        super().__init__(project_id=project_id, host_addr=host_addr, schema=schema, **kwargs)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_relations_per_batch = max_relations_per_batch
        self.min_relations_per_type = min_relations_per_type
        self.enable_multi_round = enable_multi_round
        
        # Token计算相关
        self.avg_chars_per_token = 2.5
        self.base_prompt_tokens = 400
        
        # 关系优先级缓存
        self._relation_priorities = None
        
        logger.info(f"Token感知关系抽取初始化: max_tokens={max_prompt_tokens}, max_relations={max_relations_per_batch}")
    
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
    
    def calculate_relation_priorities(self, identified_entities: List[str] = None) -> Dict[str, float]:
        """
        计算关系优先级
        
        Args:
            identified_entities: 已识别的实体列表
            
        Returns:
            关系名称到优先级的映射
        """
        if self._relation_priorities is not None and not identified_entities:
            return self._relation_priorities
        
        priorities = {}
        entity_set = set(identified_entities) if identified_entities else set()
        
        # schema_relations的结构是{category: [relation_names]}
        for category, relation_names in self.schema_relations.items():
            for relation_name in relation_names:
                priority = 1.0  # 基础优先级
                
                # 1. 基于关系类型的优先级
                if category in ['构成关系', '位置关系', '属性关系']:
                    priority += 0.5  # 核心关系类型
                elif category in ['时间关系', '因果关系']:
                    priority += 0.3  # 重要关系类型
                
                # 2. 基于关系涉及的实体类型（简化处理）
                subject_type = ''
                object_type = ''
            
            # 如果有已识别实体，优先选择相关关系
            if identified_entities:
                subject_relevance = any(entity in subject_type or subject_type in entity for entity in entity_set)
                object_relevance = any(entity in object_type or object_type in entity for entity in entity_set)
                
                if subject_relevance and object_relevance:
                    priority += 1.0  # 主客体都相关
                elif subject_relevance or object_relevance:
                    priority += 0.5  # 部分相关
            
            # 3. 基于关系名称的通用性
            chinese_name = relation_name  # 直接使用关系名称
            if len(chinese_name) <= 4:  # 短名称通常更通用
                priority += 0.2
            
            # 4. 基于关系的使用频率（模拟）
            common_relations = ['包含', '位于', '连接', '属于', '具有', '导致']
            if any(common in chinese_name for common in common_relations):
                priority += 0.3
            
            priorities[relation_name] = priority
        
        # 缓存结果
        if not identified_entities:
            self._relation_priorities = priorities
        
        return priorities
    
    def select_relations_by_entities(self, identified_entities: List[str]) -> Dict[str, List[str]]:
        """
        根据已识别实体选择相关关系
        
        Args:
            identified_entities: 已识别的实体列表
            
        Returns:
            按类型分组的选中关系
        """
        # 计算关系优先级
        priorities = self.calculate_relation_priorities(identified_entities)
        
        # 按优先级排序关系
        sorted_relations = sorted(
            priorities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 按关系类型分组
        relation_types = self.get_relation_types()
        selected_relations = defaultdict(list)
        current_tokens = 0
        target_tokens = self.max_prompt_tokens - self.base_prompt_tokens
        
        # 确保每个类型至少有最少数量的关系
        for rel_type, relations in relation_types.items():
            if not relations:
                continue
                
            # 为该类型选择高优先级关系
            type_relations = [r for r in relations if r in priorities]
            type_relations.sort(key=lambda x: priorities[x], reverse=True)
            
            # 添加最少数量的关系
            for relation in type_relations[:self.min_relations_per_type]:
                if current_tokens < target_tokens:
                    selected_relations[rel_type].append(relation)
                    # 估算添加这个关系的token消耗（简化处理）
                    chinese_name = relation
                    description = ''
                    relation_text = f"{chinese_name}: {description[:50]}"  # 限制描述长度
                    current_tokens += self.estimate_tokens(relation_text) + 3
        
        # 剩余token用于添加更多高优先级关系
        remaining_relations = []
        for relation_name, priority in sorted_relations:
            rel_type = self._get_relation_type(relation_name)
            
            if rel_type and relation_name not in selected_relations[rel_type]:
                remaining_relations.append((relation_name, rel_type, priority))
        
        # 按优先级添加剩余关系
        for relation_name, rel_type, priority in remaining_relations:
            if len(selected_relations[rel_type]) >= self.max_relations_per_batch // len(relation_types):
                continue  # 该类型已满
                
            # 简化处理关系信息
            chinese_name = relation_name
            description = ''
            relation_text = f"{chinese_name}: {description[:50]}"
            relation_tokens = self.estimate_tokens(relation_text) + 3
            
            if current_tokens + relation_tokens <= target_tokens:
                selected_relations[rel_type].append(relation_name)
                current_tokens += relation_tokens
            else:
                break  # token已满
        
        # 转换为普通字典并添加省略标记
        result = {}
        for rel_type, relations in selected_relations.items():
            result[rel_type] = relations
            # 如果该类型还有更多关系，添加省略标记
            total_in_type = len(relation_types.get(rel_type, []))
            if len(relations) < total_in_type:
                result[rel_type].append(f"...等（还有{total_in_type - len(relations)}个）")
        
        logger.info(f"Token感知关系选择: 选中{sum(len(relations) for relations in result.values())}个关系，预估{current_tokens}个token")
        return result
    
    def _get_relation_type(self, relation_name: str) -> Optional[str]:
        """
        获取关系所属类型
        """
        relation_types = self.get_relation_types()
        for rel_type, relations in relation_types.items():
            if relation_name in relations:
                return rel_type
        return None
    
    def build_entity_relation_constraints(self, identified_entities: List[str], 
                                        selected_relations: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        构建实体-关系约束
        
        Args:
            identified_entities: 已识别的实体列表
            selected_relations: 选中的关系
            
        Returns:
            实体-关系约束映射
        """
        constraints = {}
        
        # 为每个实体构建可能的关系约束
        for entity in identified_entities:
            entity_constraints = {
                'as_subject': [],  # 作为主体的关系
                'as_object': [],   # 作为客体的关系
                'entity_type': self._infer_entity_type(entity)
            }
            
            # 检查每个选中的关系
            for rel_type, relations in selected_relations.items():
                for relation_name in relations:
                    if relation_name.endswith('等'):
                        continue  # 跳过省略标记
                        
                    # 简化处理关系信息
                    subject_type = ''
                    object_type = ''
                    
                    # 检查实体是否可以作为主体
                    if self._entity_matches_type(entity, subject_type):
                        entity_constraints['as_subject'].append({
                            'relation': relation_name,
                            'chinese_name': relation_name,
                            'object_type': object_type
                        })
                    
                    # 检查实体是否可以作为客体
                    if self._entity_matches_type(entity, object_type):
                        entity_constraints['as_object'].append({
                            'relation': relation_name,
                            'chinese_name': relation_name,
                            'subject_type': subject_type
                        })
            
            if entity_constraints['as_subject'] or entity_constraints['as_object']:
                constraints[entity] = entity_constraints
        
        return constraints
    
    def _infer_entity_type(self, entity: str) -> str:
        """
        推断实体类型
        """
        # 简单的实体类型推断
        if any(keyword in entity for keyword in ['隧道', '洞', '段']):
            return '工程结构'
        elif any(keyword in entity for keyword in ['混凝土', '钢筋', '材料']):
            return '材料设备'
        elif any(keyword in entity for keyword in ['施工', '工艺', '方法']):
            return '施工工艺'
        elif any(keyword in entity for keyword in ['厚度', '长度', '压力']):
            return '技术参数'
        else:
            return '其他'
    
    def _entity_matches_type(self, entity: str, entity_type: str) -> bool:
        """
        检查实体是否匹配指定类型
        """
        if not entity_type:
            return True  # 如果没有类型限制，则匹配
        
        inferred_type = self._infer_entity_type(entity)
        return inferred_type == entity_type or entity_type in inferred_type or inferred_type in entity_type
    
    def get_triple_extraction_prompt_zh(self, text: str, identified_entities: List[str]) -> Dict[str, Any]:
        """
        获取Token感知的中文关系抽取提示
        
        Args:
            text: 输入文本
            identified_entities: 已识别的实体列表
            
        Returns:
            优化后的关系抽取提示
        """
        # 根据实体选择相关关系
        selected_relations = self.select_relations_by_entities(identified_entities)
        
        # 构建实体-关系约束
        entity_constraints = self.build_entity_relation_constraints(identified_entities, selected_relations)
        
        # 计算统计信息
        total_selected = sum(len(relations) for relations in selected_relations.values())
        total_available = len(self.schema_relations)
        
        return {
            "instruction": f"请从以下隧道工程文本中抽取三元组关系。已识别{len(identified_entities)}个实体，当前Schema包含{total_available}种关系，本次重点抽取以下{len(selected_relations)}个类别的{total_selected}种关系：",
            "input_text": text,
            "identified_entities": identified_entities,
            "relation_types": selected_relations,
            "entity_constraints": entity_constraints,
            "extraction_rules": {
                "完整性": "抽取文本中所有明确表达的关系",
                "准确性": "确保关系符合Schema定义的约束",
                "实体匹配": "只使用已识别的实体构建三元组",
                "关系验证": "验证主体和客体的类型是否符合关系定义",
                "优先级": "优先抽取与已识别实体最相关的关系"
            },
            "examples": self._generate_triple_examples(identified_entities, selected_relations),
            "output_format": {
                "description": "输出格式为JSON数组，每个三元组包含subject、predicate、object",
                "schema": {
                    "triples": [
                        {
                            "subject": "主体实体",
                            "predicate": "关系名称",
                            "object": "客体实体",
                            "confidence": "置信度(0-1)"
                        }
                    ]
                }
            },
            "optimization_info": {
                "entity_count": len(identified_entities),
                "selected_relations": total_selected,
                "total_relations": total_available,
                "selection_ratio": f"{total_selected}/{total_available} ({total_selected/total_available*100:.1f}%)",
                "token_optimized": True
            }
        }
    
    def _generate_triple_examples(self, identified_entities: List[str], 
                                selected_relations: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        生成三元组抽取示例
        """
        examples = []
        
        # 基于已识别实体和选中关系生成相关示例
        if identified_entities and selected_relations:
            # 构建示例三元组
            example_triples = []
            
            # 尝试构建一些合理的示例
            for entity in identified_entities[:3]:  # 最多使用3个实体
                entity_type = self._infer_entity_type(entity)
                
                # 根据实体类型选择合适的关系
                if entity_type == '工程结构':
                    if '包含' in str(selected_relations):
                        example_triples.append({
                            "subject": entity,
                            "predicate": "包含",
                            "object": "防水层",
                            "confidence": 0.9
                        })
                elif entity_type == '材料设备':
                    if '用于' in str(selected_relations):
                        example_triples.append({
                            "subject": entity,
                            "predicate": "用于",
                            "object": "隧道施工",
                            "confidence": 0.8
                        })
            
            if example_triples:
                examples.append({
                    "input": f"隧道二次衬砌采用{identified_entities[0] if identified_entities else 'C30混凝土'}，具有良好的防水性能。",
                    "entities": identified_entities[:3],
                    "output": {"triples": example_triples[:2]}  # 最多2个示例三元组
                })
        
        # 添加通用示例
        examples.append({
            "input": "隧道防水系统包括防水板和排水管，防水板厚度为2mm。",
            "entities": ["隧道防水系统", "防水板", "排水管", "2mm"],
            "output": {
                "triples": [
                    {
                        "subject": "隧道防水系统",
                        "predicate": "包含",
                        "object": "防水板",
                        "confidence": 0.9
                    },
                    {
                        "subject": "隧道防水系统",
                        "predicate": "包含",
                        "object": "排水管",
                        "confidence": 0.9
                    },
                    {
                        "subject": "防水板",
                        "predicate": "具有属性",
                        "object": "厚度",
                        "confidence": 0.8
                    }
                ]
            }
        })
        
        return examples
    
    def generate_multi_round_prompts(self, text: str, identified_entities: List[str]) -> List[Dict[str, Any]]:
        """
        生成多轮关系抽取提示
        
        Args:
            text: 输入文本
            identified_entities: 已识别的实体列表
            
        Returns:
            多轮提示列表
        """
        if not self.enable_multi_round:
            return [self.get_triple_extraction_prompt_zh(text, identified_entities)]
        
        all_relation_types = self.get_relation_types()
        total_relations = sum(len(relations) for relations in all_relation_types.values())
        
        # 如果关系数量不多，使用单轮
        if total_relations <= self.max_relations_per_batch:
            return [self.get_triple_extraction_prompt_zh(text, identified_entities)]
        
        # 计算需要的轮数
        rounds_needed = (total_relations + self.max_relations_per_batch - 1) // self.max_relations_per_batch
        
        prompts = []
        priorities = self.calculate_relation_priorities(identified_entities)
        
        # 按优先级分组关系
        sorted_relations = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
        
        # 分批生成提示
        relations_per_round = self.max_relations_per_batch
        for round_num in range(rounds_needed):
            start_idx = round_num * relations_per_round
            end_idx = min(start_idx + relations_per_round, len(sorted_relations))
            
            round_relations = sorted_relations[start_idx:end_idx]
            
            # 按类型重新组织这一轮的关系
            round_relation_types = defaultdict(list)
            for relation_name, _ in round_relations:
                rel_type = self._get_relation_type(relation_name)
                if rel_type:
                    round_relation_types[rel_type].append(relation_name)
            
            # 生成这一轮的提示
            round_prompt = self._generate_round_triple_prompt(
                text, identified_entities, dict(round_relation_types), 
                round_num + 1, rounds_needed
            )
            prompts.append(round_prompt)
        
        logger.info(f"生成{len(prompts)}轮关系抽取提示，总计{total_relations}个关系")
        return prompts
    
    def _generate_round_triple_prompt(self, text: str, identified_entities: List[str],
                                    relation_types: Dict[str, List[str]], 
                                    round_num: int, total_rounds: int) -> Dict[str, Any]:
        """
        生成单轮关系抽取提示
        """
        total_selected = sum(len(relations) for relations in relation_types.values())
        
        return {
            "instruction": f"第{round_num}/{total_rounds}轮关系抽取：请从隧道工程文本中抽取以下{len(relation_types)}个类别的{total_selected}种关系的三元组：",
            "input_text": text,
            "identified_entities": identified_entities,
            "relation_types": relation_types,
            "round_info": {
                "current_round": round_num,
                "total_rounds": total_rounds,
                "relations_in_round": total_selected
            },
            "extraction_rules": {
                "专注性": f"本轮只抽取指定的{total_selected}种关系",
                "完整性": "抽取文本中所有相关的关系实例",
                "准确性": "确保关系符合Schema定义",
                "实体匹配": "只使用已识别的实体构建三元组"
            },
            "output_format": {
                "description": "输出格式为JSON数组，每个三元组包含subject、predicate、object",
                "schema": {
                    "triples": [
                        {
                            "subject": "主体实体",
                            "predicate": "关系名称",
                            "object": "客体实体",
                            "confidence": "置信度(0-1)"
                        }
                    ]
                }
            }
        }
    
    def build_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        构建关系抽取提示（兼容接口）
        
        Args:
            inputs: 输入参数字典，包含 'input' 和 'entity_list' 键
            
        Returns:
            格式化的提示字符串
        """
        text = inputs.get('input', '')
        entity_list = inputs.get('entity_list', [])
        
        # 如果entity_list是字符串，按逗号分割
        if isinstance(entity_list, str):
            entities = [e.strip() for e in entity_list.split(',') if e.strip()]
        else:
            entities = entity_list
        
        return self.format_prompt(text, entities)
    
    def format_prompt(self, text: str, identified_entities: List[str]) -> str:
        """
        格式化Token感知的关系抽取提示
        
        Args:
            text: 输入文本
            identified_entities: 已识别的实体列表
            
        Returns:
            格式化的提示字符串
        """
        prompt_data = self.get_triple_extraction_prompt_zh(text, identified_entities)
        
        # 构建格式化的提示
        formatted_prompt = f"""{prompt_data['instruction']}

**输入文本：**
{prompt_data['input_text']}

**已识别实体：**
{', '.join(prompt_data['identified_entities'])}

**本次抽取的关系类型：**
"""
        
        # 添加关系类型信息
        for rel_type, relations in prompt_data['relation_types'].items():
            formatted_prompt += f"\n### {rel_type}\n"
            for relation in relations:
                if relation.endswith('等'):
                    formatted_prompt += f"- {relation}\n"
                else:
                    # 简化处理关系信息
                    chinese_name = relation
                    description = ''
                    formatted_prompt += f"- {chinese_name}: {description[:100]}\n"
        
        # 添加抽取规则
        formatted_prompt += "\n**抽取规则：**\n"
        for rule_name, rule_desc in prompt_data['extraction_rules'].items():
            formatted_prompt += f"- {rule_name}: {rule_desc}\n"
        
        # 添加示例
        if prompt_data['examples']:
            formatted_prompt += "\n**示例：**\n"
            for i, example in enumerate(prompt_data['examples'][:2], 1):  # 最多显示2个示例
                formatted_prompt += f"\n示例{i}：\n"
                formatted_prompt += f"输入：{example['input']}\n"
                formatted_prompt += f"实体：{', '.join(example['entities'])}\n"
                formatted_prompt += f"输出：{json.dumps(example['output'], ensure_ascii=False, indent=2)}\n"
        
        # 添加输出格式
        formatted_prompt += f"\n**输出格式：**\n{prompt_data['output_format']['description']}\n"
        formatted_prompt += f"\n请按照以下JSON格式输出：\n{json.dumps(prompt_data['output_format']['schema'], ensure_ascii=False, indent=2)}\n"
        
        # 添加优化信息
        opt_info = prompt_data['optimization_info']
        formatted_prompt += f"\n**Token优化信息：**\n"
        formatted_prompt += f"- 实体数量: {opt_info['entity_count']}\n"
        formatted_prompt += f"- 选中关系: {opt_info['selection_ratio']}\n"
        formatted_prompt += f"- Token优化: {'是' if opt_info['token_optimized'] else '否'}\n"
        
        return formatted_prompt
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        获取Token使用统计
        """
        total_relations = len(self.schema_relations)
        relation_types = self.get_relation_types()
        
        return {
            "total_relations": total_relations,
            "relation_types_count": len(relation_types),
            "max_relations_per_batch": self.max_relations_per_batch,
            "max_prompt_tokens": self.max_prompt_tokens,
            "estimated_rounds_needed": (total_relations + self.max_relations_per_batch - 1) // self.max_relations_per_batch,
            "multi_round_enabled": self.enable_multi_round
        }


if __name__ == "__main__":
    # 测试Token感知关系抽取
    triple_prompt = TunnelEngineeringTriplePrompt("Tunnelknowledge")
    
    test_text = "隧道二次衬砌采用C30混凝土，厚度为35cm，防水层采用2mm厚EVA防水板。"
    test_entities = ["隧道二次衬砌", "C30混凝土", "防水层", "EVA防水板", "35cm", "2mm"]
    
    print("=== Token感知关系抽取测试 ===")
    print(f"Token统计: {triple_prompt.get_token_statistics()}")
    
    # 测试单轮提示
    print("\n=== 单轮提示测试 ===")
    single_prompt = triple_prompt.format_prompt(test_text, test_entities)
    print(f"提示长度: {len(single_prompt)} 字符")
    print(f"估算token: {triple_prompt.estimate_tokens(single_prompt)}")
    
    # 测试多轮提示
    print("\n=== 多轮提示测试 ===")
    multi_prompts = triple_prompt.generate_multi_round_prompts(test_text, test_entities)
    print(f"生成{len(multi_prompts)}轮提示")
    
    for i, prompt in enumerate(multi_prompts, 1):
        relation_count = sum(len(relations) for relations in prompt["relation_types"].values())
        print(f"第{i}轮: {relation_count}个关系类型")