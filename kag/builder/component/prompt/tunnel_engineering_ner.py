#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token感知的Schema NER提示生成器

解决大规模Schema实体定义导致提示词过长、超出LLM token限制的问题。
采用智能分批、优先级排序、动态裁剪等策略优化提示长度。
"""

from typing import Dict, List, Any, Tuple, Optional
from kag.common.registry import Registrable
from .schema_aware_prompt import SchemaAwarePromptBase
import json
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@Registrable.register("tunnel_engineering_ner")
class TunnelEngineeringNERPrompt(SchemaAwarePromptBase):
    """
    隧道工程专业NER提示生成器
    
    特点：
    1. 智能token计算和管理
    2. 基于相关性的实体优先级排序
    3. 动态实体分批处理
    4. 上下文感知的实体选择
    5. 多轮抽取策略支持
    """
    
    # 定义NER专用模板
    template_zh = {
        "instruction": "请从以下隧道工程文本中识别出所有相关的专业实体。",
        "entity_types": "$entity_types",
        "recognition_rules": "$recognition_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "请按照JSON格式输出识别结果，格式为：{\"实体类型\": [\"实体1\", \"实体2\"]}"
    }
    
    template_en = {
        "instruction": "Please identify all relevant professional entities from the following tunnel engineering text.",
        "entity_types": "$entity_types",
        "recognition_rules": "$recognition_rules",
        "examples": "$examples",
        "input": "$input",
        "output_format": "Please output the recognition results in JSON format: {\"Entity Type\": [\"Entity1\", \"Entity2\"]}"
    }
    
    @property
    def template_variables(self) -> List[str]:
        """返回模板变量列表"""
        return ["entity_types", "recognition_rules", "examples", "input"]
    
    def __init__(self, 
                 project_id: str = None,
                 host_addr: str = None,
                 schema=None,
                 max_prompt_tokens: int = 3000,
                 max_entities_per_batch: int = 50,
                 min_entities_per_category: int = 3,
                 enable_multi_round: bool = True,
                 **kwargs):
        """
        初始化Token感知NER提示生成器
        
        Args:
            project_id: 项目ID
            host_addr: OpenSPG服务器地址
            schema: 已加载的Schema对象
            max_prompt_tokens: 最大提示token数
            max_entities_per_batch: 每批最大实体数
            min_entities_per_category: 每个类别最少实体数
            enable_multi_round: 是否启用多轮抽取
        """
        super().__init__(project_id=project_id, host_addr=host_addr, schema=schema, **kwargs)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_entities_per_batch = max_entities_per_batch
        self.min_entities_per_category = min_entities_per_category
        self.enable_multi_round = enable_multi_round
        
        # Token计算相关
        self.avg_chars_per_token = 2.5  # 中文平均字符/token比例
        self.base_prompt_tokens = 500   # 基础提示模板的token数
        
        # 实体优先级缓存
        self._entity_priorities = None
        self._context_keywords = set()
        
        logger.info(f"隧道工程NER初始化: max_tokens={max_prompt_tokens}, max_entities={max_entities_per_batch}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数
        """
        # 简单的token估算：中文按字符数/2.5，英文按单词数*1.3
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(len(word) for word in re.findall(r'[a-zA-Z]+', text))
        
        estimated_tokens = int(
            chinese_chars / self.avg_chars_per_token +
            english_words * 1.3 +
            other_chars / 4
        )
        
        return max(estimated_tokens, len(text) // 4)  # 最少按1/4字符数计算
    
    def calculate_entity_priorities(self, context_text: str = "") -> Dict[str, float]:
        """
        计算实体优先级
        
        Args:
            context_text: 上下文文本，用于计算相关性
            
        Returns:
            实体名称到优先级的映射
        """
        if self._entity_priorities is not None and not context_text:
            return self._entity_priorities
        
        priorities = {}
        
        # 提取上下文关键词
        if context_text:
            self._context_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', context_text))
        
        for entity_name, entity_info in self.schema_entities.items():
            priority = 1.0  # 基础优先级
            
            # 1. 基于实体类型的优先级
            entity_type = entity_info.get('type', '')
            if entity_type in ['TunnelSegment', 'ConstructionActivity', 'TreatmentMethod']:
                priority += 0.5  # 核心实体类型
            elif entity_type in ['LogisticsSupportMaterial', 'CommunicationEquipment']:
                priority += 0.3  # 重要实体类型
            
            # 2. 基于实体名称长度（短名称优先）
            chinese_name = entity_info.get('chinese_name', entity_name)
            if len(chinese_name) <= 4:
                priority += 0.2
            elif len(chinese_name) >= 8:
                priority -= 0.1
            
            # 3. 基于上下文相关性
            if self._context_keywords:
                # 检查实体名称是否在上下文中出现
                if chinese_name in context_text or entity_name in context_text:
                    priority += 1.0  # 直接匹配
                
                # 检查相关关键词
                entity_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', chinese_name))
                overlap = len(entity_keywords & self._context_keywords)
                if overlap > 0:
                    priority += 0.3 * overlap
            
            # 4. 基于实体属性数量（属性多的实体更重要）
            properties = entity_info.get('properties', {})
            if len(properties) > 3:
                priority += 0.2
            
            priorities[entity_name] = priority
        
        # 缓存结果
        if not context_text:
            self._entity_priorities = priorities
        
        return priorities
    
    def select_entities_by_token_limit(self, 
                                     context_text: str = "",
                                     target_tokens: int = None) -> Dict[str, List[str]]:
        """
        根据token限制选择实体
        
        Args:
            context_text: 上下文文本
            target_tokens: 目标token数，默认使用配置值
            
        Returns:
            按类别分组的选中实体
        """
        if target_tokens is None:
            target_tokens = self.max_prompt_tokens - self.base_prompt_tokens
        
        logger.info(f"开始Token感知实体选择，目标token数: {target_tokens}")
        
        # 计算实体优先级
        priorities = self.calculate_entity_priorities(context_text)
        
        # 按优先级排序实体
        sorted_entities = sorted(
            priorities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 按类别分组
        entity_types = self.get_entity_types()
        logger.info(f"获取到实体类型: {len(entity_types)}个类别")
        
        if not entity_types:
            logger.warning("没有可用的实体类型")
            return {}
        
        # 检查实体数量
        total_available = 0
        for category, entities in entity_types.items():
            entity_count = len(entities)
            total_available += entity_count
            logger.info(f"类别 '{category}': {entity_count}个实体")
        
        logger.info(f"总可用实体数: {total_available}")
        
        if total_available == 0:
            logger.warning("实体数量为0，可能Schema解析失败")
            return {}
        # 简化实体选择逻辑，直接从所有实体中选择
        selected_entities = []
        current_tokens = 0
        
        # 获取所有实体（现在只有一个类别）
        all_entities = []
        for category, entities in entity_types.items():
            all_entities.extend(entities)
        
        # 为所有实体分配优先级，没有优先级的实体使用默认值1.0
        available_entities = all_entities.copy()
        available_entities.sort(key=lambda x: priorities.get(x, 1.0), reverse=True)
        
        logger.info(f"可用实体数: {len(available_entities)}，开始按优先级选择")
        
        # 按优先级和token限制选择实体
        for entity in available_entities:
            entity_tokens = self.estimate_tokens(entity) + 2  # +2 for formatting
            
            if current_tokens + entity_tokens <= target_tokens and len(selected_entities) < self.max_entities_per_batch:
                selected_entities.append(entity)
                current_tokens += entity_tokens
                logger.debug(f"选择实体: {entity}, 当前token: {current_tokens}")
            else:
                break  # token已满或实体数量已满
        
        # 构建返回结果
        result = {"所有实体": selected_entities}
        
        # 如果还有更多实体，添加省略标记
        total_entities = len(all_entities)
        if len(selected_entities) < total_entities:
            result["所有实体"].append(f"...等（还有{total_entities - len(selected_entities)}个）")
        
        logger.info(f"Token感知实体选择: 选中{len(selected_entities)}个实体，预估{current_tokens}个token")
        return result
    
    def _get_entity_category(self, entity_name: str) -> Optional[str]:
        """
        获取实体所属类别
        """
        entity_types = self.get_entity_types()
        for category, entities in entity_types.items():
            if entity_name in entities:
                return category
        return None
    
    def generate_multi_round_prompts(self, context_text: str) -> List[Dict[str, Any]]:
        """
        生成多轮抽取提示
        
        Args:
            context_text: 上下文文本
            
        Returns:
            多轮提示列表
        """
        if not self.enable_multi_round:
            return [self.get_ner_prompt_zh(context_text)]
        
        all_entity_types = self.get_entity_types()
        total_entities = sum(len(entities) for entities in all_entity_types.values())
        
        # 如果实体数量不多，使用单轮
        if total_entities <= self.max_entities_per_batch:
            return [self.get_ner_prompt_zh(context_text)]
        
        # 计算需要的轮数
        rounds_needed = (total_entities + self.max_entities_per_batch - 1) // self.max_entities_per_batch
        
        prompts = []
        priorities = self.calculate_entity_priorities(context_text)
        
        # 按优先级分组实体
        sorted_entities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
        
        for round_idx in range(rounds_needed):
            start_idx = round_idx * self.max_entities_per_batch
            end_idx = min(start_idx + self.max_entities_per_batch, len(sorted_entities))
            
            round_entities = sorted_entities[start_idx:end_idx]
            
            # 按类别重新组织
            round_entity_types = defaultdict(list)
            for entity_name, _ in round_entities:
                category = self._get_entity_category(entity_name)
                if category:
                    round_entity_types[category].append(entity_name)
            
            # 生成该轮的提示
            prompt = self._generate_round_prompt(dict(round_entity_types), round_idx + 1, rounds_needed)
            prompts.append(prompt)
        
        logger.info(f"生成{len(prompts)}轮抽取提示")
        return prompts
    
    def parse_response(self, response: str, **kwargs):
        """
        解析NER响应
        
        Args:
            response: LLM响应字符串
            **kwargs: 其他参数
            
        Returns:
            解析后的实体列表或字典
        """
        # 添加调试日志：打印LLM原始响应
        logger.info(f"=== LLM响应调试 ===")
        logger.info(f"响应类型: {type(response)}")
        logger.info(f"响应长度: {len(str(response))} 字符")
        logger.info(f"原始响应内容:\n{response}")
        
        try:
            # 如果响应是字符串，尝试解析为JSON
            if isinstance(response, str):
                # 查找JSON部分
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    logger.info(f"提取的JSON字符串:\n{json_str}")
                    response = json.loads(json_str)
                else:
                    # 如果没有JSON格式，返回空字典
                    logger.warning(f"无法解析NER响应: {response[:100]}...")
                    return {}
            
            # 处理不同的响应格式
            if isinstance(response, dict):
                if "output" in response:
                    response = response["output"]
                if "named_entities" in response:
                    parsed_result = response["named_entities"]
                else:
                    parsed_result = response
            else:
                parsed_result = response
            
            # 打印解析结果
            logger.info(f"解析结果类型: {type(parsed_result)}")
            logger.info(f"解析结果内容: {parsed_result}")
            if isinstance(parsed_result, dict):
                total_entities = sum(len(entities) if isinstance(entities, list) else 0 for entities in parsed_result.values())
                logger.info(f"解析出的实体总数: {total_entities}")
            logger.info(f"=== 响应解析结束 ===")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"解析NER响应时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            return {}
    
    def _generate_round_prompt(self, entity_types: Dict[str, List[str]], 
                              round_num: int, total_rounds: int) -> Dict[str, Any]:
        """
        生成单轮提示
        """
        return {
            "instruction": f"请从以下隧道防排水工程文本中识别出相关的专业实体。这是第{round_num}轮抽取（共{total_rounds}轮），重点识别以下类别的实体：",
            "entity_types": entity_types,
            "round_info": {
                "current_round": round_num,
                "total_rounds": total_rounds,
                "focus": "本轮重点关注以下实体类型" if round_num == 1 else "继续识别剩余实体类型"
            },
            "recognition_rules": {
                "完整性": "识别文本中所有出现的专业实体，包括同义词和变体",
                "准确性": "确保识别的实体属于当前轮次指定的类型",
                "标准化": "使用Schema中定义的标准名称",
                "去重": "避免重复识别前几轮已识别的实体" if round_num > 1 else "首轮识别，注意实体完整性"
            },
            "output_format": {
                "description": "输出格式为JSON，按实体类型分组",
                "schema": {category: "List[str]" for category in entity_types.keys()}
            }
        }
    
    def get_ner_prompt_zh(self, context_text: str = "") -> Dict[str, Any]:
        """
        获取Token感知的中文NER提示
        
        Args:
            context_text: 上下文文本，用于优化实体选择
            
        Returns:
            优化后的NER提示
        """
        # 根据token限制选择实体
        selected_entities = self.select_entities_by_token_limit(context_text)
        
        # 计算统计信息
        total_selected = sum(len(entities) for entities in selected_entities.values())
        total_available = len(self.schema_entities)
        
        return {
            "instruction": f"请从以下隧道防排水工程文本中识别出所有相关的专业实体。当前Schema包含{total_available}种实体类型，本次重点识别以下{len(selected_entities)}个类别的{total_selected}种实体：",
            "entity_types": selected_entities,
            "recognition_rules": {
                "完整性": "识别文本中所有出现的专业实体，包括同义词和变体",
                "准确性": "确保识别的实体属于Schema定义的类型",
                "标准化": "使用Schema中定义的标准名称",
                "上下文": "考虑实体在隧道工程上下文中的含义",
                "优先级": "优先识别与上下文最相关的实体"
            },
            "examples": self._generate_context_examples(context_text),
            "output_format": {
                "description": "输出格式为JSON，按实体类型分组",
                "schema": {category: "List[str]" for category in selected_entities.keys()}
            },
            "token_info": {
                "selected_entities": total_selected,
                "total_entities": total_available,
                "selection_ratio": f"{total_selected}/{total_available} ({total_selected/total_available*100:.1f}%)" if total_available > 0 else "0/0 (0.0%)",
                "multi_round_available": self.enable_multi_round and total_available > self.max_entities_per_batch
            }
        }
    
    def _generate_context_examples(self, context_text: str) -> List[Dict[str, Any]]:
        """
        基于上下文生成相关示例
        """
        examples = []
        
        # 如果有上下文，生成相关示例
        if context_text and len(context_text) > 20:
            # 提取上下文中的关键词
            keywords = re.findall(r'[\u4e00-\u9fff]{2,}', context_text[:100])
            
            if '混凝土' in keywords or '衬砌' in keywords:
                examples.append({
                    "input": "隧道二次衬砌采用C30混凝土，厚度为35cm。",
                    "output": {
                        "工程结构": ["隧道", "二次衬砌"],
                        "材料设备": ["C30混凝土"],
                        "技术参数": ["厚度", "35cm"]
                    }
                })
            
            if '防水' in keywords or '排水' in keywords:
                examples.append({
                    "input": "防水层采用2mm厚EVA防水板，排水系统包括纵向排水管。",
                    "output": {
                        "工程结构": ["防水层", "排水系统"],
                        "材料设备": ["EVA防水板", "纵向排水管"],
                        "技术参数": ["2mm厚"]
                    }
                })
        
        # 如果没有生成上下文相关示例，使用默认示例
        if not examples:
            examples.append({
                "input": "隧道施工采用新奥法，支护结构包括钢拱架和喷射混凝土。",
                "output": {
                    "施工工艺": ["新奥法"],
                    "工程结构": ["支护结构"],
                    "材料设备": ["钢拱架", "喷射混凝土"]
                }
            })
        
        return examples
    
    def build_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        构建NER提示（兼容接口）
        
        Args:
            inputs: 输入参数字典，包含 'input' 键
            
        Returns:
            格式化的提示字符串
        """
        text = inputs.get('input', '')
        return self.format_prompt(text)
    
    def format_prompt(self, text: str, language: str = "zh") -> str:
        """
        格式化提示，支持Token感知
        
        Args:
            text: 输入文本
            language: 语言
            
        Returns:
            格式化的提示字符串
        """
        if language == "zh":
            prompt_data = self.get_ner_prompt_zh(text)
        else:
            # 英文版本可以类似实现
            prompt_data = self.get_ner_prompt_zh(text)  # 暂时使用中文版本
        
        # 构建提示字符串
        prompt_parts = [
            prompt_data["instruction"],
            "\n实体类型定义："
        ]
        
        # 添加实体类型
        for category, entities in prompt_data["entity_types"].items():
            prompt_parts.append(f"\n{category}：{', '.join(entities)}")
        
        # 添加识别规则
        prompt_parts.append("\n\n识别规则：")
        for rule, description in prompt_data["recognition_rules"].items():
            prompt_parts.append(f"- {rule}：{description}")
        
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
        
        # 添加待处理文本
        prompt_parts.append(f"\n\n请处理以下文本：\n{text}")
        
        final_prompt = "\n".join(prompt_parts)
        
        # 检查token数量
        estimated_tokens = self.estimate_tokens(final_prompt)
        if estimated_tokens > self.max_prompt_tokens:
            logger.warning(f"提示token数量({estimated_tokens})超过限制({self.max_prompt_tokens})")
        
        # 添加调试日志：打印完整的提示词
        logger.info(f"=== NER提示词调试 ===")
        logger.info(f"提示词长度: {len(final_prompt)} 字符")
        logger.info(f"估算token数: {estimated_tokens}")
        logger.info(f"完整提示词内容:\n{final_prompt}")
        logger.info(f"=== 提示词结束 ===")
        
        return final_prompt
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        获取token统计信息
        """
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "max_entities_per_batch": self.max_entities_per_batch,
            "total_entities": len(self.schema_entities),
            "avg_chars_per_token": self.avg_chars_per_token,
            "multi_round_enabled": self.enable_multi_round,
            "estimated_rounds_needed": (len(self.schema_entities) + self.max_entities_per_batch - 1) // self.max_entities_per_batch
        }


if __name__ == "__main__":
    # 测试Token感知NER
    ner_prompt = TunnelEngineeringNERPrompt("Tunnelknowledge")
    
    test_text = "隧道二次衬砌采用C30混凝土，厚度为35cm，防水层采用2mm厚EVA防水板，排水系统包括纵向排水管和横向排水管。施工过程中需要注意安全管理和质量控制。"
    
    print("=== Token感知NER测试 ===")
    print(f"Token统计: {ner_prompt.get_token_statistics()}")
    
    print("\n=== 单轮提示测试 ===")
    single_prompt = ner_prompt.format_prompt(test_text)
    print(f"提示长度: {len(single_prompt)} 字符")
    print(f"估算token: {ner_prompt.estimate_tokens(single_prompt)}")
    
    print("\n=== 多轮提示测试 ===")
    multi_prompts = ner_prompt.generate_multi_round_prompts(test_text)
    print(f"生成{len(multi_prompts)}轮提示")
    
    for i, prompt in enumerate(multi_prompts, 1):
        entity_count = sum(len(entities) for entities in prompt["entity_types"].values())
        print(f"第{i}轮: {entity_count}个实体类型")