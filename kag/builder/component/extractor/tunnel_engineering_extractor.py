#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隧道工程专业知识抽取器

基于 schema_free_extractor 架构，使用 init_prompt_with_fallback 机制
支持隧道工程专业的NER、实体标准化和关系抽取功能。
"""

import logging
from typing import Dict, List, Any, Optional
from kag.builder.model.chunk import Chunk
from kag.builder.model.sub_graph import SubGraph, Node, Edge
from kag.interface import ExtractorABC, LLMClient, PromptABC
from kag.interface.common.model.chunk import ChunkTypeEnum
from knext.common.base.runnable import Input, Output
from knext.schema.client import SchemaClient
from kag.builder.prompt.utils import init_prompt_with_fallback
from kag.builder.component.prompt.tunnel_engineering_ner import TunnelEngineeringNERPrompt
from kag.builder.component.prompt.tunnel_engineering_std import TunnelEngineeringStandardizationPrompt
from kag.builder.component.prompt.tunnel_engineering_triple import TunnelEngineeringTriplePrompt
import json

logger = logging.getLogger(__name__)


@ExtractorABC.register("tunnel_engineering_extractor")
class TunnelEngineeringExtractor(ExtractorABC):
    """
    隧道工程专业知识抽取器
    
    特点：
    1. 智能Token管理，避免超出LLM限制
    2. 基于上下文的动态Schema选择
    3. 多轮抽取策略支持
    4. 实体相关性优化
    5. 性能监控和统计
    """
    
    def __init__(self,
                 llm: LLMClient = None,
                 ner_prompt=None,
                 std_prompt=None,
                 triple_prompt=None,
                 **kwargs):
        """
        初始化隧道工程专业抽取器
        
        Args:
            llm: LLM客户端实例（必需）
            ner_prompt: NER提示组件（可选）
            std_prompt: 标准化提示组件（可选）
            triple_prompt: 关系抽取提示组件（可选）
        """
        super().__init__(**kwargs)
        
        # 处理LLM客户端 - 必须通过参数传入
        if llm is None:
            logger.error("LLM初始化失败: llm参数为None")
            raise ValueError("llm parameter is required. Please provide LLMClient instance.")
        
        self.llm = llm
        logger.info(f"LLM初始化成功: 类型={type(llm)}, 有invoke方法={hasattr(llm, 'invoke')}, invoke可调用={callable(getattr(llm, 'invoke', None))}")
        
        # 测试LLM连接
        try:
            if hasattr(llm, 'invoke') and callable(getattr(llm, 'invoke', None)):
                logger.info("LLM连接测试: invoke方法可用")
                # 进行真实的LLM调用测试
                try:
                    from kag.interface import PromptABC
                    # 创建一个简单的测试提示
                    class SimpleTestPrompt(PromptABC):
                        def build_prompt(self, variables, **kwargs):
                            return "请回复'Hello'来确认连接正常。"
                        
                        def parse_response(self, response, **kwargs):
                            return response
                    
                    test_prompt = SimpleTestPrompt()
                    test_response = llm.invoke({"input": "test"}, test_prompt)
                    logger.info(f"LLM真实调用测试成功: 响应类型={type(test_response)}")
                    if hasattr(test_response, 'content'):
                        logger.info(f"LLM响应内容: {str(test_response.content)[:100]}...")
                    else:
                        logger.info(f"LLM响应: {str(test_response)[:100]}...")
                except Exception as real_test_error:
                    logger.warning(f"LLM真实调用测试失败: {str(real_test_error)}")
            else:
                logger.warning(f"LLM连接测试: invoke方法不可用，LLM类型={type(llm)}")
        except Exception as test_error:
            logger.warning(f"LLM连接测试失败: {str(test_error)}")
        
        # 从kag_project_config获取项目配置信息（由基类初始化）
        # 从服务器加载Schema
        self.schema = SchemaClient(
            host_addr=self.kag_project_config.host_addr,
            project_id=self.kag_project_config.project_id,
        ).load()
        
        # 输出Schema信息
        if self.schema and isinstance(self.schema, dict) and len(self.schema) > 0:
            logger.info(f"Schema加载完成: 共{len(self.schema)}个实体类型")
        else:
            logger.warning("Schema加载失败或无实体类型定义")
        
        biz_scene = self.kag_project_config.biz_scene
        project_id = self.kag_project_config.project_id
        host_addr = self.kag_project_config.host_addr
        
        # 初始化提示模板 - 检查是否为ConfigTree对象，如果是则重新初始化
        from pyhocon.config_tree import ConfigTree
        
        # 处理NER提示
        if ner_prompt is None or isinstance(ner_prompt, ConfigTree):
            try:
                self.ner_prompt = TunnelEngineeringNERPrompt(
                    project_id=project_id,
                    host_addr=host_addr,
                    schema=self.schema
                )
                logger.info(f"成功初始化自定义NER提示模板: 类型={type(self.ner_prompt)}, 有build_prompt方法={hasattr(self.ner_prompt, 'build_prompt')}")
            except Exception as e:
                logger.error(f"自定义NER提示初始化失败: {e}")
                # 如果自定义提示初始化失败，尝试使用注册机制
                try:
                    from kag.interface import PromptABC
                    self.ner_prompt = PromptABC.from_config({"type": "tunnel_engineering_ner"})
                    logger.info(f"通过注册机制初始化NER提示成功: 类型={type(self.ner_prompt)}")
                except Exception as fallback_error:
                    logger.error(f"注册机制初始化NER提示也失败: {fallback_error}")
                    # 最后的回退选项
                    self.ner_prompt = init_prompt_with_fallback("ner", biz_scene)
                    logger.warning(f"使用默认回退NER提示: 类型={type(self.ner_prompt)}")
        else:
            self.ner_prompt = ner_prompt
            logger.info(f"使用传入的NER提示: 类型={type(self.ner_prompt)}")
        
        # 处理标准化提示
        if std_prompt is None or isinstance(std_prompt, ConfigTree):
            try:
                self.std_prompt = TunnelEngineeringStandardizationPrompt(
                    project_id=project_id,
                    host_addr=host_addr,
                    schema=self.schema
                )
                logger.info(f"成功初始化自定义标准化提示模板: 类型={type(self.std_prompt)}, 有build_prompt方法={hasattr(self.std_prompt, 'build_prompt')}")
            except Exception as e:
                logger.error(f"自定义标准化提示初始化失败: {e}")
                # 如果自定义提示初始化失败，尝试使用注册机制
                try:
                    from kag.interface import PromptABC
                    self.std_prompt = PromptABC.from_config({"type": "tunnel_engineering_std"})
                    logger.info(f"通过注册机制初始化标准化提示成功: 类型={type(self.std_prompt)}")
                except Exception as fallback_error:
                    logger.error(f"注册机制初始化标准化提示也失败: {fallback_error}")
                    # 最后的回退选项
                    self.std_prompt = init_prompt_with_fallback("std", biz_scene)
                    logger.warning(f"使用默认回退标准化提示: 类型={type(self.std_prompt)}")
        else:
            self.std_prompt = std_prompt
            logger.info(f"使用传入的标准化提示: 类型={type(self.std_prompt)}")
        
        # 处理三元组提示
        if triple_prompt is None or isinstance(triple_prompt, ConfigTree):
            try:
                self.triple_prompt = TunnelEngineeringTriplePrompt(
                    project_id=project_id,
                    host_addr=host_addr,
                    schema=self.schema
                )
                logger.info(f"成功初始化自定义三元组提示模板: 类型={type(self.triple_prompt)}, 有build_prompt方法={hasattr(self.triple_prompt, 'build_prompt')}")
            except Exception as e:
                logger.error(f"自定义三元组提示初始化失败: {e}")
                # 如果自定义提示初始化失败，尝试使用注册机制
                try:
                    from kag.interface import PromptABC
                    self.triple_prompt = PromptABC.from_config({"type": "tunnel_engineering_triple"})
                    logger.info(f"通过注册机制初始化三元组提示成功: 类型={type(self.triple_prompt)}")
                except Exception as fallback_error:
                    logger.error(f"注册机制初始化三元组提示也失败: {fallback_error}")
                    # 最后的回退选项
                    self.triple_prompt = init_prompt_with_fallback("triple", biz_scene)
                    logger.warning(f"使用默认回退三元组提示: 类型={type(self.triple_prompt)}")
        else:
            self.triple_prompt = triple_prompt
            logger.info(f"使用传入的三元组提示: 类型={type(self.triple_prompt)}")
        
        # 验证所有提示词是否正确初始化
        self._validate_prompts()
        
        # 性能统计
        self.extraction_stats = {
            'total_extractions': 0,
            'avg_entities_per_text': 0,
            'avg_relations_per_text': 0
        }
        
        logger.info(f"隧道工程抽取器初始化完成: project_id={self.kag_project_config.project_id}")
    
    def _validate_prompts(self):
        """验证所有提示词是否正确初始化"""
        from pyhocon.config_tree import ConfigTree
        
        prompts_to_check = [
            ("ner_prompt", self.ner_prompt),
            ("std_prompt", self.std_prompt),
            ("triple_prompt", self.triple_prompt)
        ]
        
        for prompt_name, prompt_obj in prompts_to_check:
            if isinstance(prompt_obj, ConfigTree):
                logger.error(f"错误：{prompt_name} 仍然是 ConfigTree 对象: {prompt_obj}")
                raise ValueError(f"{prompt_name} 初始化失败，仍然是 ConfigTree 对象而不是提示类实例")
            elif not hasattr(prompt_obj, 'build_prompt'):
                logger.error(f"错误：{prompt_name} 没有 build_prompt 方法: 类型={type(prompt_obj)}")
                raise ValueError(f"{prompt_name} 缺少必需的 build_prompt 方法")
            else:
                logger.info(f"✓ {prompt_name} 验证通过: 类型={type(prompt_obj)}, 有build_prompt方法={hasattr(prompt_obj, 'build_prompt')}")
    
    @property
    def entity_types(self) -> Dict[str, List[str]]:
        """获取实体类型定义"""
        if hasattr(self.ner_prompt, 'get_entity_types'):
            return self.ner_prompt.get_entity_types()
        return {}
    
    @property
    def relation_types(self) -> Dict[str, List[str]]:
        """获取关系类型定义"""
        if hasattr(self.triple_prompt, 'get_relation_types'):
            return self.triple_prompt.get_relation_types()
        return {}
    
    @property
    def input_types(self):
        return Chunk

    @property
    def output_types(self):
        return SubGraph

    @property
    def inherit_input_key(self):
        return True

    @staticmethod
    def output_indices() -> List[str]:
        return ["TunnelEngineering"]
    
    def _invoke(self, input: Input, **kwargs) -> List[Output]:
        """
        处理输入数据并生成子图
        
        Args:
            input: 输入数据
            **kwargs: 其他参数
            
        Returns:
            包含子图信息的输出列表
        """
        try:
            self.extraction_stats['total_extractions'] += 1
            
            content = input.content
            logger.info(f"开始隧道工程知识抽取，文本长度: {len(content)}")
            
            # 1. 实体识别
            entities = self._extract_entities_token_aware(content)
            
            # 2. 实体标准化
            standardized_entities = self._standardize_entities_token_aware(entities, content)
            
            # 3. 关系抽取
            relations = self._extract_relations_token_aware(content, list(standardized_entities.keys()))
            
            # 4. 构建子图
            subgraph = self._build_subgraph(standardized_entities, relations, input)
            
            # 5. 更新统计信息
            self._update_extraction_stats(entities, relations)
            
            logger.info(f"隧道工程抽取完成: 实体{len(standardized_entities)}个, 关系{len(relations)}个")
            
            return [subgraph]
            
        except Exception as e:
            import traceback
            logger.error(f"Token感知知识抽取失败: {str(e)}")
            logger.error(f"完整异常堆栈: {traceback.format_exc()}")
            # 返回空子图
            empty_subgraph = SubGraph([], [])
            return [empty_subgraph]
    
    def _extract_entities_token_aware(self, text: str) -> Dict[str, List[str]]:
        """
        隧道工程专业实体抽取
        
        Args:
            text: 输入文本
            
        Returns:
            实体抽取结果
        """
        try:
            # 详细的LLM可用性检查和诊断
            if self.llm is None:
                logger.warning("LLM不可用: LLM实例为None")
                entities = self._simple_entity_extraction(text)
            elif not hasattr(self.llm, 'invoke'):
                logger.warning(f"LLM不可用: LLM实例({type(self.llm)})没有invoke方法")
                entities = self._simple_entity_extraction(text)
            elif not callable(getattr(self.llm, 'invoke', None)):
                logger.warning(f"LLM不可用: LLM实例({type(self.llm)})的invoke方法不可调用")
                entities = self._simple_entity_extraction(text)
            else:
                try:
                    logger.debug(f"尝试调用LLM，实例类型: {type(self.llm)}")
                    
                    # 构建NER提示词
                    try:
                        prompt_content = self.ner_prompt.build_prompt({"input": text})
                        logger.debug(f"NER提示词构建成功，长度: {len(prompt_content)}")
                    except Exception as prompt_error:
                        logger.error(f"构建NER提示词失败: {prompt_error}")
                        entities = self._simple_entity_extraction(text)
                        return entities
                    
                    # 使用标准的LLM调用方式：invoke(data_dict, prompt)
                    response = self.llm.invoke({"input": text}, self.ner_prompt)
                    logger.debug(f"LLM调用成功，响应类型: {type(response)}")
                    entities = self._parse_ner_response(response)
                except Exception as llm_error:
                    import traceback
                    logger.error(f"LLM调用失败，使用简单实体抽取: {str(llm_error)}")
                    logger.error(f"LLM调用详细异常堆栈: {traceback.format_exc()}")
                    logger.error(f"LLM实例详细信息: {type(self.llm)}, 属性: {dir(self.llm)}")
                    logger.error(f"提示词实例详细信息: {type(self.ner_prompt)}, 属性: {dir(self.ner_prompt)}")
                    entities = self._simple_entity_extraction(text)
            
            return entities
            
        except Exception as e:
            import traceback
            logger.error(f"隧道工程NER抽取失败: {str(e)}")
            logger.error(f"完整异常堆栈: {traceback.format_exc()}")
            return {}
    

    
    def _simple_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """
        简单的实体抽取（关键词匹配）
        """
        entities = {}
        
        for category, entity_list in self.entity_types.items():
            found_entities = []
            for entity in entity_list:
                if entity in text:
                    found_entities.append(entity)
            if found_entities:
                entities[category] = found_entities
        
        return entities
    
    def _standardize_entities_token_aware(self, entities: Dict[str, List[str]], text: str = "") -> Dict[str, str]:
        """
        隧道工程专业实体标准化
        
        Args:
            entities: 原始实体字典
            
        Returns:
            标准化后的实体映射
        """
        try:
            logger.info(f"=== 实体标准化调试 ===")
            logger.info(f"输入实体字典: {entities}")
            
            # 展平实体列表
            flat_entities = []
            for category, entity_list in entities.items():
                flat_entities.extend(entity_list)
            
            logger.info(f"展平后的实体列表: {flat_entities}")
            
            if not flat_entities:
                logger.info(f"没有实体需要标准化，返回空字典")
                return {}
            
            # 使用标准化提示组件 - 详细的LLM可用性检查
            if self.llm is None:
                logger.warning("实体标准化: LLM实例为None，使用原始实体")
                standardized = {entity: entity for entity in flat_entities}

            elif not hasattr(self.llm, 'invoke'):
                logger.warning(f"实体标准化: LLM实例({type(self.llm)})没有invoke方法")
                standardized = {entity: entity for entity in flat_entities}
            elif not callable(getattr(self.llm, 'invoke', None)):
                logger.warning(f"实体标准化: LLM实例({type(self.llm)})的invoke方法不可调用")
                standardized = {entity: entity for entity in flat_entities}
            else:
                try:
                    # 构造named_entities格式
                    named_entities = [{"name": entity, "category": "Unknown"} for entity in flat_entities]
                    
                    logger.debug(f"实体标准化: 尝试调用LLM，实例类型: {type(self.llm)}")
                    
                    # 构建标准化提示词
                    try:
                        prompt_content = self.std_prompt.build_prompt({"input": text, "named_entities": entities})
                        logger.debug(f"标准化提示词构建成功，长度: {len(prompt_content)}")
                    except Exception as prompt_error:
                        logger.error(f"构建标准化提示词失败: {prompt_error}")
                        standardized = {entity: entity for entity in flat_entities}
                        return standardized
                    
                    # 使用标准的LLM调用方式：invoke(data_dict, prompt)
                    response = self.llm.invoke({"input": text, "named_entities": entities}, self.std_prompt)
                    logger.debug(f"实体标准化: LLM调用成功，响应类型: {type(response)}")
                    
                    # 解析标准化结果
                    standardized = self._parse_standardization_response(response, flat_entities)
                except Exception as llm_error:
                    import traceback
                    logger.error(f"实体标准化: LLM调用失败，使用原始实体: {str(llm_error)}")
                    logger.error(f"实体标准化: LLM调用详细异常堆栈: {traceback.format_exc()}")
                    logger.error(f"标准化LLM实例详细信息: {type(self.llm)}, 属性: {dir(self.llm)}")
                    logger.error(f"标准化提示词实例详细信息: {type(self.std_prompt)}, 属性: {dir(self.std_prompt)}")
                    standardized = {entity: entity for entity in flat_entities}
            
            logger.info(f"标准化结果: {standardized}")
            logger.info(f"标准化实体数量: {len(standardized)}")
            logger.info(f"=== 实体标准化结束 ===")
            return standardized
            
        except Exception as e:
            logger.error(f"隧道工程实体标准化失败: {str(e)}")
            # 返回原始实体作为备用
            flat_entities = []
            for category, entity_list in entities.items():
                flat_entities.extend(entity_list)
            return {entity: entity for entity in flat_entities}
    
    def _extract_relations_token_aware(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        隧道工程专业关系抽取
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表
            
        Returns:
            关系抽取结果
        """
        try:
            if len(entities) < 2:
                return []
            
            # 使用关系抽取提示组件 - 详细的LLM可用性检查
            if self.llm is None:
                logger.warning("关系抽取: LLM实例为None，使用简单关系抽取")
                relations = self._simple_relation_extraction(text, entities)

            elif not hasattr(self.llm, 'invoke'):
                logger.warning(f"关系抽取: LLM实例({type(self.llm)})没有invoke方法")
                relations = self._simple_relation_extraction(text, entities)
            elif not callable(getattr(self.llm, 'invoke', None)):
                logger.warning(f"关系抽取: LLM实例({type(self.llm)})的invoke方法不可调用")
                relations = self._simple_relation_extraction(text, entities)
            else:
                try:
                    # 构造entity_list格式
                    entity_list = [{"name": entity, "category": "Unknown"} for entity in entities]
                    
                    logger.debug(f"关系抽取: 尝试调用LLM，实例类型: {type(self.llm)}")
                    
                    # 构建关系抽取提示词
                    try:
                        print(f"=== 关系抽取调试 ===")
                        print(f"entities类型: {type(entities)}")
                        print(f"entities内容: {entities}")
                        print(f"entities长度: {len(entities) if hasattr(entities, '__len__') else 'N/A'}")
                        
                        prompt_content = self.triple_prompt.build_prompt({"input": text, "entity_list": entities})
                        logger.debug(f"关系抽取提示词构建成功，长度: {len(prompt_content)}")
                    except Exception as prompt_error:
                        logger.error(f"构建关系抽取提示词失败: {prompt_error}")
                        import traceback
                        logger.error(f"完整异常堆栈: {traceback.format_exc()}")
                        relations = self._simple_relation_extraction(text, entities)
                        return relations
                    
                    # 使用标准的LLM调用方式：invoke(data_dict, prompt)
                    response = self.llm.invoke({"input": text, "entity_list": entities}, self.triple_prompt)
                    logger.debug(f"关系抽取: LLM调用成功，响应类型: {type(response)}")
                    print(f"=== LLM响应调试 ===")
                    print(f"响应类型: {type(response)}")
                    print(f"响应内容: {response}")
                    
                    relations = self._parse_relation_response(response)
                    print(f"=== 关系解析调试 ===")
                    print(f"解析后关系数量: {len(relations) if relations else 0}")
                    print(f"解析后关系内容: {relations}")
                    logger.info(f"关系抽取完成: 解析出{len(relations) if relations else 0}个关系")
                except Exception as llm_error:
                    import traceback
                    logger.error(f"关系抽取: LLM调用失败，使用简单关系抽取: {str(llm_error)}")
                    logger.error(f"关系抽取: LLM调用详细异常堆栈: {traceback.format_exc()}")
                    logger.error(f"关系抽取LLM实例详细信息: {type(self.llm)}, 属性: {dir(self.llm)}")
                    logger.error(f"关系抽取提示词实例详细信息: {type(self.triple_prompt)}, 属性: {dir(self.triple_prompt)}")
                    relations = self._simple_relation_extraction(text, entities)
            
            return relations
            
        except Exception as e:
            logger.error(f"隧道工程关系抽取失败: {str(e)}")
            return []
    
    def _parse_standardization_response(self, response: str, entities: List[str]) -> Dict[str, str]:
        """
        解析标准化响应
        
        Args:
            response: LLM响应
            entities: 原始实体列表
            
        Returns:
            标准化映射
        """
        try:
            # 简单的解析逻辑，实际应根据提示格式调整
            standardized = {}
            for entity in entities:
                # 默认映射到自身
                standardized[entity] = entity.strip()
            return standardized
        except Exception as e:
            logger.error(f"解析标准化响应失败: {str(e)}")
            return {entity: entity for entity in entities}
    
    def _parse_relation_response(self, response) -> List[Dict[str, Any]]:
        """
        解析关系抽取响应
        
        Args:
            response: LLM响应（可能是字典或字符串）
            
        Returns:
            关系列表
        """
        try:
            print(f"=== 解析关系响应调试 ===")
            print(f"响应类型: {type(response)}")
            print(f"响应内容: {response}")
            
            relations = []
            
            # 如果响应是字典格式
            if isinstance(response, dict):
                if 'triples' in response:
                    triples = response['triples']
                    print(f"找到triples字段，包含{len(triples)}个三元组")
                    
                    for triple in triples:
                        if isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
                            relation = {
                                'subject': triple['subject'],
                                'predicate': triple['predicate'],
                                'object': triple['object'],
                                'confidence': triple.get('confidence', 0.5)
                            }
                            relations.append(relation)
                            print(f"解析关系: {relation}")
                else:
                    print("响应字典中未找到triples字段")
            
            # 如果响应是字符串格式，尝试解析JSON
            elif isinstance(response, str):
                import json
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and 'triples' in parsed:
                        return self._parse_relation_response(parsed)  # 递归调用
                except json.JSONDecodeError:
                    print("字符串响应不是有效的JSON格式")
            
            print(f"最终解析出{len(relations)}个关系")
            return relations
            
        except Exception as e:
            logger.error(f"解析关系响应失败: {str(e)}")
            import traceback
            logger.error(f"解析关系响应异常堆栈: {traceback.format_exc()}")
            return []
    
    def _simple_relation_extraction(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        简单的关系抽取逻辑（备用方案）
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表
        """
        try:
            relations = []
            
            # 简单的关系抽取逻辑
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    if entity1 in text and entity2 in text:
                        # 检查实体在文本中的位置关系
                        pos1 = text.find(entity1)
                        pos2 = text.find(entity2)
                        
                        if abs(pos1 - pos2) < 100:  # 如果两个实体距离较近
                            relation = {
                                'subject': entity1,
                                'predicate': '相关',
                                'object': entity2,
                                'confidence': 0.8
                            }
                            relations.append(relation)
            
            return relations
        except Exception as e:
            logger.error(f"简单关系抽取失败: {str(e)}")
            return []
    
    def _parse_ner_response(self, response) -> Dict[str, List[str]]:
        """
        解析NER响应
        """
        try:
            logger.info(f"=== _parse_ner_response 调试 ===")
            logger.info(f"响应类型: {type(response)}")
            logger.info(f"响应内容: {response}")
            
            # 如果响应已经是字典类型，直接返回
            if isinstance(response, dict):
                logger.info(f"响应已是字典类型，直接返回: {response}")
                return response
            
            # 如果是字符串，尝试解析JSON
            if isinstance(response, str):
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    logger.info(f"从字符串解析JSON成功: {result}")
                    return result
                else:
                    logger.warning(f"字符串响应不包含JSON格式: {response}")
                    return {}
            
            # 其他类型，尝试转换为字符串再解析
            logger.warning(f"未知响应类型，尝试转换为字符串: {type(response)}")
            response_str = str(response)
            if '{' in response_str and '}' in response_str:
                json_start = response_str.find('{')
                json_end = response_str.rfind('}') + 1
                json_str = response_str[json_start:json_end]
                result = json.loads(json_str)
                logger.info(f"从转换字符串解析JSON成功: {result}")
                return result
            
            logger.warning(f"无法解析响应，返回空字典")
            return {}
            
        except Exception as e:
            logger.warning(f"NER响应解析失败: {str(e)}")
            logger.warning(f"响应类型: {type(response)}, 响应内容: {response}")
            return {}
    
    def _build_subgraph(self, entities: Dict[str, str], relations: List[Dict[str, Any]], chunk: Chunk) -> SubGraph:
        """
        构建知识子图
        
        Args:
            entities: 标准化后的实体映射
            relations: 抽取的关系列表
            chunk: 原始chunk信息，用于设置节点的source属性
            
        Returns:
            知识子图
        """
        logger.info(f"=== 构建子图调试 ===")
        logger.info(f"输入实体映射: {entities}")
        logger.info(f"输入关系列表: {relations}")
        
        nodes = []
        edges = []
        
        # 构建节点
        entity_set = set(entities.values())
        logger.info(f"实体集合: {entity_set}")
        
        for entity in entity_set:
            # 将中文实体名称映射到英文实体类型
            english_entity_type = self._map_chinese_to_english_entity_type(entity)
            
            # 构建节点属性，包含source信息
            node_properties = {
                "type": "entity", 
                "chinese_name": entity,
                "source_id": getattr(chunk, 'id', 'unknown'),
                "source_content": getattr(chunk, 'content', '')[:200] if hasattr(chunk, 'content') else ''  # 截取前200字符作为摘要
            }
            
            node = Node(
                _id=entity,
                name=entity,
                label=english_entity_type,  # 使用英文实体类型作为label
                properties=node_properties
            )
            nodes.append(node)
            logger.debug(f"创建节点: {entity} -> {english_entity_type}, source_id: {node_properties['source_id']}")
        
        # 构建边
        for relation in relations:
            subject = relation.get('subject', '')
            predicate = relation.get('predicate', '')
            obj = relation.get('object', '')
            confidence = relation.get('confidence', 1.0)
            
            if subject in entity_set and obj in entity_set:
                # 找到对应的节点对象
                subject_node = None
                object_node = None
                for node in nodes:
                    if node.name == subject:
                        subject_node = node
                    if node.name == obj:
                        object_node = node
                
                if subject_node and object_node:
                    # 将中文关系名称映射到英文关系名称
                    english_relation = self._map_chinese_to_english_relation(predicate)
                    
                    edge = Edge(
                        _id=f"{subject}_{english_relation}_{obj}",
                        from_node=subject_node,
                        to_node=object_node,
                        label=english_relation,  # 使用英文关系名称作为label
                        properties={"confidence": confidence, "chinese_relation": predicate}
                    )
                    edges.append(edge)
                    logger.debug(f"创建边: {subject} -[{predicate} -> {english_relation}]-> {obj}")
        
        # 创建SubGraph对象
        sub_graph = SubGraph(nodes, edges)
        
        # 添加Chunk节点并建立实体与Chunk的关系（参考schema_free_extractor.py的实现）
        self._assemble_sub_graph_with_chunk(sub_graph, chunk)
        
        logger.info(f"构建完成 - 节点数: {len(sub_graph.nodes)}, 边数: {len(sub_graph.edges)}")
        logger.info(f"=== 构建子图结束 ===")
        return sub_graph
    
    def _assemble_sub_graph_with_chunk(self, sub_graph: SubGraph, chunk: Chunk):
        """
        将Chunk信息添加到子图中，为每个实体节点添加指向Chunk的"source"边
        
        Args:
            sub_graph: 要添加Chunk信息的子图
            chunk: 包含文本和元数据的Chunk对象
        """
        from knext.schema.client import CHUNK_TYPE
        
        # 为每个现有节点添加指向Chunk的"source"边
        for node in sub_graph.nodes:
            sub_graph.add_edge(node.id, node.label, "source", chunk.id, CHUNK_TYPE)
            logger.debug(f"为节点 {node.name} 添加source边指向chunk {chunk.id}")
        
        # 添加Chunk节点到子图中
        chunk_properties = {
            "id": chunk.id,
            "name": getattr(chunk, 'name', chunk.id),
            "content": f"{getattr(chunk, 'name', chunk.id)}\n{getattr(chunk, 'content', '')}",
        }
        
        # 添加chunk的其他属性（如果有的话）
        if hasattr(chunk, 'kwargs') and chunk.kwargs:
            chunk_properties.update(chunk.kwargs)
        
        sub_graph.add_node(
            chunk.id,
            getattr(chunk, 'name', chunk.id),
            CHUNK_TYPE,
            chunk_properties
        )
        
        # 设置子图的ID为chunk的ID
        sub_graph.id = chunk.id
        
        logger.debug(f"添加Chunk节点: {chunk.id}, 类型: {CHUNK_TYPE}")
    
    def _map_chinese_to_english_entity_type(self, chinese_entity: str) -> str:
        """
        将中文实体名称映射到Schema中定义的英文实体类型
        
        Args:
            chinese_entity: 中文实体名称
            
        Returns:
            对应的英文实体类型名称
        """
        try:
            # 如果有Schema信息，尝试从中查找映射
            if hasattr(self, 'schema') and self.schema:
                for english_name, entity_info in self.schema.items():
                    if isinstance(entity_info, dict):
                        # 检查中文名称是否匹配
                        chinese_name = entity_info.get('label', '')
                        if chinese_name == chinese_entity:
                            logger.debug(f"找到精确匹配: {chinese_entity} -> {english_name}")
                            return english_name
                        
                        # 检查是否包含关键词
                        if chinese_entity in chinese_name or chinese_name in chinese_entity:
                            logger.debug(f"找到部分匹配: {chinese_entity} -> {english_name}")
                            return english_name
            
            # 如果没有找到精确匹配，使用基于规则的映射
            entity_mapping = self._get_entity_type_mapping()
            
            for english_type, chinese_keywords in entity_mapping.items():
                for keyword in chinese_keywords:
                    if keyword in chinese_entity or chinese_entity in keyword:
                        logger.debug(f"基于规则映射: {chinese_entity} -> {english_type}")
                        return english_type
            
            # 如果都没有找到，返回通用实体类型
            logger.warning(f"无法映射实体类型: {chinese_entity}，使用默认类型")
            return "Entity"
            
        except Exception as e:
            logger.error(f"实体类型映射失败: {chinese_entity}, 错误: {e}")
            return "Entity"
    
    def _get_entity_type_mapping(self) -> Dict[str, List[str]]:
        """
        从Schema中动态获取实体类型映射规则
        
        Returns:
            英文实体类型到中文关键词的映射
        """
        entity_mapping = {}
        
        # 如果有Schema信息，从中提取实体类型和中文标签
        if hasattr(self, 'schema') and self.schema:
            # 正确的schema使用方式：遍历schema中的所有类型
            try:
                # schema是一个SchemaClient对象，需要通过正确的方式访问
                # 根据schema_free_extractor.py的用法，应该通过schema.get(type_name)获取spg_type
                # 但首先需要获取所有可用的类型名称
                if hasattr(self.schema, '__iter__'):
                    for type_name in self.schema:
                        spg_type = self.schema.get(type_name)
                        if spg_type is not None:
                            # 获取类型的中文标签（如果有的话）
                            chinese_label = getattr(spg_type, 'label', '') or getattr(spg_type, 'name_zh', '') or type_name
                            
                            # 将中文标签分解为关键词
                            keywords = [chinese_label, type_name]  # 包含英文类型名和中文标签
                            
                            # 添加一些常见的同义词或相关词
                            if '隧道' in chinese_label:
                                keywords.extend(['隧道', '隧道段', '段落', '区段'])
                            elif '排水' in chinese_label:
                                keywords.extend(['排水', '排水井', '沉沙井', '检查井'])
                            elif '防水' in chinese_label:
                                keywords.extend(['防水', '防水板', '防水层', '防水材料'])
                            elif '止水' in chinese_label:
                                keywords.extend(['止水', '止水条', '止水带'])
                            elif '处理' in chinese_label:
                                keywords.extend(['处理', '处理方法', '施工方法', '工艺', '技术'])
                            elif '安全' in chinese_label or '奖惩' in chinese_label:
                                keywords.extend(['安全', '制度', '奖惩', '管理制度'])
                            elif '物资' in chinese_label or '材料' in chinese_label:
                                keywords.extend(['物资', '材料', '设备'])
                            elif '缝' in chinese_label:
                                keywords.extend(['缝', '施工缝', '沉降缝', '伸缩缝'])
                            
                            entity_mapping[type_name] = list(set(keywords))
            except Exception as e:
                logger.warning(f"从Schema中提取实体类型映射失败: {e}")
        
        # 如果Schema为空或没有足够信息，使用基础映射作为后备
        if not entity_mapping:
            entity_mapping = {
                "Entity": ["实体", "对象", "物体"],  # 通用实体类型
                "Material": ["材料", "物质", "物资"],
                "Equipment": ["设备", "机械", "工具"],
                "Method": ["方法", "工艺", "技术"],
                "System": ["系统", "制度", "体系"]
            }
        
        return entity_mapping
    
    def _map_chinese_to_english_relation(self, chinese_relation: str) -> str:
        """
        将中文关系名称映射到Schema中定义的英文关系名称
        
        Args:
            chinese_relation: 中文关系名称（可能包含命名空间前缀）
            
        Returns:
            对应的英文关系名称
        """
        try:
            # 处理带命名空间的关系名称，如 'appliedTo_Tunnelknowledge.Hazard'
            base_relation = chinese_relation
            if '_' in chinese_relation:
                # 提取基础关系名称（下划线前的部分）
                base_relation = chinese_relation.split('_')[0]
                logger.debug(f"提取基础关系名称: {chinese_relation} -> {base_relation}")
            
            # 如果有Schema信息，尝试从中查找映射
            if hasattr(self, 'schema') and self.schema:
                try:
                    # 正确的schema使用方式：遍历schema中的所有类型
                    if hasattr(self.schema, '__iter__'):
                        for type_name in self.schema:
                            spg_type = self.schema.get(type_name)
                            if spg_type is not None and hasattr(spg_type, 'properties'):
                                # 遍历该类型的所有属性（包括关系）
                                for prop_name, prop in spg_type.properties.items():
                                    # 检查是否是关系属性（非基础类型）
                                    if hasattr(prop, 'object_type_name_en'):
                                        from knext.schema.client import BASIC_TYPES
                                        if prop.object_type_name_en not in BASIC_TYPES:
                                            # 检查是否有关系名称的映射
                                            if base_relation == prop_name or chinese_relation == prop_name:
                                                return prop_name
                except Exception as e:
                    logger.warning(f"从Schema中查找关系映射失败: {e}")
            
            # 使用基于规则的映射
            relation_mapping = self._get_relation_type_mapping()
            
            # 首先尝试精确匹配基础关系名称
            if base_relation in relation_mapping:
                logger.debug(f"关系映射: {chinese_relation} -> {base_relation}")
                return base_relation
            
            # 然后尝试关键词匹配
            for english_relation, chinese_keywords in relation_mapping.items():
                for keyword in chinese_keywords:
                    if keyword in base_relation or base_relation in keyword:
                        logger.debug(f"关系映射: {chinese_relation} -> {english_relation}")
                        return english_relation
            
            # 如果都没有找到，返回基础关系名称（去掉命名空间）
            logger.warning(f"无法映射关系类型: {chinese_relation}，使用基础名称: {base_relation}")
            return base_relation
            
        except Exception as e:
            logger.error(f"关系类型映射失败: {chinese_relation}, 错误: {e}")
            return chinese_relation
    
    def _get_relation_type_mapping(self) -> Dict[str, List[str]]:
        """
        从Schema中动态获取关系类型映射规则
        
        Returns:
            英文关系类型到中文关键词的映射
        """
        relation_mapping = {}
        
        # 如果有Schema信息，从中提取关系类型和中文标签
        if hasattr(self, 'schema') and self.schema:
            try:
                # 正确的schema使用方式：遍历schema中的所有类型
                if hasattr(self.schema, '__iter__'):
                    for type_name in self.schema:
                        spg_type = self.schema.get(type_name)
                        if spg_type is not None and hasattr(spg_type, 'properties'):
                            # 遍历该类型的所有属性（包括关系）
                            for prop_name, prop in spg_type.properties.items():
                                # 检查是否是关系属性（非基础类型）
                                if hasattr(prop, 'object_type_name_en'):
                                    from knext.schema.client import BASIC_TYPES
                                    if prop.object_type_name_en not in BASIC_TYPES:
                                        # 这是一个关系属性
                                        english_relation = prop_name
                                        
                                        # 基于英文关系名称生成中文关键词
                                        keywords = []
                                        if english_relation == 'contains':
                                            keywords = ["包含", "含有", "包括"]
                                        elif english_relation == 'partOf':
                                            keywords = ["属于", "组成", "部分"]
                                        elif english_relation == 'locatedIn':
                                            keywords = ["位于", "在", "处于"]
                                        elif english_relation == 'installedIn':
                                            keywords = ["安装于", "安装在", "设置在"]
                                        elif english_relation == 'connectedTo':
                                            keywords = ["连接至", "连接到", "连通"]
                                        elif english_relation == 'usedIn':
                                            keywords = ["用于", "使用于", "应用于"]
                                        elif english_relation == 'appliedTo':
                                            keywords = ["应用于", "适用于", "施用于"]
                                        elif english_relation == 'appliesTo':
                                            keywords = ["适用于", "应用于"]
                                        elif english_relation == 'involves':
                                            keywords = ["涉及", "包括", "涉及到"]
                                        elif english_relation == 'protectedBy':
                                            keywords = ["被保护", "保护", "防护"]
                                        elif english_relation == 'equippedWith':
                                            keywords = ["配备", "装备", "设有"]
                                        elif english_relation == 'prohibits':
                                            keywords = ["禁止", "不允许", "严禁"]
                                        elif english_relation == 'requires':
                                            keywords = ["需要", "要求", "必须"]
                                        elif english_relation == 'ensures':
                                            keywords = ["确保", "保证", "保障"]
                                        elif english_relation == 'meets':
                                            keywords = ["满足", "达到", "符合"]
                                        else:
                                            # 对于其他关系，使用通用映射
                                            keywords = ["相关", "关联"]
                                        
                                        if keywords:
                                            relation_mapping[english_relation] = keywords
            except Exception as e:
                logger.warning(f"从Schema中提取关系类型映射失败: {e}")
        
        # 如果Schema为空或没有足够信息，使用基础映射作为后备
        if not relation_mapping:
            relation_mapping = {
                "relatedTo": ["相关", "关联", "有关"],
                "contains": ["包含", "含有", "包括"],
                "partOf": ["属于", "组成", "部分"],
                "usedIn": ["用于", "使用于", "应用于"]
            }
        
        return relation_mapping
    
    def _update_extraction_stats(self, entities: Dict[str, List[str]], relations: List[Dict[str, Any]]):
        """
        更新抽取统计信息
        
        Args:
            entities: 抽取的实体
            relations: 抽取的关系
        """
        try:
            total = self.extraction_stats['total_extractions']
            
            # 计算实体平均数
            entity_count = sum(len(entity_list) for entity_list in entities.values())
            current_avg_entities = self.extraction_stats['avg_entities_per_text']
            self.extraction_stats['avg_entities_per_text'] = (
                (current_avg_entities * total + entity_count) / (total + 1)
            )
            
            # 计算关系平均数
            relation_count = len(relations)
            current_avg_relations = self.extraction_stats['avg_relations_per_text']
            self.extraction_stats['avg_relations_per_text'] = (
                (current_avg_relations * total + relation_count) / (total + 1)
            )
            
            logger.debug(f"统计更新: 实体{entity_count}个, 关系{relation_count}个")
            
        except Exception as e:
            logger.warning(f"统计更新失败: {str(e)}")
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        获取抽取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'extractor_info': {
                'name': 'TunnelEngineeringExtractor',
                'version': '1.0.0',
                'project_id': self.kag_project_config.project_id
            },
            'performance_stats': self.extraction_stats,
            'schema_info': {
                'entity_types': len(self.entity_types),
                'relation_types': len(self.relation_types),
                'total_entities': sum(len(entities) for entities in self.entity_types.values()),
                'total_relations': sum(len(relations) for relations in self.relation_types.values())
            }
        }