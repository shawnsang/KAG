#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema感知的提示模板基类

从OpenSPG Schema中动态获取实体类型和关系定义，
用于NER、实体标准化和三元组抽取等任务。
"""

from typing import Dict, List, Any, Set, Tuple
from kag.common.registry import Registrable
from kag.interface import PromptABC
from knext.schema.client import SchemaClient
import re
import logging
import json

logger = logging.getLogger(__name__)


class SchemaAwarePromptBase(PromptABC):
    """
    Schema感知的提示模板基类
    
    从OpenSPG Schema中动态获取实体类型和关系定义，
    为各种NLP任务提供基础支持。
    """
    
    # 定义模板属性以符合PromptABC规范
    template_zh = {
        "instruction": "请根据以下要求处理文本：",
        "input": "$input",
        "output_format": "请按照指定格式输出结果。"
    }
    
    template_en = {
        "instruction": "Please process the text according to the following requirements:",
        "input": "$input",
        "output_format": "Please output the result in the specified format."
    }
    
    @property
    def template_variables(self) -> List[str]:
        """返回模板变量列表"""
        return ["input"]
    
    def __init__(self, project_id: str = None, host_addr: str = None, schema=None, **kwargs):
        super().__init__(**kwargs)
        # project_id is read-only property from PromptABC, don't set it directly
        self._project_id_param = project_id or "Tunnelknowledge"
        self.host_addr = host_addr or "http://127.0.0.1:8887"
        self.schema_client = None
        self.schema_entities = {}
        self.schema_relations = {}
        self.entity_properties = {}
        
        # 如果直接传入了schema对象，使用它；否则从服务器加载
        if schema is not None:
            self._parse_schema_object(schema)
        else:
            self._load_schema()
    
    def _parse_schema_object(self, schema):
        """
        解析已加载的Schema对象（字典格式）
        """
        try:
            # 直接使用传入的schema字典
            self.schema_client = schema
            
            # 解析实体类型（schema是字典，键为类型名，值为BaseSpgType对象）
            for type_name, spg_type in schema.items():
                if hasattr(spg_type, 'spg_type_enum'):
                    # 获取实体的中文名称
                    chinese_name = getattr(spg_type, 'name_zh', type_name)
                    
                    # 存储实体信息
                    self.schema_entities[type_name] = {
                        'chinese_name': chinese_name,
                        'properties': getattr(spg_type, 'properties', {}),
                        'relations': getattr(spg_type, 'relations', {})
                    }
                    
                    # 存储属性信息
                    if hasattr(spg_type, 'properties'):
                        self.entity_properties[type_name] = spg_type.properties
                    
                    # 解析关系类型
                    if hasattr(spg_type, 'relations'):
                        for relation_name, relation_obj in spg_type.relations.items():
                            relation_category = self._categorize_relation(relation_name)
                            if relation_category not in self.schema_relations:
                                self.schema_relations[relation_category] = set()
                            self.schema_relations[relation_category].add(relation_name)
            
            # 转换为列表格式
            for category in self.schema_relations:
                self.schema_relations[category] = list(self.schema_relations[category])
            
            logger.info(f"成功解析Schema对象: {len(self.schema_entities)}个实体类型, {len(self.schema_relations)}个关系类型")
            
        except Exception as e:
            logger.warning(f"无法解析Schema对象: {e}, 使用默认实体定义")
            self._load_default_entities()
    
    def _load_schema(self):
        """
        从OpenSPG服务器加载Schema定义
        """
        try:
            self.schema_client = SchemaClient(
                host_addr=self.host_addr,
                project_id=self._project_id_param
            )
            schema_data = self.schema_client.load()
            
            # 解析实体类型
            self._parse_entities(schema_data)
            
            # 解析关系类型
            self._parse_relations(schema_data)
            
            logger.info(f"成功加载Schema: {len(self.schema_entities)}个实体类型, {len(self.schema_relations)}个关系类型")
            
        except Exception as e:
            logger.warning(f"无法加载Schema: {e}, 使用默认实体定义")
            self._load_default_entities()
    
    def _parse_entities(self, schema_data: Dict):
        """
        解析Schema中的实体类型定义
        """
        if not schema_data:
            return
            
        for entity_name, entity_info in schema_data.items():
            if isinstance(entity_info, dict) and entity_info.get('type') == 'EntityType':
                # 获取实体的中文名称
                chinese_name = entity_info.get('label', entity_name)
                
                # 存储实体信息
                self.schema_entities[entity_name] = {
                    'chinese_name': chinese_name,
                    'properties': entity_info.get('properties', {}),
                    'relations': entity_info.get('relations', {})
                }
                
                # 存储属性信息
                if 'properties' in entity_info:
                    self.entity_properties[entity_name] = entity_info['properties']
    
    def _parse_relations(self, schema_data: Dict):
        """
        解析Schema中的关系类型定义
        """
        relation_set = set()
        
        for entity_name, entity_info in schema_data.items():
            if isinstance(entity_info, dict) and 'relations' in entity_info:
                for relation_name, target_entities in entity_info['relations'].items():
                    relation_set.add(relation_name)
                    
                    # 按关系类型分组
                    relation_category = self._categorize_relation(relation_name)
                    if relation_category not in self.schema_relations:
                        self.schema_relations[relation_category] = set()
                    self.schema_relations[relation_category].add(relation_name)
        
        # 转换为列表格式
        for category in self.schema_relations:
            self.schema_relations[category] = list(self.schema_relations[category])
    
    def _categorize_relation(self, relation_name: str) -> str:
        """
        根据关系名称推断关系类别
        """
        # 结构关系
        if any(keyword in relation_name for keyword in ['contains', 'partOf', 'locatedIn', 'installedIn', 'connectedTo']):
            return '结构关系'
        
        # 属性关系
        elif any(keyword in relation_name for keyword in ['has', '具有', 'equals', 'measures']):
            return '属性关系'
        
        # 工艺关系
        elif any(keyword in relation_name for keyword in ['usedIn', 'appliedTo', 'performedOn', 'implements']):
            return '工艺关系'
        
        # 质量关系
        elif any(keyword in relation_name for keyword in ['compliesWith', 'meets', 'requires', 'ensures']):
            return '质量关系'
        
        # 时间关系
        elif any(keyword in relation_name for keyword in ['before', 'after', 'during', 'performedAfter']):
            return '时间关系'
        
        # 默认为其他关系
        else:
            return '其他关系'
    
    def _load_default_entities(self):
        """
        加载默认实体定义（当Schema不可用时）
        """
        self.schema_entities = {
            # 工程结构
            "TunnelSegment": {"chinese_name": "隧道段"},
            "TunnelLining": {"chinese_name": "隧道衬砌"},
            "WaterproofLayer": {"chinese_name": "防水层"},
            "DrainageSystem": {"chinese_name": "排水系统"},
            "SupportStructure": {"chinese_name": "支护结构"},
            "TunnelArch": {"chinese_name": "拱顶"},
            "SideWall": {"chinese_name": "边墙"},
            "Invert": {"chinese_name": "仰拱"},
            
            # 材料设备
            "WaterproofBoard": {"chinese_name": "防水板"},
            "WaterproofCoating": {"chinese_name": "防水涂料"},
            "WaterStop": {"chinese_name": "止水带"},
            "WaterStopStrip": {"chinese_name": "止水条"},
            "GroutingMaterial": {"chinese_name": "注浆材料"},
            "Sealant": {"chinese_name": "密封胶"},
            "DrainagePipe": {"chinese_name": "排水管"},
            "CollectionWell": {"chinese_name": "集水井"},
            "DrainagePump": {"chinese_name": "排水泵"},
            "DrainageDitch": {"chinese_name": "排水沟"},
            "BlindDitch": {"chinese_name": "盲沟"},
            "PermeablePipe": {"chinese_name": "渗排水管"},
            "Concrete": {"chinese_name": "混凝土"},
            "SteelFrame": {"chinese_name": "钢架"},
            
            # 施工工艺
            "GroutingProcess": {"chinese_name": "注浆工艺"},
            "SprayingProcess": {"chinese_name": "喷射工艺"},
            "LayingProcess": {"chinese_name": "铺设工艺"},
            "WeldingProcess": {"chinese_name": "焊接工艺"},
            "BondingProcess": {"chinese_name": "粘贴工艺"},
            "SealingProcess": {"chinese_name": "嵌缝工艺"},
            "CuringProcess": {"chinese_name": "养护工艺"},
            
            # 质量标准
            "QualityInspection": {"chinese_name": "质量检测"},
            "Acceptance": {"chinese_name": "验收标准"},
            "Testing": {"chinese_name": "试验标准"},
            "Monitoring": {"chinese_name": "监测标准"},
            "Assessment": {"chinese_name": "评估标准"},
            
            # 技术参数
            "Thickness": {"chinese_name": "厚度"},
            "Strength": {"chinese_name": "强度"},
            "Pressure": {"chinese_name": "压力"},
            "FlowRate": {"chinese_name": "流量"},
            "PermeabilityCoeff": {"chinese_name": "渗透系数"},
            "BondStrength": {"chinese_name": "粘结强度"},
            "TensileStrength": {"chinese_name": "抗拉强度"}
        }
        
        self.schema_relations = {
            "结构关系": ["contains", "partOf", "locatedIn", "installedIn", "connectedTo"],
            "属性关系": ["hasThickness", "hasStrength", "hasPressure", "hasFlowRate"],
            "工艺关系": ["usedIn", "appliedTo", "performedOn", "implements"],
            "质量关系": ["compliesWith", "meets", "requires", "ensures"]
        }
    
    def get_entity_types(self) -> Dict[str, List[str]]:
        """
        获取所有实体类型（不进行分组）
        """
        logger.info(f"开始获取实体类型，当前schema_entities数量: {len(self.schema_entities)}")
        
        if not self.schema_entities:
            logger.warning("schema_entities为空，可能Schema加载失败，使用默认实体")
            self._load_default_entities()
        
        # 直接返回所有实体，不进行分组
        all_entities = []
        for entity_name, entity_info in self.schema_entities.items():
            chinese_name = entity_info.get('chinese_name', entity_name)
            all_entities.append(chinese_name)
            logger.debug(f"实体: {entity_name}({chinese_name})")
        
        logger.info(f"获取到实体类型总计: {len(all_entities)}个")
        
        # 返回单一类别包含所有实体
        return {"所有实体": all_entities}
    
    def _categorize_entity(self, entity_name: str, chinese_name: str) -> str:
        """
        根据实体名称推断实体类别
        """
        # 工程结构
        if any(keyword in entity_name.lower() for keyword in ['tunnel', 'segment', 'joint', 'chamber', 'section', 'lining', 'arch', 'wall', 'invert', 'support']):
            return "工程结构"
        elif any(keyword in chinese_name for keyword in ['隧道', '段', '接缝', '洞室', '断面', '衬砌', '拱顶', '边墙', '仰拱', '支护', '防水层', '排水系统']):
            return "工程结构"
        
        # 材料设备
        elif any(keyword in entity_name.lower() for keyword in ['concrete', 'steel', 'material', 'pipe', 'board', 'waterproof', 'drainage', 'pump', 'well', 'ditch', 'stop', 'coating', 'grouting', 'sealant']):
            return "材料设备"
        elif any(keyword in chinese_name for keyword in ['混凝土', '钢', '材料', '管', '板', '设备', '防水', '排水', '泵', '井', '沟', '止水', '涂料', '注浆', '密封胶', '盲沟']):
            return "材料设备"
        
        # 施工工艺
        elif any(keyword in entity_name.lower() for keyword in ['method', 'operation', 'construction', 'welding', 'grouting', 'process', 'spraying', 'laying', 'bonding', 'sealing', 'curing']):
            return "施工工艺"
        elif any(keyword in chinese_name for keyword in ['方法', '作业', '施工', '焊接', '注浆', '工艺', '喷射', '铺设', '粘贴', '嵌缝', '养护']):
            return "施工工艺"
        
        # 质量标准
        elif any(keyword in entity_name.lower() for keyword in ['standard', 'quality', 'control', 'test', 'inspection', 'acceptance', 'monitoring', 'assessment']):
            return "质量标准"
        elif any(keyword in chinese_name for keyword in ['标准', '质量', '控制', '检测', '规范', '验收', '监测', '评估']):
            return "质量标准"
        
        # 安全措施
        elif any(keyword in entity_name.lower() for keyword in ['safety', 'protection', 'emergency', 'fire']):
            return "安全措施"
        elif any(keyword in chinese_name for keyword in ['安全', '防护', '应急', '消防', '保护']):
            return "安全措施"
        
        # 技术参数
        elif any(keyword in entity_name.lower() for keyword in ['thickness', 'strength', 'pressure', 'temperature', 'flow', 'permeability', 'bond', 'tensile']):
            return "技术参数"
        elif any(keyword in chinese_name for keyword in ['厚度', '强度', '压力', '温度', '参数', '流量', '渗透', '粘结', '抗拉']):
            return "技术参数"
        
        else:
            return "其他实体"
    
    def get_relation_types(self) -> Dict[str, List[str]]:
        """
        获取关系类型定义
        """
        return self.schema_relations
    
    def get_entity_properties(self, entity_name: str) -> Dict[str, str]:
        """
        获取指定实体的属性定义
        """
        return self.entity_properties.get(entity_name, {})
    
    def get_all_entities(self) -> List[str]:
        """
        获取所有实体的中文名称列表
        """
        return [info.get('chinese_name', name) for name, info in self.schema_entities.items()]
    
    def get_all_relations(self) -> List[str]:
        """
        获取所有关系名称列表
        """
        all_relations = []
        for relations in self.schema_relations.values():
            all_relations.extend(relations)
        return all_relations
    
    def build_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        构建提示（基类默认实现）
        
        Args:
            inputs: 输入参数字典
            
        Returns:
            格式化的提示字符串
        """
        # 基类提供默认实现，子类应该重写此方法
        text = inputs.get('input', '')
        return f"请处理以下文本：\n{text}"
    
    def refresh_schema(self):
        """
        重新加载Schema定义
        """
        self.schema_entities.clear()
        self.schema_relations.clear()
        self.entity_properties.clear()
        self._load_schema()
    
    def parse_response(self, response: str, **kwargs) -> Any:
        """
        解析LLM响应（基类默认实现）
        
        Args:
            response: LLM响应字符串
            **kwargs: 其他参数
            
        Returns:
            解析后的结果
        """
        try:
            # 尝试解析JSON格式的响应
            if isinstance(response, str):
                # 查找JSON部分
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # 如果没有JSON格式，返回原始字符串
                    return response.strip()
            else:
                return response
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 返回原始响应")
            return response.strip() if isinstance(response, str) else response
        except Exception as e:
            logger.error(f"响应解析异常: {e}")
            return response