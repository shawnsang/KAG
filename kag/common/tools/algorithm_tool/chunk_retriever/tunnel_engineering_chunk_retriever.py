# -*- coding: utf-8 -*-
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

import logging
from typing import List, Dict, Any, Optional

from kag.interface import (
    RetrieverABC,
    VectorizeModelABC,
    ChunkData,
    RetrieverOutput,
    EntityData,
    Task,
)
from kag.interface.solver.model.schema_utils import SchemaUtils
from kag.common.config import LogicFormConfiguration
from kag.common.tools.search_api.search_api_abc import SearchApiABC
from kag.common.tools.graph_api.graph_api_abc import GraphApiABC
from knext.schema.client import CHUNK_TYPE

logger = logging.getLogger(__name__)


@RetrieverABC.register("tunnel_engineering_chunk_retriever")
class TunnelEngineeringChunkRetriever(RetrieverABC):
    """
    隧道工程专业知识检索器
    
    特点：
    1. 智能Token管理，优化检索效率
    2. 基于隧道工程领域的专业检索
    3. 支持多模态检索策略
    4. 动态调整检索参数
    5. 结果相关性优化
    """
    
    def __init__(
        self,
        vectorize_model: VectorizeModelABC = None,
        search_api: SearchApiABC = None,
        graph_api: GraphApiABC = None,
        top_k: int = 10,
        score_threshold: float = 0.75,
        max_tokens_per_chunk: int = 512,
        enable_semantic_search: bool = True,
        enable_keyword_search: bool = True,
        **kwargs,
    ):
        """
        初始化Token感知检索器
        
        Args:
            vectorize_model: 向量化模型
            search_api: 搜索API
            graph_api: 图API
            top_k: 返回结果数量
            score_threshold: 分数阈值
            max_tokens_per_chunk: 每个chunk的最大token数
            enable_semantic_search: 是否启用语义搜索
            enable_keyword_search: 是否启用关键词搜索
        """
        super().__init__(top_k, **kwargs)
        
        self.vectorize_model = vectorize_model or VectorizeModelABC.from_config(
            self.kag_config.all_config["vectorize_model"]
        )
        self.search_api = search_api or SearchApiABC.from_config(
            {"type": "openspg_search_api"}
        )
        self.graph_api = graph_api or GraphApiABC.from_config(
            {"type": "openspg_graph_api"}
        )
        
        self.score_threshold = score_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.enable_semantic_search = enable_semantic_search
        self.enable_keyword_search = enable_keyword_search
        
        # 初始化Schema工具
        self.schema_helper: SchemaUtils = SchemaUtils(
            LogicFormConfiguration(
                {
                    "KAG_PROJECT_ID": self.kag_project_config.project_id,
                    "KAG_PROJECT_HOST_ADDR": self.kag_project_config.host_addr,
                }
            )
        )
        
        # 隧道工程专业关键词
        self.tunnel_keywords = {
            "结构": ["隧道", "衬砌", "拱顶", "边墙", "仰拱", "洞门"],
            "材料": ["混凝土", "钢筋", "防水板", "土工布", "排水管"],
            "工艺": ["开挖", "支护", "衬砌", "防水", "排水", "注浆"],
            "质量": ["强度", "厚度", "密实度", "渗透系数"],
            "安全": ["监测", "预警", "应急", "防护"]
        }
        
        # 检索统计
        self.retrieval_stats = {
            "total_queries": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "avg_results_per_query": 0,
            "avg_score": 0.0
        }
        
        logger.info(f"隧道工程检索器初始化完成: top_k={top_k}, threshold={score_threshold}")
    
    def invoke(self, task: Task, **kwargs) -> RetrieverOutput:
        """
        执行Token感知检索
        
        Args:
            task: 检索任务
            **kwargs: 其他参数
            
        Returns:
            检索结果
        """
        try:
            self.retrieval_stats["total_queries"] += 1
            
            query = task.query if hasattr(task, 'query') else str(task)
            logger.info(f"开始Token感知检索: {query[:100]}...")
            
            # 1. 语义检索
            semantic_results = []
            if self.enable_semantic_search:
                semantic_results = self._semantic_search(query)
                self.retrieval_stats["semantic_searches"] += 1
            
            # 2. 关键词检索
            keyword_results = []
            if self.enable_keyword_search:
                keyword_results = self._keyword_search(query)
                self.retrieval_stats["keyword_searches"] += 1
            
            # 3. 结果融合和排序
            merged_results = self._merge_and_rank_results(
                semantic_results, keyword_results, query
            )
            
            # 4. Token感知过滤
            filtered_results = self._token_aware_filter(merged_results)
            
            # 5. 构建返回结果
            chunk_data_list = self._build_chunk_data_list(filtered_results)
            
            # 6. 更新统计信息
            self._update_retrieval_stats(chunk_data_list)
            
            logger.info(f"Token感知检索完成: 返回{len(chunk_data_list)}个结果")
            
            return RetrieverOutput(
                chunk_data_list=chunk_data_list,
                entity_data_list=[],
                graph_data_list=[]
            )
            
        except Exception as e:
            logger.error(f"Token感知检索失败: {str(e)}")
            return RetrieverOutput(
                chunk_data_list=[],
                entity_data_list=[],
                graph_data_list=[]
            )
    
    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """
        语义检索
        
        Args:
            query: 查询文本
            
        Returns:
            语义检索结果
        """
        try:
            # 向量化查询
            query_vector = self.vectorize_model.vectorize(query)
            
            # 执行向量搜索
            results = self.search_api.search_vector(
                label=self.schema_helper.get_label_within_prefix("TokenAware"),
                property_key="content",
                query_vector=query_vector,
                topk=self.top_k * 2,  # 获取更多候选结果
                ef_search=self.top_k * 3,
            )
            
            # 过滤低分结果
            filtered_results = [
                result for result in results
                if result.get("score", 0) >= self.score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"语义检索失败: {str(e)}")
            return []
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            
        Returns:
            关键词检索结果
        """
        try:
            # 提取查询中的专业关键词
            query_keywords = self._extract_tunnel_keywords(query)
            
            if not query_keywords:
                return []
            
            # 构建关键词查询
            keyword_query = " OR ".join(query_keywords)
            
            # 执行关键词搜索
            results = self.search_api.search_text(
                label=self.schema_helper.get_label_within_prefix("TokenAware"),
                property_key="content",
                query=keyword_query,
                topk=self.top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"关键词检索失败: {str(e)}")
            return []
    
    def _extract_tunnel_keywords(self, query: str) -> List[str]:
        """
        从查询中提取隧道工程关键词
        
        Args:
            query: 查询文本
            
        Returns:
            提取的关键词列表
        """
        keywords = []
        
        for category, keyword_list in self.tunnel_keywords.items():
            for keyword in keyword_list:
                if keyword in query:
                    keywords.append(keyword)
        
        return list(set(keywords))  # 去重
    
    def _merge_and_rank_results(
        self, 
        semantic_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        合并和排序检索结果
        
        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            query: 原始查询
            
        Returns:
            合并排序后的结果
        """
        # 使用字典去重，以node id为key
        merged_dict = {}
        
        # 添加语义检索结果
        for result in semantic_results:
            node_id = result.get("node", {}).get("id")
            if node_id:
                result["search_type"] = "semantic"
                result["combined_score"] = result.get("score", 0) * 0.7  # 语义权重0.7
                merged_dict[node_id] = result
        
        # 添加关键词检索结果
        for result in keyword_results:
            node_id = result.get("node", {}).get("id")
            if node_id:
                if node_id in merged_dict:
                    # 如果已存在，增加关键词分数
                    merged_dict[node_id]["combined_score"] += result.get("score", 0) * 0.3
                    merged_dict[node_id]["search_type"] = "hybrid"
                else:
                    result["search_type"] = "keyword"
                    result["combined_score"] = result.get("score", 0) * 0.3  # 关键词权重0.3
                    merged_dict[node_id] = result
        
        # 按综合分数排序
        merged_results = list(merged_dict.values())
        merged_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return merged_results[:self.top_k]
    
    def _token_aware_filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Token感知过滤
        
        Args:
            results: 检索结果
            
        Returns:
            过滤后的结果
        """
        filtered_results = []
        total_tokens = 0
        
        for result in results:
            content = result.get("node", {}).get("properties", {}).get("content", "")
            
            # 估算token数量（简单估算：字符数/4）
            estimated_tokens = len(content) // 4
            
            if total_tokens + estimated_tokens <= self.max_tokens_per_chunk * self.top_k:
                filtered_results.append(result)
                total_tokens += estimated_tokens
            else:
                # 如果超出token限制，截断内容
                remaining_tokens = self.max_tokens_per_chunk * self.top_k - total_tokens
                if remaining_tokens > 100:  # 至少保留100个token
                    truncated_content = content[:remaining_tokens * 4]
                    result["node"]["properties"]["content"] = truncated_content
                    filtered_results.append(result)
                break
        
        return filtered_results
    
    def _build_chunk_data_list(self, results: List[Dict[str, Any]]) -> List[ChunkData]:
        """
        构建ChunkData列表
        
        Args:
            results: 检索结果
            
        Returns:
            ChunkData列表
        """
        chunk_data_list = []
        
        for result in results:
            node = result.get("node", {})
            properties = node.get("properties", {})
            
            chunk_data = ChunkData(
                chunk_id=node.get("id", ""),
                content=properties.get("content", ""),
                score=result.get("combined_score", 0),
                chunk_type=CHUNK_TYPE,
                metadata={
                    "search_type": result.get("search_type", "unknown"),
                    "original_score": result.get("score", 0),
                    "source": properties.get("source", ""),
                    "timestamp": properties.get("timestamp", "")
                }
            )
            
            chunk_data_list.append(chunk_data)
        
        return chunk_data_list
    
    def _update_retrieval_stats(self, chunk_data_list: List[ChunkData]):
        """
        更新检索统计信息
        
        Args:
            chunk_data_list: 检索结果列表
        """
        total_queries = self.retrieval_stats["total_queries"]
        result_count = len(chunk_data_list)
        
        # 更新平均结果数
        self.retrieval_stats["avg_results_per_query"] = (
            (self.retrieval_stats["avg_results_per_query"] * (total_queries - 1) + result_count) / total_queries
        )
        
        # 更新平均分数
        if chunk_data_list:
            avg_score = sum(chunk.score for chunk in chunk_data_list) / len(chunk_data_list)
            self.retrieval_stats["avg_score"] = (
                (self.retrieval_stats["avg_score"] * (total_queries - 1) + avg_score) / total_queries
            )
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "retriever_info": {
                "name": "TunnelEngineeringChunkRetriever",
                "version": "1.0.0",
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "max_tokens_per_chunk": self.max_tokens_per_chunk,
                "semantic_search_enabled": self.enable_semantic_search,
                "keyword_search_enabled": self.enable_keyword_search
            },
            "performance_stats": self.retrieval_stats,
            "keyword_categories": len(self.tunnel_keywords),
            "total_keywords": sum(len(keywords) for keywords in self.tunnel_keywords.values())
        }