#!/usr/bin/env python3
"""
模型API接口 - 为后端模型预留的接口
当后端模型准备好后，替换这里的实现即可
"""

import json
import time
import random
from typing import Dict, List, Any, Optional

class ModelAPI:
    """模型API接口类 - 为后端团队预留"""
    
    def __init__(self, model_endpoint: str = None, api_key: str = None):
        """
        初始化模型API
        
        Args:
            model_endpoint: 模型服务端点
            api_key: API密钥
        """
        self.model_endpoint = model_endpoint or "http://localhost:8000"
        self.api_key = api_key
        self.threat_types = [
            "Benign", "DoS", "DDoS", "Botnet", "Bruteforce", 
            "Infiltration", "Portscan", "WebAttacks"
        ]
    
    def predict_threat(self, input_text: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        威胁预测接口 - 待后端团队实现
        
        Args:
            input_text: 输入文本
            use_rag: 是否使用RAG增强
            
        Returns:
            预测结果字典
        """
        # TODO: 替换为真实的模型API调用
        # 示例调用方式:
        # response = requests.post(
        #     f"{self.model_endpoint}/predict",
        #     json={"text": input_text, "use_rag": use_rag},
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        # return response.json()
        
        # 现在返回模拟结果
        return self._mock_prediction(input_text, use_rag)
    
    def batch_predict(self, texts: List[str], use_rag: bool = True) -> List[Dict[str, Any]]:
        """
        批量预测接口 - 待后端团队实现
        
        Args:
            texts: 文本列表
            use_rag: 是否使用RAG
            
        Returns:
            预测结果列表
        """
        # TODO: 替换为真实的批量预测API
        # response = requests.post(
        #     f"{self.model_endpoint}/batch_predict",
        #     json={"texts": texts, "use_rag": use_rag},
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        # return response.json()
        
        # 现在返回模拟结果
        return [self._mock_prediction(text, use_rag) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息 - 待后端团队实现
        
        Returns:
            模型信息字典
        """
        # TODO: 替换为真实的模型信息获取
        return {
            "model_name": "DeepSeek-v2-lite",
            "embedding_model": "minicpm-2b",
            "framework": "UltraRAG",
            "version": "1.0.0",
            "status": "ready",
            "supported_threats": self.threat_types
        }
    
    def _mock_prediction(self, input_text: str, use_rag: bool) -> Dict[str, Any]:
        """模拟预测 - 临时实现"""
        
        # 模拟处理时间
        time.sleep(0.5)
        
        text_lower = input_text.lower()
        
        # 简单的关键词匹配逻辑
        if any(word in text_lower for word in ["normal", "regular", "legitimate"]):
            prediction = "Benign"
            confidence = 0.85
        elif any(word in text_lower for word in ["ddos", "distributed", "coordinated"]):
            prediction = "DDoS" 
            confidence = 0.82
        elif any(word in text_lower for word in ["scan", "probe", "nmap", "port"]):
            prediction = "Portscan"
            confidence = 0.78
        elif any(word in text_lower for word in ["dos", "flood", "overwhelm"]):
            prediction = "DoS"
            confidence = 0.75
        elif any(word in text_lower for word in ["sql", "injection", "xss", "web"]):
            prediction = "WebAttacks"
            confidence = 0.73
        elif any(word in text_lower for word in ["brute", "password", "login"]):
            prediction = "Bruteforce"
            confidence = 0.70
        elif any(word in text_lower for word in ["botnet", "zombie", "malware"]):
            prediction = "Botnet"
            confidence = 0.72
        elif any(word in text_lower for word in ["infiltration", "penetration", "unauthorized"]):
            prediction = "Infiltration"
            confidence = 0.76
        else:
            # 随机选择
            prediction = random.choice(self.threat_types)
            confidence = random.uniform(0.6, 0.9)
        
        # 生成置信度分布
        confidence_scores = {}
        for threat in self.threat_types:
            if threat == prediction:
                confidence_scores[threat] = confidence
            else:
                confidence_scores[threat] = random.uniform(0.01, confidence - 0.1)
        
        # 归一化
        total = sum(confidence_scores.values())
        confidence_scores = {k: v/total for k, v in confidence_scores.items()}
        
        return {
            "prediction": prediction,
            "confidence": confidence_scores[prediction],
            "confidence_scores": confidence_scores,
            "input_text": input_text,
            "use_rag": use_rag,
            "processing_time": 0.5,
            "model_version": "mock-1.0.0"
        }

# 全局API实例
model_api = ModelAPI()

def get_model_api() -> ModelAPI:
    """获取模型API实例"""
    return model_api

# 便捷函数
def predict_threat(text: str, use_rag: bool = True) -> Dict[str, Any]:
    """便捷的威胁预测函数"""
    return model_api.predict_threat(text, use_rag)

def batch_predict_threats(texts: List[str], use_rag: bool = True) -> List[Dict[str, Any]]:
    """便捷的批量预测函数"""
    return model_api.batch_predict(texts, use_rag)
