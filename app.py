#!/usr/bin/env python3
"""
网络威胁检测 Agent - Gradio Web界面
基于UltraRAG的网络威胁检测agent，集成DeepSeek-v2-lite和minicpm-2b embedding模型
"""

import gradio as gr
import pandas as pd
import numpy as np
import json
import yaml
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logger
from threat_detector import ThreatDetector

# 设置日志
logger = setup_logger(__name__)

class ThreatDetectionApp:
    """网络威胁检测应用主类"""
    
    def __init__(self):
        self.detector = ThreatDetector()
        self.threat_types = [
            "Benign", "DoS", "DDoS", "Botnet", "Bruteforce", 
            "Infiltration", "Portscan", "WebAttacks"
        ]
        self.colors = {
            "Benign": "#2E8B57",
            "DoS": "#FF6B6B", 
            "DDoS": "#FF4757",
            "Botnet": "#5F27CD",
            "Bruteforce": "#FF9F43",
            "Infiltration": "#FF3838",
            "Portscan": "#FFA502",
            "WebAttacks": "#FF6348"
        }
        
    def predict_threat(self, input_text: str, use_rag: bool = True) -> Tuple[str, Dict, str]:
        """
        预测网络威胁类型
        
        Args:
            input_text: 输入的网络流量特征或描述
            use_rag: 是否使用RAG检索增强
            
        Returns:
            prediction: 预测结果
            confidence_scores: 置信度分数
            explanation: 解释文本
        """
        try:
            if not input_text.strip():
                return "请输入有效的网络流量特征", {}, ""
            
            # 调用威胁检测器
            result = self.detector.detect(input_text, use_rag=use_rag)
            
            prediction = result.get("prediction", "未知")
            confidence_scores = result.get("confidence_scores", {})
            explanation = result.get("explanation", "无法生成解释")
            
            return prediction, confidence_scores, explanation
            
        except Exception as e:
            logger.error(f"预测错误: {str(e)}")
            return f"预测失败: {str(e)}", {}, ""
    
    def create_confidence_chart(self, confidence_scores: Dict) -> go.Figure:
        """创建置信度图表"""
        if not confidence_scores:
            fig = go.Figure()
            fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
            return fig
            
        threats = list(confidence_scores.keys())
        scores = list(confidence_scores.values())
        colors = [self.colors.get(threat, "#95A5A6") for threat in threats]
        
        fig = go.Figure(data=[
            go.Bar(
                x=threats,
                y=scores,
                marker_color=colors,
                text=[f"{score:.2%}" for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="威胁类型置信度分布",
            xaxis_title="威胁类型",
            yaxis_title="置信度",
            yaxis=dict(tickformat=".0%"),
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_threat_distribution_chart(self) -> go.Figure:
        """创建威胁类型分布图"""
        # 示例数据，实际应该从数据库或日志中获取
        sample_data = {
            "DoS": 584991,
            "Benign": 458831, 
            "Bruteforce": 389714,
            "DDoS": 221264,
            "Infiltration": 207630,
            "Botnet": 176038,
            "WebAttacks": 155820,
            "Portscan": 119522
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(sample_data.keys()),
                values=list(sample_data.values()),
                marker_colors=[self.colors.get(threat, "#95A5A6") 
                              for threat in sample_data.keys()],
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title="数据集威胁类型分布 (CIC-IDS2017)",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def batch_detect(self, file) -> Tuple[str, str]:
        """批量检测"""
        try:
            if file is None:
                return "请上传文件", ""
            
            # 读取文件
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.jsonl'):
                df = pd.read_json(file.name, lines=True)
            else:
                return "不支持的文件格式，请上传CSV或JSONL文件", ""
            
            if df.empty:
                return "文件为空", ""
            
            results = []
            
            # 假设文本列名为 'text' 或第一列
            text_column = 'text' if 'text' in df.columns else df.columns[0]
            
            for idx, row in df.head(10).iterrows():  # 限制处理前10行
                text = str(row[text_column])
                prediction, confidence_scores, explanation = self.predict_threat(text)
                
                results.append({
                    'index': idx,
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'prediction': prediction,
                    'confidence': max(confidence_scores.values()) if confidence_scores else 0
                })
            
            # 创建结果DataFrame
            result_df = pd.DataFrame(results)
            
            # 生成统计信息
            stats = {
                '总处理数量': len(results),
                '威胁预测': len([r for r in results if r['prediction'] != 'Benign']),
                '正常流量': len([r for r in results if r['prediction'] == 'Benign']),
                '平均置信度': f"{np.mean([r['confidence'] for r in results]):.2%}"
            }
            
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            
            return result_df.to_string(index=False), stats_text
            
        except Exception as e:
            logger.error(f"批量检测错误: {str(e)}")
            return f"处理失败: {str(e)}", ""

def create_interface():
    """创建Gradio界面"""
    
    app = ThreatDetectionApp()
    
    # 自定义CSS
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .threat-high { background-color: #FFE6E6 !important; }
    .threat-medium { background-color: #FFF4E6 !important; }
    .threat-low { background-color: #E6F7E6 !important; }
    """
    
    with gr.Blocks(css=css, title="网络威胁检测系统", theme=gr.themes.Soft()) as demo:
        
        # 标题和描述
        gr.Markdown("""
        # 🛡️ 网络威胁检测 Agent
        
        基于 **UltraRAG** 的智能网络威胁检测系统，集成 **DeepSeek-v2-lite** 和 **minicpm-2b** embedding 模型
        
        ---
        """)
        
        with gr.Tab("🔍 单次检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="网络流量特征描述",
                        placeholder="请输入网络流量特征或异常行为描述...\n例如：发现大量TCP连接尝试，源IP重复，目标端口集中在22和23",
                        lines=5,
                        max_lines=10
                    )
                    
                    use_rag = gr.Checkbox(
                        label="启用RAG检索增强",
                        value=True,
                        info="使用知识库增强检测准确性"
                    )
                    
                    detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    prediction_output = gr.Textbox(
                        label="检测结果",
                        interactive=False
                    )
                    
                    confidence_plot = gr.Plot(
                        label="置信度分析"
                    )
            
            explanation_output = gr.Textbox(
                label="🤖 AI 分析解释",
                lines=4,
                interactive=False
            )
            
            # 检测按钮事件
            detect_btn.click(
                fn=lambda text, rag: (
                    app.predict_threat(text, rag)[0],  # prediction
                    app.create_confidence_chart(app.predict_threat(text, rag)[1]),  # chart
                    app.predict_threat(text, rag)[2]   # explanation
                ),
                inputs=[input_text, use_rag],
                outputs=[prediction_output, confidence_plot, explanation_output]
            )
        
        with gr.Tab("📊 批量检测"):
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="上传数据文件",
                        file_types=[".csv", ".jsonl"],
                        info="支持CSV和JSONL格式，建议小于1000行"
                    )
                    
                    batch_btn = gr.Button("🚀 批量检测", variant="primary")
                
                with gr.Column():
                    batch_stats = gr.Textbox(
                        label="检测统计",
                        lines=5,
                        interactive=False
                    )
            
            batch_results = gr.Textbox(
                label="检测结果",
                lines=15,
                interactive=False
            )
            
            batch_btn.click(
                fn=app.batch_detect,
                inputs=[file_upload],
                outputs=[batch_results, batch_stats]
            )
        
        with gr.Tab("📈 数据分析"):
            with gr.Row():
                threat_dist_plot = gr.Plot(
                    label="威胁类型分布",
                    value=app.create_threat_distribution_chart()
                )
            
            gr.Markdown("""
            ### 威胁类型说明
            
            - **Benign**: 正常网络流量
            - **DoS**: 拒绝服务攻击
            - **DDoS**: 分布式拒绝服务攻击  
            - **Botnet**: 僵尸网络流量
            - **Bruteforce**: 暴力破解攻击
            - **Infiltration**: 渗透攻击
            - **Portscan**: 端口扫描
            - **WebAttacks**: Web应用攻击
            """)
        
        with gr.Tab("⚙️ 系统信息"):
            gr.Markdown(f"""
            ### 系统配置
            
            - **模型**: DeepSeek-v2-lite + minicpm-2b embedding
            - **框架**: UltraRAG
            - **数据集**: CIC-IDS2017, HateSpeech-Davidson, SpamAssassin
            - **启动时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **版本**: v1.0.0
            
            ### 技术栈
            
            - **后端**: Python, UltraRAG, PyTorch
            - **前端**: Gradio
            - **部署**: Docker, GPU加速
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
