#!/usr/bin/env python3
"""
网络威胁检测 Agent - Gradio Web界面
基于UltraRAG的网络威胁检测agent，集成DeepSeek-v2-lite和minicpm-2b embedding模型
前端展示界面 - 为后端模型预留接口
"""

import gradio as gr
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import time
from typing import Dict, List, Tuple, Any
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatDetectionFrontend:
    """网络威胁检测前端应用"""
    
    def __init__(self):
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
        
    def predict_threat_mock(self, input_text: str, use_rag: bool = True) -> Tuple[str, Dict, str]:
        """
        模拟威胁预测 - 这里将被替换为真实的模型调用
        
        Args:
            input_text: 输入的网络流量特征或描述
            use_rag: 是否使用RAG检索增强
            
        Returns:
            prediction: 预测结果
            confidence_scores: 置信度分数
            explanation: 解释文本
        """
        if not input_text.strip():
            return "请输入有效的网络流量特征", {}, ""
        
        # 模拟处理延迟
        time.sleep(1)
        
        # TODO: 这里将被替换为真实的模型API调用
        # 模拟预测逻辑
        text_lower = input_text.lower()
        
        if any(word in text_lower for word in ["normal", "regular", "legitimate"]):
            prediction = "Benign"
            confidence_scores = {
                "Benign": 0.85, "DoS": 0.05, "DDoS": 0.02, "Botnet": 0.02,
                "Bruteforce": 0.02, "Infiltration": 0.02, "Portscan": 0.01, "WebAttacks": 0.01
            }
        elif any(word in text_lower for word in ["ddos", "distributed", "coordinated"]):
            prediction = "DDoS"
            confidence_scores = {
                "DDoS": 0.82, "DoS": 0.10, "Benign": 0.03, "Botnet": 0.02,
                "Bruteforce": 0.01, "Infiltration": 0.01, "Portscan": 0.01, "WebAttacks": 0.00
            }
        elif any(word in text_lower for word in ["scan", "probe", "nmap"]):
            prediction = "Portscan"
            confidence_scores = {
                "Portscan": 0.78, "Infiltration": 0.12, "Benign": 0.05, "DoS": 0.02,
                "DDoS": 0.01, "Botnet": 0.01, "Bruteforce": 0.01, "WebAttacks": 0.00
            }
        else:
            # 随机选择一个威胁类型作为示例
            prediction = random.choice(["DoS", "Bruteforce", "WebAttacks", "Botnet"])
            confidence_scores = {threat: random.uniform(0.05, 0.9) for threat in self.threat_types}
            # 归一化
            total = sum(confidence_scores.values())
            confidence_scores = {k: v/total for k, v in confidence_scores.items()}
            # 确保预测类型有最高分
            confidence_scores[prediction] = max(confidence_scores.values()) + 0.1
            
        explanation = self._generate_explanation_mock(input_text, prediction, confidence_scores[prediction], use_rag)
        
        return prediction, confidence_scores, explanation
    
    def _generate_explanation_mock(self, text: str, prediction: str, confidence: float, use_rag: bool) -> str:
        """生成模拟解释"""
        base_explanation = f"🎯 检测结果: {prediction} (置信度: {confidence:.1%})\n\n"
        
        threat_descriptions = {
            "Benign": "分析结果显示该网络流量为正常流量，未发现明显的威胁特征。建议继续监控。",
            "DoS": "检测到拒绝服务攻击特征，可能存在资源耗尽攻击行为。建议立即启动防护措施。",
            "DDoS": "检测到分布式拒绝服务攻击特征，来自多个源的协调攻击。需要紧急响应。",
            "Botnet": "检测到僵尸网络活动特征，可能存在被感染的设备通信。建议隔离可疑主机。",
            "Bruteforce": "检测到暴力破解攻击特征，存在大量登录尝试行为。建议加强访问控制。",
            "Infiltration": "检测到渗透攻击特征，可能存在未授权访问尝试。需要详细分析攻击路径。",
            "Portscan": "检测到端口扫描活动，攻击者可能在进行网络侦察。建议监控后续活动。",
            "WebAttacks": "检测到Web应用攻击特征，可能存在SQL注入或XSS等攻击。需要检查Web应用安全。"
        }
        
        base_explanation += threat_descriptions.get(prediction, "未知威胁类型")
        
        if use_rag:
            base_explanation += "\n\n🤖 RAG增强分析: 基于DeepSeek-v2-lite和minicpm-2b embedding模型，结合知识库检索增强了检测准确性。"
        
        base_explanation += "\n\n💡 处理建议:\n"
        if prediction != "Benign":
            base_explanation += "• 立即启动安全事件响应流程\n"
            base_explanation += "• 收集和保存相关日志证据\n"
            base_explanation += "• 通知安全团队进行详细分析\n"
            base_explanation += "• 考虑临时阻断可疑流量源"
        else:
            base_explanation += "• 继续保持网络监控\n"
            base_explanation += "• 定期更新威胁检测规则"
            
        return base_explanation
    
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
                text=[f"{score:.1%}" for score in scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>置信度: %{y:.1%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="威胁类型置信度分布",
            xaxis_title="威胁类型",
            yaxis_title="置信度",
            yaxis=dict(tickformat=".0%"),
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_threat_distribution_chart(self) -> go.Figure:
        """创建威胁类型分布图"""
        # CIC-IDS2017数据集的真实分布
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
        
        colors_list = [self.colors.get(threat, "#95A5A6") for threat in sample_data.keys()]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(sample_data.keys()),
                values=list(sample_data.values()),
                marker_colors=colors_list,
                hole=0.3,
                hovertemplate='<b>%{label}</b><br>数量: %{value:,}<br>占比: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="训练数据集威胁类型分布 (CIC-IDS2017)",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def batch_detect_mock(self, file) -> Tuple[str, str]:
        """模拟批量检测"""
        if file is None:
            return "请上传文件", ""
        
        try:
            # 读取文件
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.jsonl'):
                df = pd.read_json(file.name, lines=True)
            else:
                return "不支持的文件格式，请上传CSV或JSONL文件", ""
            
            if df.empty:
                return "文件为空", ""
            
            # 模拟处理延迟
            time.sleep(2)
            
            # 假设文本列名为 'text' 或第一列
            text_column = 'text' if 'text' in df.columns else df.columns[0]
            
            results = []
            sample_size = min(10, len(df))  # 限制处理前10行
            
            for idx, row in df.head(sample_size).iterrows():
                text = str(row[text_column])
                # TODO: 这里将调用真实的模型API
                prediction, confidence_scores, _ = self.predict_threat_mock(text)
                
                results.append({
                    '序号': idx + 1,
                    '输入文本': text[:80] + "..." if len(text) > 80 else text,
                    '预测结果': prediction,
                    '置信度': f"{max(confidence_scores.values()):.1%}" if confidence_scores else "0%"
                })
            
            # 创建结果表格
            result_df = pd.DataFrame(results)
            
            # 生成统计信息
            threat_counts = {}
            for result in results:
                threat = result['预测结果']
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
            
            stats = {
                '处理总数': len(results),
                '检测到威胁': len([r for r in results if r['预测结果'] != 'Benign']),
                '正常流量': threat_counts.get('Benign', 0),
                '最多威胁类型': max(threat_counts.items(), key=lambda x: x[1])[0] if threat_counts else 'N/A'
            }
            
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            
            return result_df.to_string(index=False), stats_text
            
        except Exception as e:
            return f"处理失败: {str(e)}", ""

def create_interface():
    """创建Gradio界面"""
    
    app = ThreatDetectionFrontend()
    
    # 自定义CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        max-width: 1200px;
        margin: 0 auto;
    }
    .threat-high { 
        background: linear-gradient(90deg, #ff6b6b, #ee5a52) !important;
        color: white !important;
    }
    .threat-medium { 
        background: linear-gradient(90deg, #ffa502, #ff6348) !important;
        color: white !important;
    }
    .threat-low { 
        background: linear-gradient(90deg, #2ed573, #1e90ff) !important;
        color: white !important;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="网络威胁检测系统", theme=gr.themes.Soft()) as demo:
        
        # 主标题
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ 智能网络威胁检测系统</h1>
            <p>基于 <strong>UltraRAG</strong> 架构 | <strong>DeepSeek-v2-lite</strong> + <strong>minicpm-2b</strong> embedding</p>
            <p>实时检测 DoS、DDoS、Botnet、Bruteforce、Infiltration、Portscan、WebAttacks 等网络威胁</p>
        </div>
        """)
        
        with gr.Tab("🔍 实时检测", elem_id="detection-tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入网络流量特征")
                    input_text = gr.Textbox(
                        label="流量描述",
                        placeholder="请输入网络流量特征、异常行为描述或日志信息...\n\n示例:\n• 发现大量TCP SYN包，源IP分散，目标端口80\n• 检测到重复登录失败，来源IP: 192.168.1.100\n• Web服务器收到异常POST请求，包含SQL语句",
                        lines=6,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        use_rag = gr.Checkbox(
                            label="🧠 启用RAG检索增强",
                            value=True,
                            info="使用知识库提升检测准确性"
                        )
                        detect_btn = gr.Button(
                            "🚀 开始检测", 
                            variant="primary", 
                            size="lg",
                            scale=1
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 检测结果")
                    prediction_output = gr.Textbox(
                        label="威胁类型",
                        interactive=False,
                        container=True
                    )
                    
                    confidence_plot = gr.Plot(
                        label="置信度分析",
                        container=True
                    )
            
            gr.Markdown("### 🤖 AI 分析报告")
            explanation_output = gr.Textbox(
                label="详细分析",
                lines=6,
                interactive=False,
                container=True
            )
            
            # 预设示例
            gr.Markdown("### 💡 示例输入")
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["发现大量TCP连接尝试，源IP重复，目标端口集中在22和23，连接频率异常"],
                        ["Web服务器日志显示大量POST请求包含'SELECT * FROM'和'UNION'关键词"],
                        ["网络中检测到多台主机同时向外部IP发送相同数据包"],
                        ["扫描工具检测到端口1-65535的顺序探测活动"],
                        ["正常的HTTP GET请求访问公司官网首页"]
                    ],
                    inputs=[input_text],
                    label="点击使用示例"
                )
            
            # 检测按钮事件
            def process_detection(text, rag):
                prediction, confidence_scores, explanation = app.predict_threat_mock(text, rag)
                chart = app.create_confidence_chart(confidence_scores)
                return prediction, chart, explanation
            
            detect_btn.click(
                fn=process_detection,
                inputs=[input_text, use_rag],
                outputs=[prediction_output, confidence_plot, explanation_output]
            )
        
        with gr.Tab("📊 批量分析", elem_id="batch-tab"):
            gr.Markdown("### 批量威胁检测")
            gr.Markdown("上传包含网络流量数据的文件，系统将批量进行威胁检测分析")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="📁 上传数据文件",
                        file_types=[".csv", ".jsonl", ".txt"],
                        info="支持CSV、JSONL格式，建议文件大小 < 10MB"
                    )
                    
                    batch_btn = gr.Button(
                        "🔄 开始批量检测", 
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    **文件格式说明:**
                    - CSV: 需包含'text'列或将第一列作为文本数据
                    - JSONL: 每行一个JSON对象，需包含'text'字段
                    """)
                
                with gr.Column(scale=1):
                    batch_stats = gr.Textbox(
                        label="📈 检测统计",
                        lines=8,
                        interactive=False
                    )
            
            batch_results = gr.Textbox(
                label="🔍 详细结果",
                lines=12,
                interactive=False
            )
            
            batch_btn.click(
                fn=app.batch_detect_mock,
                inputs=[file_upload],
                outputs=[batch_results, batch_stats]
            )
        
        with gr.Tab("📈 数据洞察", elem_id="analytics-tab"):
            gr.Markdown("### 训练数据集分析")
            
            threat_dist_plot = gr.Plot(
                label="威胁类型分布统计",
                value=app.create_threat_distribution_chart()
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🎯 威胁类型详解
                    
                    | 威胁类型 | 描述 | 危险级别 |
                    |---------|------|----------|
                    | **Benign** | 正常网络流量 | 🟢 低 |
                    | **DoS** | 拒绝服务攻击 | 🔴 高 |
                    | **DDoS** | 分布式拒绝服务攻击 | 🔴 高 |
                    | **Botnet** | 僵尸网络活动 | 🔴 高 |
                    | **Bruteforce** | 暴力破解攻击 | 🟡 中 |
                    | **Infiltration** | 渗透攻击 | 🔴 高 |
                    | **Portscan** | 端口扫描 | 🟡 中 |
                    | **WebAttacks** | Web应用攻击 | 🟡 中 |
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### ⚡ 系统特性
                    
                    - **🚀 实时检测**: 毫秒级响应威胁识别
                    - **🧠 智能分析**: AI驱动的深度学习检测
                    - **📚 知识增强**: RAG技术提升准确性
                    - **🔄 批量处理**: 支持大规模数据分析
                    - **📊 可视化**: 直观的威胁分析图表
                    - **🛡️ 多威胁**: 覆盖8大类网络威胁
                    """)
        
        with gr.Tab("⚙️ 系统信息", elem_id="system-tab"):
            gr.Markdown(f"""
            ## 🖥️ 系统配置信息
            
            ### 核心技术栈
            - **🤖 生成模型**: DeepSeek-v2-lite
            - **📊 嵌入模型**: minicpm-2b embedding  
            - **🏗️ 框架**: UltraRAG (Retrieval-Augmented Generation)
            - **🎨 前端**: Gradio Web界面
            - **⚡ 加速**: GPU优化推理
            
            ### 训练数据集
            - **🔍 CIC-IDS2017**: 网络入侵检测数据集
            - **💬 HateSpeech-Davidson**: 恶意内容检测
            - **📧 SpamAssassin**: 垃圾邮件检测
            
            ### 系统状态
            - **🕐 启动时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **📱 界面版本**: v1.0.0
            - **🔄 状态**: 运行中
            - **🌐 访问模式**: Web界面
            
            ### 部署信息
            - **🐳 容器化**: Docker支持
            - **☁️ 云部署**: 支持服务器部署
            - **🔒 安全**: HTTPS加密传输
            - **📈 监控**: 实时性能监控
            
            ---
            
            ### 📞 技术支持
            
            如遇到技术问题，请联系开发团队或查看项目文档。
            
            **项目仓库**: [GitHub](https://github.com/vvbabe/agent-for-harmful-information-detection)
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
