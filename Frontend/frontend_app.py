#!/usr/bin/env python3
"""
 Agent - Gradio Web
UltraRAGagentDeepSeek-v2-liteminicpm-2b embedding
 - 
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

# 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatDetectionFrontend:
    """"""
    
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
         - 
        
        Args:
            input_text: 
            use_rag: RAG
            
        Returns:
            prediction: 
            confidence_scores: 
            explanation: 
        """
        if not input_text.strip():
            return "", {}, ""
        
        # 
        time.sleep(1)
        
        # TODO: API
        # 
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
            # 
            prediction = random.choice(["DoS", "Bruteforce", "WebAttacks", "Botnet"])
            confidence_scores = {threat: random.uniform(0.05, 0.9) for threat in self.threat_types}
            # 
            total = sum(confidence_scores.values())
            confidence_scores = {k: v/total for k, v in confidence_scores.items()}
            # 
            confidence_scores[prediction] = max(confidence_scores.values()) + 0.1
            
        explanation = self._generate_explanation_mock(input_text, prediction, confidence_scores[prediction], use_rag)
        
        return prediction, confidence_scores, explanation
    
    def _generate_explanation_mock(self, text: str, prediction: str, confidence: float, use_rag: bool) -> str:
        """"""
        base_explanation = f" : {prediction} (: {confidence:.1%})\n\n"
        
        threat_descriptions = {
            "Benign": "",
            "DoS": "",
            "DDoS": "",
            "Botnet": "",
            "Bruteforce": "",
            "Infiltration": "",
            "Portscan": "",
            "WebAttacks": "WebSQLXSSWeb"
        }
        
        base_explanation += threat_descriptions.get(prediction, "")
        
        if use_rag:
            base_explanation += "\n\n🤖 RAG: DeepSeek-v2-liteminicpm-2b embedding"
        
        base_explanation += "\n\n :\n"
        if prediction != "Benign":
            base_explanation += "• \n"
            base_explanation += "• \n"
            base_explanation += "• \n"
            base_explanation += "• "
        else:
            base_explanation += "• \n"
            base_explanation += "• "
            
        return base_explanation
    
    def create_confidence_chart(self, confidence_scores: Dict) -> go.Figure:
        """"""
        if not confidence_scores:
            fig = go.Figure()
            fig.add_annotation(text="", x=0.5, y=0.5, showarrow=False)
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
                hovertemplate='<b>%{x}</b><br>: %{y:.1%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(tickformat=".0%"),
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_threat_distribution_chart(self) -> go.Figure:
        """"""
        # CIC-IDS2017
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
                hovertemplate='<b>%{label}</b><br>: %{value:,}<br>: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=" (CIC-IDS2017)",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def batch_detect_mock(self, file) -> Tuple[str, str]:
        """"""
        if file is None:
            return "", ""
        
        try:
            # 
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.jsonl'):
                df = pd.read_json(file.name, lines=True)
            else:
                return "CSVJSONL", ""
            
            if df.empty:
                return "", ""
            
            # 
            time.sleep(2)
            
            #  'text' 
            text_column = 'text' if 'text' in df.columns else df.columns[0]
            
            results = []
            sample_size = min(10, len(df))  # 10
            
            for idx, row in df.head(sample_size).iterrows():
                text = str(row[text_column])
                # TODO: API
                prediction, confidence_scores, _ = self.predict_threat_mock(text)
                
                results.append({
                    '': idx + 1,
                    '': text[:80] + "..." if len(text) > 80 else text,
                    '': prediction,
                    '': f"{max(confidence_scores.values()):.1%}" if confidence_scores else "0%"
                })
            
            # 
            result_df = pd.DataFrame(results)
            
            # 
            threat_counts = {}
            for result in results:
                threat = result['']
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
            
            stats = {
                '': len(results),
                '': len([r for r in results if r[''] != 'Benign']),
                '': threat_counts.get('Benign', 0),
                '': max(threat_counts.items(), key=lambda x: x[1])[0] if threat_counts else 'N/A'
            }
            
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            
            return result_df.to_string(index=False), stats_text
            
        except Exception as e:
            return f": {str(e)}", ""

def create_interface():
    """Gradio"""
    
    app = ThreatDetectionFrontend()
    
    # CSS
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
    
    with gr.Blocks(css=css, title="", theme=gr.themes.Soft()) as demo:
        
        # 
        gr.HTML("""
        <div class="main-header">
            <h1> </h1>
            <p> <strong>UltraRAG</strong>  | <strong>DeepSeek-v2-lite</strong> + <strong>minicpm-2b</strong> embedding</p>
            <p> DoSDDoSBotnetBruteforceInfiltrationPortscanWebAttacks </p>
        </div>
        """)
        
        with gr.Tab(" ", elem_id="detection-tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ")
                    input_text = gr.Textbox(
                        label="",
                        placeholder="...\n\n:\n• TCP SYNIP80\n• IP: 192.168.1.100\n• WebPOSTSQL",
                        lines=6,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        use_rag = gr.Checkbox(
                            label="🧠 RAG",
                            value=True,
                            info=""
                        )
                        detect_btn = gr.Button(
                            " ", 
                            variant="primary", 
                            size="lg",
                            scale=1
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### ")
                    prediction_output = gr.Textbox(
                        label="",
                        interactive=False,
                        container=True
                    )
                    
                    confidence_plot = gr.Plot(
                        label="",
                        container=True
                    )
            
            gr.Markdown("### 🤖 AI ")
            explanation_output = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                container=True
            )
            
            # 
            gr.Markdown("###  ")
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["TCPIP2223"],
                        ["WebPOST'SELECT * FROM''UNION'"],
                        ["IP"],
                        ["1-65535"],
                        ["HTTP GET"]
                    ],
                    inputs=[input_text],
                    label=""
                )
            
            # 
            def process_detection(text, rag):
                prediction, confidence_scores, explanation = app.predict_threat_mock(text, rag)
                chart = app.create_confidence_chart(confidence_scores)
                return prediction, chart, explanation
            
            detect_btn.click(
                fn=process_detection,
                inputs=[input_text, use_rag],
                outputs=[prediction_output, confidence_plot, explanation_output]
            )
        
        with gr.Tab(" ", elem_id="batch-tab"):
            gr.Markdown("### ")
            gr.Markdown("")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label=" ",
                        file_types=[".csv", ".jsonl", ".txt"],
                        info="CSVJSONL < 10MB"
                    )
                    
                    batch_btn = gr.Button(
                        " ", 
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    **:**
                    - CSV: 'text'
                    - JSONL: JSON'text'
                    """)
                
                with gr.Column(scale=1):
                    batch_stats = gr.Textbox(
                        label=" ",
                        lines=8,
                        interactive=False
                    )
            
            batch_results = gr.Textbox(
                label=" ",
                lines=12,
                interactive=False
            )
            
            batch_btn.click(
                fn=app.batch_detect_mock,
                inputs=[file_upload],
                outputs=[batch_results, batch_stats]
            )
        
        with gr.Tab(" ", elem_id="analytics-tab"):
            gr.Markdown("### ")
            
            threat_dist_plot = gr.Plot(
                label="",
                value=app.create_threat_distribution_chart()
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ###  
                    
                    |  |  |  |
                    |---------|------|----------|
                    | **Benign** |  | 🟢  |
                    | **DoS** |  |   |
                    | **DDoS** |  |   |
                    | **Botnet** |  |   |
                    | **Bruteforce** |  | 🟡  |
                    | **Infiltration** |  |   |
                    | **Portscan** |  | 🟡  |
                    | **WebAttacks** | Web | 🟡  |
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ###  
                    
                    - ** **: 
                    - **🧠 **: AI
                    - ** **: RAG
                    - ** **: 
                    - ** **: 
                    - ** **: 8
                    """)
        
        with gr.Tab(" ", elem_id="system-tab"):
            gr.Markdown(f"""
            ##  
            
            ### 
            - **🤖 **: DeepSeek-v2-lite
            - ** **: minicpm-2b embedding  
            - ** **: UltraRAG (Retrieval-Augmented Generation)
            - ** **: Gradio Web
            - ** **: GPU
            
            ### 
            - ** CIC-IDS2017**: 
            - ** HateSpeech-Davidson**: 
            - ** SpamAssassin**: 
            
            ### 
            - ** **: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - ** **: v1.0.0
            - ** **: 
            - ** **: Web
            
            ### 
            - ** **: Docker
            - ** **: 
            - ** **: HTTPS
            - ** **: 
            
            ---
            
            ###  
            
            
            
            ****: [GitHub](https://github.com/vvbabe/agent-for-harmful-information-detection)
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
