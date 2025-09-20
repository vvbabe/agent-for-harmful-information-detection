# 网络威胁检测前端界面

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动前端
```bash
python run_app.py
```

### 3. 访问界面
浏览器打开: http://localhost:7860

## 📁 前端文件说明

- `frontend_app.py` - 主要的Gradio前端应用
- `model_api.py` - 模型API接口 (预留给后端团队)
- `run_app.py` - 启动脚本
- `requirements.txt` - 依赖包列表

## 🎨 界面功能

### 1. 实时检测
- 单次威胁检测
- 输入网络流量特征描述
- 实时显示检测结果和置信度
- AI分析报告

### 2. 批量分析  
- 支持CSV、JSONL文件上传
- 批量威胁检测
- 统计分析报告

### 3. 数据洞察
- 威胁类型分布可视化
- 训练数据集统计
- 威胁类型详解

### 4. 系统信息
- 技术栈信息
- 系统配置
- 部署状态

## 🔧 后端集成说明

### 模型API接口

后端团队需要实现 `model_api.py` 中的接口:

```python
def predict_threat(input_text: str, use_rag: bool = True) -> Dict[str, Any]:
    """
    威胁预测接口
    
    Args:
        input_text: 输入文本
        use_rag: 是否使用RAG增强
        
    Returns:
        {
            "prediction": "威胁类型",
            "confidence": 0.85,
            "confidence_scores": {"DoS": 0.85, "Benign": 0.10, ...},
            "explanation": "分析解释"
        }
    """
```

### API调用示例

```python
# 单次预测
result = model_api.predict_threat("检测到大量TCP连接")

# 批量预测  
results = model_api.batch_predict(["文本1", "文本2"])
```

## 🐳 Docker部署

### 构建镜像
```bash
docker build -t threat-detection-frontend .
```

### 运行容器
```bash
docker run -p 7860:7860 threat-detection-frontend
```

### 使用Docker Compose
```bash
docker-compose up -d
```

## 🔒 安全配置

### 生产环境部署
1. 设置HTTPS
2. 配置防火墙
3. 启用访问日志
4. 设置资源限制

### 环境变量
```bash
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export MODEL_ENDPOINT=http://model-server:8000
export API_KEY=your-api-key
```

## 🛠️ 开发说明

### 修改界面
主要界面代码在 `frontend_app.py` 的 `create_interface()` 函数中。

### 添加新功能
1. 在 `ThreatDetectionFrontend` 类中添加方法
2. 在界面中添加对应的组件
3. 绑定事件处理函数

### 自定义样式
修改 `create_interface()` 中的 `css` 变量。

## 📊 监控和日志

### 访问日志
```bash
tail -f logs/app.log
```

### 性能监控
- 内存使用: `docker stats`
- CPU使用: `top`
- 网络流量: `netstat`

## 🔧 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   lsof -i :7860
   kill -9 <PID>
   ```

2. **依赖包问题**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **权限问题**
   ```bash
   chmod +x run_app.py
   chmod +x deploy.sh
   ```

### 日志查看
```bash
# 应用日志
tail -f logs/app.log

# Docker日志
docker-compose logs -f
```

## 📞 技术支持

如遇问题请联系:
- 前端开发: 当前团队
- 后端模型: 合作团队
- 部署运维: DevOps团队

---

**注意**: 当前版本为前端展示版本，模型接口使用模拟数据。后端模型准备好后，只需替换 `model_api.py` 中的实现即可。
