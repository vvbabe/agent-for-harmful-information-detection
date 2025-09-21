# 网络威胁检测系统 - 前端界面

## 项目概述

基于UltraRAG的网络威胁检测agent前端界面，集成DeepSeek-v2-lite和minicpm-2b embedding模型展示。提供Web界面进行网络威胁检测、数据分析和可视化展示。

## 技术栈

- **前端框架**: Gradio 4.15+
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly
- **部署**: Docker, Docker Compose
- **语言**: Python 3.10+

## 目录结构

```
Frontend/
├── frontend_app.py          # 主要的Gradio Web应用
├── model_api.py            # 模型API接口 (预留给后端)
├── run_app.py              # 应用启动脚本
├── test_frontend.py        # 前端功能测试脚本
├── Dockerfile              # Docker镜像构建文件
├── docker-compose.yml      # Docker容器编排配置
├── deploy.sh               # 自动化部署脚本
└── FRONTEND_README.md      # 详细使用说明
```

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器
- 可选: Docker 和 Docker Compose

### 本地运行

1. **安装依赖**
   ```bash
   cd Frontend
   pip install -r ../requirements.txt
   ```

2. **测试前端** (可选)
   ```bash
   python test_frontend.py
   ```

3. **启动应用**
   ```bash
   python run_app.py
   ```

4. **访问界面**
   - 打开浏览器访问: http://localhost:7860
   - 默认端口: 7860

### Docker部署

1. **快速部署**
   ```bash
   cd Frontend
   ./deploy.sh
   ```

2. **手动部署**
   ```bash
   # 构建镜像
   docker build -t threat-detection-frontend .
   
   # 运行容器
   docker run -p 7860:7860 threat-detection-frontend
   ```

3. **使用Docker Compose**
   ```bash
   docker-compose up -d
   ```

## 界面功能

### 1. 实时检测

**路径**: 首页 -> "实时检测" 标签页

**功能**:
- 输入网络流量特征描述
- 实时威胁检测和分类
- 置信度可视化分析
- AI生成的威胁分析报告
- 处理建议和安全措施

**支持的威胁类型**:
- Benign (正常流量)
- DoS (拒绝服务攻击)
- DDoS (分布式拒绝服务攻击)
- Botnet (僵尸网络)
- Bruteforce (暴力破解)
- Infiltration (渗透攻击)
- Portscan (端口扫描)
- WebAttacks (Web应用攻击)

### 2. 批量分析

**路径**: "批量分析" 标签页

**功能**:
- 支持CSV和JSONL文件上传
- 批量威胁检测处理
- 统计分析报告
- 结果导出和下载

**文件格式要求**:
- CSV: 需包含'text'列或第一列为文本数据
- JSONL: 每行JSON对象包含'text'字段
- 文件大小建议 < 10MB

### 3. 数据洞察

**路径**: "数据洞察" 标签页

**功能**:
- 威胁类型分布可视化
- 训练数据集统计展示
- 交互式图表和图形
- 威胁类型详细说明

### 4. 系统信息

**路径**: "系统信息" 标签页

**功能**:
- 技术栈信息展示
- 模型配置详情
- 系统运行状态
- 部署信息说明

## API接口规范

### 模型预测接口

前端通过 `model_api.py` 与后端模型交互，主要接口如下:

```python
def predict_threat(input_text: str, use_rag: bool = True) -> Dict[str, Any]:
    """
    单次威胁预测接口
    
    参数:
        input_text: 输入的网络流量特征描述
        use_rag: 是否启用RAG检索增强
    
    返回:
        {
            "prediction": "威胁类型",
            "confidence": 0.85,
            "confidence_scores": {
                "DoS": 0.85,
                "Benign": 0.10,
                ...
            },
            "input_text": "原始输入",
            "use_rag": true,
            "processing_time": 0.5,
            "model_version": "1.0.0"
        }
    """
```

```python
def batch_predict(texts: List[str], use_rag: bool = True) -> List[Dict[str, Any]]:
    """
    批量预测接口
    
    参数:
        texts: 文本列表
        use_rag: 是否启用RAG
    
    返回:
        [预测结果字典列表]
    """
```

### 后端集成说明

1. **替换模拟实现**
   - 修改 `model_api.py` 中的 `_mock_prediction` 方法
   - 实现真实的模型调用逻辑

2. **配置模型端点**
   ```python
   model_api = ModelAPI(
       model_endpoint="http://your-model-server:8000",
       api_key="your-api-key"
   )
   ```

3. **错误处理**
   - API调用异常处理
   - 超时和重试机制
   - 降级处理策略

## 配置选项

### 环境变量

```bash
# 服务器配置
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860

# 模型API配置
export MODEL_ENDPOINT=http://model-server:8000
export API_KEY=your-api-key

# 应用配置
export DEBUG_MODE=false
export MAX_FILE_SIZE=10485760  # 10MB
```

### 应用设置

在 `frontend_app.py` 中可以修改的配置:

```python
# 端口设置
server_port = 7860

# 界面主题
theme = gr.themes.Soft()

# 文件大小限制
max_file_size = "10MB"

# 威胁类型配置
threat_types = [
    "Benign", "DoS", "DDoS", "Botnet", 
    "Bruteforce", "Infiltration", "Portscan", "WebAttacks"
]
```

## 开发指南

### 添加新功能

1. **新增威胁类型**
   - 修改 `ThreatDetectionFrontend` 类中的 `threat_types`
   - 更新颜色配置 `colors`
   - 添加对应的处理逻辑

2. **界面组件修改**
   - 在 `create_interface()` 函数中添加新组件
   - 绑定事件处理函数
   - 更新CSS样式

3. **API接口扩展**
   - 在 `model_api.py` 中添加新方法
   - 更新接口文档
   - 添加相应的测试

### 自定义样式

修改 `create_interface()` 中的CSS:

```python
css = """
.gradio-container {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
}
/* 添加自定义样式 */
"""
```

### 测试和调试

1. **运行测试套件**
   ```bash
   python test_frontend.py
   ```

2. **开发模式启动**
   ```bash
   # 启用调试模式
   python run_app.py --debug
   ```

3. **查看日志**
   ```bash
   tail -f logs/app.log
   ```

## 部署指南

### 生产环境部署

1. **安全配置**
   - 配置HTTPS证书
   - 设置防火墙规则
   - 启用访问控制

2. **性能优化**
   - 配置反向代理 (Nginx)
   - 启用资源压缩
   - 设置缓存策略

3. **监控和日志**
   - 配置应用监控
   - 设置日志轮转
   - 建立告警机制

### 容器化部署

1. **镜像构建**
   ```bash
   docker build -t threat-detection-frontend:latest .
   ```

2. **推送到仓库**
   ```bash
   docker tag threat-detection-frontend:latest your-registry/threat-detection-frontend:latest
   docker push your-registry/threat-detection-frontend:latest
   ```

3. **Kubernetes部署**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: threat-detection-frontend
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: threat-detection-frontend
     template:
       metadata:
         labels:
           app: threat-detection-frontend
       spec:
         containers:
         - name: frontend
           image: your-registry/threat-detection-frontend:latest
           ports:
           - containerPort: 7860
   ```

## 故障排除

### 常见问题

1. **端口占用**
   ```bash
   # 查找占用进程
   lsof -i :7860
   
   # 终止进程
   kill -9 <PID>
   ```

2. **依赖安装失败**
   ```bash
   # 清理pip缓存
   pip cache purge
   
   # 升级pip
   pip install --upgrade pip
   
   # 重新安装依赖
   pip install -r ../requirements.txt --force-reinstall
   ```

3. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   
   # 调整Docker内存限制
   docker run --memory=2g -p 7860:7860 threat-detection-frontend
   ```

### 日志分析

1. **应用日志**
   ```bash
   # 实时查看日志
   tail -f logs/app.log
   
   # 查看错误日志
   grep ERROR logs/app.log
   ```

2. **Docker日志**
   ```bash
   # 查看容器日志
   docker logs threat-detection-app
   
   # 实时跟踪日志
   docker logs -f threat-detection-app
   ```

## 性能优化

### 前端优化

1. **组件懒加载**
   - 按需加载图表组件
   - 延迟初始化复杂组件

2. **数据处理优化**
   - 批量数据分页处理
   - 异步数据加载
   - 缓存计算结果

3. **界面响应优化**
   - 减少不必要的重新渲染
   - 优化CSS选择器
   - 压缩静态资源

### 后端优化

1. **API调用优化**
   - 连接池复用
   - 请求批处理
   - 超时控制

2. **缓存策略**
   - Redis缓存预测结果
   - 本地缓存模型输出
   - CDN静态资源

## 安全考虑

### 输入验证

1. **文件上传安全**
   - 文件类型验证
   - 文件大小限制
   - 恶意文件检测

2. **输入过滤**
   - XSS防护
   - SQL注入防护
   - 命令注入防护

### 访问控制

1. **认证授权**
   - 用户身份验证
   - 角色权限控制
   - API访问限制

2. **网络安全**
   - HTTPS强制
   - 跨域请求控制
   - 防火墙配置

## 更新和维护

### 版本管理

1. **代码版本**
   - 使用语义化版本号
   - 维护变更日志
   - 标记发布版本

2. **依赖更新**
   - 定期更新依赖包
   - 安全补丁应用
   - 兼容性测试

### 备份策略

1. **代码备份**
   - Git仓库备份
   - 配置文件备份
   - 部署脚本备份

2. **数据备份**
   - 应用日志备份
   - 用户数据备份
   - 配置数据备份

## 联系支持

- **前端开发团队**: 负责界面开发和维护
- **后端模型团队**: 负责模型集成和API实现
- **运维团队**: 负责部署和基础设施

---

**注意**: 当前版本为前端展示版本，使用模拟数据进行演示。后端模型团队完成开发后，只需要替换 `model_api.py` 中的实现即可实现完整功能。
