# 

##  

### 1. 
```bash
pip install -r requirements.txt
```

### 2. 
```bash
python run_app.py
```

### 3. 
: http://localhost:7860

##  

- `frontend_app.py` - Gradio
- `model_api.py` - API ()
- `run_app.py` - 
- `requirements.txt` - 

##  

### 1. 
- 
- 
- 
- AI

### 2.   
- CSVJSONL
- 
- 

### 3. 
- 
- 
- 

### 4. 
- 
- 
- 

##  

### API

 `model_api.py` :

```python
def predict_threat(input_text: str, use_rag: bool = True) -> Dict[str, Any]:
    """
    
    
    Args:
        input_text: 
        use_rag: RAG
        
    Returns:
        {
            "prediction": "",
            "confidence": 0.85,
            "confidence_scores": {"DoS": 0.85, "Benign": 0.10, ...},
            "explanation": ""
        }
    """
```

### API

```python
# 
result = model_api.predict_threat("TCP")

#   
results = model_api.batch_predict(["1", "2"])
```

##  Docker

### 
```bash
docker build -t threat-detection-frontend .
```

### 
```bash
docker run -p 7860:7860 threat-detection-frontend
```

### Docker Compose
```bash
docker-compose up -d
```

##  

### 
1. HTTPS
2. 
3. 
4. 

### 
```bash
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export MODEL_ENDPOINT=http://model-server:8000
export API_KEY=your-api-key
```

##  

### 
 `frontend_app.py`  `create_interface()` 

### 
1.  `ThreatDetectionFrontend` 
2. 
3. 

### 
 `create_interface()`  `css` 

##  

### 
```bash
tail -f logs/app.log
```

### 
- : `docker stats`
- CPU: `top`
- : `netstat`

##  

### 

1. ****
   ```bash
   lsof -i :7860
   kill -9 <PID>
   ```

2. ****
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. ****
   ```bash
   chmod +x run_app.py
   chmod +x deploy.sh
   ```

### 
```bash
# 
tail -f logs/app.log

# Docker
docker-compose logs -f
```

##  

:
- : 
- : 
- : DevOps

---

****:  `model_api.py` 
