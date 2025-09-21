#把模型参数转成safetensors
from transformers import AutoModel
model = AutoModel.from_pretrained("../models/MiniCPM-2B", trust_remote_code=True)
model.save_pretrained("../models/MiniCPM-2B-safetensors", safe_serialization=True)
