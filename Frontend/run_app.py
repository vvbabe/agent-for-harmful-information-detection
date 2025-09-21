#!/usr/bin/env python3
"""
网络威胁检测系统启动脚本 - 前端版本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查前端依赖"""
    required_packages = [
        'gradio',
        'pandas', 
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("正在安装依赖包...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade"] + missing_packages
            )
            print("依赖包安装完成")
        except subprocess.CalledProcessError:
            print("依赖包安装失败，请手动运行: pip install -r ../requirements.txt")
            return False
    
    print("所有依赖包检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    directories = [
        "logs",
        "temp", 
        "data/uploads",
        "data/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """主函数"""
    print("正在启动网络威胁检测前端系统...")
    
    # 创建目录
    create_directories()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 启动应用
    try:
        print("正在启动Gradio前端界面...")
        
        # 导入并启动前端应用
        from frontend_app import create_interface
        
        demo = create_interface()
        
        print("前端应用已启动!")
        print("本地访问: http://localhost:7860")
        print("自动重载已启用")
        print("按 Ctrl+C 停止服务")
        print("\n提示: 这是前端展示版本，模型接口已预留给后端团队")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
