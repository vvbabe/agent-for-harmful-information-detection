#!/bin/bash

# 网络威胁检测系统部署脚本

set -e

echo "开始部署网络威胁检测系统..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印彩色输出
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    print_status "Docker环境检查通过"
}

# 检查端口是否被占用
check_ports() {
    local port=7860
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "端口 $port 已被占用，尝试停止现有服务..."
        docker-compose down 2>/dev/null || true
        sleep 2
    fi
}

# 构建和启动服务
deploy_services() {
    print_status "构建Docker镜像..."
    docker-compose build --no-cache
    
    print_status "启动服务..."
    docker-compose up -d
    
    print_status "等待服务启动..."
    sleep 10
    
    # 检查服务状态
    if docker-compose ps | grep -q "Up"; then
        print_status "服务启动成功!"
        echo ""
        echo "应用访问地址:"
        echo "   - 本地: http://localhost:7860"
        echo "   - 网络: http://$(hostname -I | awk '{print $1}'):7860"
        echo ""
        echo "服务状态:"
        docker-compose ps
        echo ""
        echo "查看日志: docker-compose logs -f"
        echo "停止服务: docker-compose down"
    else
        print_error "服务启动失败"
        echo "查看错误日志:"
        docker-compose logs
        exit 1
    fi
}

# 创建必要的目录
setup_directories() {
    print_status "创建必要的目录..."
    mkdir -p logs data/uploads data/results configs
    chmod 755 logs data configs
}

# 主函数
main() {
    echo "========================================"
    echo "  网络威胁检测系统部署脚本 v1.0"
    echo "========================================"
    echo ""
    
    # 检查依赖
    check_docker
    
    # 设置目录
    setup_directories
    
    # 检查端口
    check_ports
    
    # 部署服务
    deploy_services
    
    echo ""
    echo "部署完成！"
    echo ""
    echo "提示："
    echo "   - 首次启动可能需要下载依赖，请耐心等待"
    echo "   - 如遇问题，请查看日志: docker-compose logs -f"
    echo "   - 更新代码后重新部署: ./deploy.sh"
}

# 脚本选项
case "${1:-}" in
    "stop")
        print_status "停止服务..."
        docker-compose down
        print_status "服务已停止"
        ;;
    "restart")
        print_status "重启服务..."
        docker-compose down
        sleep 2
        docker-compose up -d
        print_status "服务已重启"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        print_warning "清理所有容器和镜像..."
        docker-compose down --rmi all --volumes
        print_status "清理完成"
        ;;
    "")
        main
        ;;
    *)
        echo "用法: $0 [stop|restart|logs|status|clean]"
        echo ""
        echo "选项:"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  logs    - 查看日志"
        echo "  status  - 查看状态"
        echo "  clean   - 清理容器和镜像"
        echo "  (无参数) - 部署服务"
        exit 1
        ;;
esac
