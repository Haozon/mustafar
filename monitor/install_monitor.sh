#!/bin/bash
# 安装脚本监控工具到全局

set -e

echo "=========================================="
echo "安装脚本监控工具"
echo "=========================================="
echo ""

# 获取当前目录
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 目标目录
INSTALL_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/script_monitor"

# 创建目录
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"

# 复制脚本
echo "1. 复制监控脚本..."
cp "$CURRENT_DIR/monitor_script.py" "$INSTALL_DIR/monitor_script"
chmod +x "$INSTALL_DIR/monitor_script"
echo "   ✓ 已安装到: $INSTALL_DIR/monitor_script"

# 复制配置文件（如果存在）
if [ -f "$CURRENT_DIR/feishu_config.json" ]; then
    echo "2. 复制配置文件..."
    cp "$CURRENT_DIR/feishu_config.json" "$CONFIG_DIR/feishu_config.json"
    echo "   ✓ 配置文件: $CONFIG_DIR/feishu_config.json"
else
    echo "2. 创建默认配置文件..."
    cat > "$CONFIG_DIR/feishu_config.json" << 'EOF'
{
  "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key",
  "secret": ""
}
EOF
    echo "   ✓ 已创建: $CONFIG_DIR/feishu_config.json"
    echo "   ⚠️  请编辑配置文件填入飞书webhook"
fi

# 修改脚本中的配置文件路径
echo "3. 更新配置文件路径..."
sed -i "s|'feishu_config.json'|'$CONFIG_DIR/feishu_config.json'|g" "$INSTALL_DIR/monitor_script"

# 添加到PATH（如果需要）
echo ""
echo "4. 检查PATH..."
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "   需要添加 $INSTALL_DIR 到 PATH"
    
    # 检测shell类型
    if [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi
    
    echo "   正在添加到 $SHELL_RC ..."
    echo "" >> "$SHELL_RC"
    echo "# Script Monitor Tool" >> "$SHELL_RC"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
    
    echo "   ✓ 已添加到 $SHELL_RC"
    echo "   请运行: source $SHELL_RC"
else
    echo "   ✓ PATH 已包含 $INSTALL_DIR"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  monitor_script \"bash your_script.sh\""
echo ""
echo "配置文件位置:"
echo "  $CONFIG_DIR/feishu_config.json"
echo ""
echo "如果PATH未生效，请运行:"
echo "  source ~/.bashrc  # 或 source ~/.zshrc"
echo ""
