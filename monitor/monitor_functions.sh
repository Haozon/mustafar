#!/bin/bash
# 添加到 ~/.bashrc 的监控函数

# 配置文件路径（修改为你的实际路径）
MONITOR_CONFIG="$HOME/.config/script_monitor/feishu_config.json"

# 发送飞书消息的函数
send_feishu() {
    local title="$1"
    local content="$2"
    local emoji="${3:-✅}"
    
    if [ ! -f "$MONITOR_CONFIG" ]; then
        echo "错误: 配置文件不存在: $MONITOR_CONFIG"
        return 1
    fi
    
    local webhook_url=$(python3 -c "import json; print(json.load(open('$MONITOR_CONFIG'))['webhook_url'])")
    
    local message=$(cat <<EOF
{
  "msg_type": "text",
  "content": {
    "text": "$emoji $title\n\n$content"
  }
}
EOF
)
    
    curl -s -X POST "$webhook_url" \
        -H 'Content-Type: application/json' \
        -d "$message" > /dev/null
}

# 监控命令运行的函数
monitor() {
    local cmd="$@"
    
    if [ -z "$cmd" ]; then
        echo "用法: monitor <命令>"
        echo "示例: monitor bash test.sh"
        return 1
    fi
    
    local start_time=$(date +%s)
    local start_datetime=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "=========================================="
    echo "开始监控: $cmd"
    echo "开始时间: $start_datetime"
    echo "=========================================="
    echo ""
    
    # 发送开始通知
    send_feishu "脚本开始运行" "命令: $cmd\n开始时间: $start_datetime" "🚀"
    
    # 运行命令
    eval "$cmd"
    local exit_code=$?
    
    local end_time=$(date +%s)
    local end_datetime=$(date "+%Y-%m-%d %H:%M:%S")
    local duration=$((end_time - start_time))
    
    # 格式化时长
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    if [ $hours -gt 0 ]; then
        local duration_str="${hours}小时${minutes}分${seconds}秒"
    elif [ $minutes -gt 0 ]; then
        local duration_str="${minutes}分${seconds}秒"
    else
        local duration_str="${seconds}秒"
    fi
    
    echo ""
    echo "=========================================="
    echo "运行完成"
    echo "结束时间: $end_datetime"
    echo "总耗时: $duration_str"
    echo "退出码: $exit_code"
    echo "=========================================="
    
    # 发送完成通知
    if [ $exit_code -eq 0 ]; then
        local content="命令: $cmd\n开始时间: $start_datetime\n结束时间: $end_datetime\n总耗时: $duration_str\n状态: 成功完成"
        send_feishu "脚本运行成功" "$content" "✅"
        echo "✓ 已发送成功通知到飞书"
    else
        local content="命令: $cmd\n开始时间: $start_datetime\n结束时间: $end_datetime\n总耗时: $duration_str\n退出码: $exit_code\n状态: 运行失败"
        send_feishu "脚本运行失败" "$content" "❌"
        echo "✓ 已发送失败通知到飞书"
    fi
    
    return $exit_code
}

# 简短别名
alias mon='monitor'

echo "✓ 监控函数已加载"
echo "  使用方法: monitor <命令>"
echo "  或使用别名: mon <命令>"
