# 脚本监控工具 - 飞书通知版

一个简单的bash函数工具，用于监控脚本运行状态，并在完成后通过飞书机器人发送通知。

## 功能特点

- ✅ 脚本开始时发送飞书通知
- ⏱️ 自动计算运行时长
- 📊 显示运行状态（成功/失败）
- 🔔 完成后自动发送飞书通知
- 🌍 全局可用，在任何目录都能使用
- 🚀 使用简单：`mon bash your_script.sh`

## 安装步骤

### 1. 创建配置目录

```bash
mkdir -p ~/.config/script_monitor
```

### 2. 创建飞书配置文件

创建文件 `~/.config/script_monitor/feishu_config.json`：

```json
{
  "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key",
  "secret": ""
}
```

**获取飞书webhook步骤：**
1. 在飞书群聊中，点击右上角设置
2. 选择「群机器人」→「添加机器人」→「自定义机器人」
3. 设置机器人名称（比如"脚本监控"）
4. 复制webhook地址，粘贴到配置文件的 `webhook_url` 字段
5. （可选）如果启用了签名校验，将secret也填入配置文件

### 3. 添加监控函数到 .bashrc

将以下内容添加到 `~/.bashrc` 文件末尾：

```bash
# ============================================================
# 脚本监控工具 - 飞书通知
# ============================================================

# 配置文件路径
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
```

### 4. 重新加载配置

```bash
source ~/.bashrc
```

## 使用方法

### 基本用法

```bash
# 使用完整命令
monitor bash your_script.sh

# 使用简短别名
mon bash your_script.sh

# 监控Python脚本
mon python train.py --epochs 100

# 监控任何命令
mon "cd /path && bash script.sh"
```

### 实际示例

```bash
# 监控深度学习训练
mon python train.py --model resnet50 --epochs 100

# 监控数据处理脚本
mon bash process_data.sh

# 监控测试脚本
mon bash test_prefill_only.sh

# 监控长时间运行的任务
mon "bash step1.sh && bash step2.sh && bash step3.sh"
```

## 通知示例

### 开始通知
```
🚀 脚本开始运行

命令: bash test_prefill_only.sh
开始时间: 2026-01-30 15:30:00
```

### 成功通知
```
✅ 脚本运行成功

命令: bash test_prefill_only.sh
开始时间: 2026-01-30 15:30:00
结束时间: 2026-01-30 17:45:30
总耗时: 2小时15分30秒
状态: 成功完成
```

### 失败通知
```
❌ 脚本运行失败

命令: bash test_script.sh
开始时间: 2026-01-30 15:30:00
结束时间: 2026-01-30 15:35:20
总耗时: 5分20秒
退出码: 1
状态: 运行失败
```

## 快速安装脚本

如果你想一键安装，可以使用以下脚本：

```bash
#!/bin/bash
# 快速安装脚本监控工具

# 创建配置目录
mkdir -p ~/.config/script_monitor

# 创建配置文件（如果不存在）
if [ ! -f ~/.config/script_monitor/feishu_config.json ]; then
    cat > ~/.config/script_monitor/feishu_config.json << 'EOF'
{
  "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key",
  "secret": ""
}
EOF
    echo "✓ 已创建配置文件: ~/.config/script_monitor/feishu_config.json"
    echo "⚠️  请编辑配置文件填入飞书webhook"
fi

# 检查 .bashrc 中是否已存在
if grep -q "# 脚本监控工具 - 飞书通知" ~/.bashrc; then
    echo "✓ 监控函数已存在于 .bashrc"
else
    echo "正在添加监控函数到 .bashrc..."
    cat >> ~/.bashrc << 'EOF'

# ============================================================
# 脚本监控工具 - 飞书通知
# ============================================================

# 配置文件路径
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
EOF
    echo "✓ 已添加监控函数到 .bashrc"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 编辑配置文件填入飞书webhook:"
echo "   vim ~/.config/script_monitor/feishu_config.json"
echo ""
echo "2. 重新加载配置:"
echo "   source ~/.bashrc"
echo ""
echo "3. 开始使用:"
echo "   mon bash your_script.sh"
echo ""
```

保存为 `install_monitor.sh`，然后运行：

```bash
bash install_monitor.sh
```

## 故障排查

### 问题1：提示配置文件不存在

**解决方法：**
```bash
# 检查配置文件是否存在
ls -la ~/.config/script_monitor/feishu_config.json

# 如果不存在，创建它
mkdir -p ~/.config/script_monitor
vim ~/.config/script_monitor/feishu_config.json
```

### 问题2：没有收到飞书通知

**检查步骤：**
1. 确认webhook地址正确
2. 测试webhook是否有效：
```bash
curl -X POST "你的webhook地址" \
  -H 'Content-Type: application/json' \
  -d '{"msg_type":"text","content":{"text":"测试消息"}}'
```
3. 检查网络连接

### 问题3：命令找不到

**解决方法：**
```bash
# 重新加载 .bashrc
source ~/.bashrc

# 检查函数是否加载
type monitor
type mon
```

## 卸载

如果需要卸载，执行以下步骤：

```bash
# 1. 从 .bashrc 中删除相关内容
vim ~/.bashrc
# 删除 "# 脚本监控工具 - 飞书通知" 到 "alias mon='monitor'" 之间的所有内容

# 2. 删除配置文件
rm -rf ~/.config/script_monitor

# 3. 重新加载
source ~/.bashrc
```

## 高级用法

### 自定义配置文件位置

如果你想使用不同的配置文件，可以修改 `MONITOR_CONFIG` 变量：

```bash
# 在 .bashrc 中修改
MONITOR_CONFIG="/path/to/your/config.json"
```

### 只发送完成通知（不发送开始通知）

修改 `monitor` 函数，注释掉开始通知部分：

```bash
# 注释掉这一行
# send_feishu "脚本开始运行" "命令: $cmd\n开始时间: $start_datetime" "🚀"
```

### 添加更多信息到通知

可以在通知内容中添加主机名、用户名等信息：

```bash
local content="命令: $cmd\n主机: $(hostname)\n用户: $(whoami)\n开始时间: $start_datetime\n..."
```

## 许可证

MIT License

## 作者

Created with ❤️ by Kiro
