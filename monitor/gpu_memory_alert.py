#!/usr/bin/env python3
"""
GPU Memory Alert - 监控GPU显存，当可用显存充足时通过飞书机器人提醒可以跑实验
"""
import subprocess
import time
import sys
import requests
from datetime import datetime
import json
import os

class FeishuNotifier:
    """飞书机器人通知器"""
    
    def __init__(self, config_file='feishu_config.json'):
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file):
        """加载飞书配置"""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建默认配置文件
            default_config = {
                "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key",
                "secret": ""
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            print(f"已创建配置文件: {config_file}")
            print("请编辑配置文件填入飞书机器人webhook后重新运行\n")
            print("获取飞书webhook步骤:")
            print("1. 在飞书群聊中，点击右上角设置")
            print("2. 选择「群机器人」->「添加机器人」->「自定义机器人」")
            print("3. 设置机器人名称和描述")
            print("4. 复制webhook地址，粘贴到配置文件的webhook_url字段")
            print("5. (可选) 如果启用了签名校验，将secret也填入配置文件")
            print("6. 保存配置文件后重新运行脚本\n")
            sys.exit(1)
        
    def send_alert(self, gpu_info, threshold_gb):
        """发送飞书消息提醒"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 构建消息内容
            content_lines = [f"**✅ GPU显存充足提醒 - 可以跑实验了！**", f"", f"**时间:** {timestamp}", f"**可用显存阈值:** {threshold_gb} GB", ""]
            
            for gpu in gpu_info:
                if gpu['free_gb'] >= threshold_gb:
                    content_lines.extend([
                        f"**GPU {gpu['id']}: {gpu['name']}**",
                        f"- 可用显存: {gpu['free_gb']:.2f} GB ✨",
                        f"- 已用显存: {gpu['used_gb']:.2f} GB / {gpu['total_gb']:.2f} GB",
                        f"- 使用率: {gpu['usage_percent']:.1f}%",
                        f"- GPU利用率: {gpu['utilization']}%",
                        f"- 🎉 可用显存达到 {threshold_gb} GB，可以开始实验！",
                        ""
                    ])
            
            content = "\n".join(content_lines)
            
            # 飞书消息格式
            message = {
                "msg_type": "text",
                "content": {
                    "text": content
                }
            }
            
            # 如果配置了签名，添加签名验证
            if self.config.get('secret'):
                import hmac
                import hashlib
                import base64
                
                timestamp = str(int(time.time()))
                string_to_sign = f"{timestamp}\n{self.config['secret']}"
                hmac_code = hmac.new(
                    string_to_sign.encode("utf-8"),
                    digestmod=hashlib.sha256
                ).digest()
                sign = base64.b64encode(hmac_code).decode('utf-8')
                
                message["timestamp"] = timestamp
                message["sign"] = sign
            
            response = requests.post(
                self.config['webhook_url'],
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    return True
                else:
                    print(f"飞书API返回错误: {result}")
                    return False
            else:
                print(f"HTTP请求失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"发送飞书消息失败: {e}")
            return False

def get_gpu_memory_usage():
    """获取所有GPU的显存使用情况（单位：GB）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total,name,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpu_id = int(parts[0])
                mem_used_mb = float(parts[1])
                mem_free_mb = float(parts[2])
                mem_total_mb = float(parts[3])
                gpu_name = parts[4]
                gpu_util = parts[5] if len(parts) > 5 else "N/A"
                
                mem_used_gb = mem_used_mb / 1024
                mem_free_gb = mem_free_mb / 1024
                mem_total_gb = mem_total_mb / 1024
                
                gpu_info.append({
                    'id': gpu_id,
                    'used_gb': mem_used_gb,
                    'free_gb': mem_free_gb,
                    'total_gb': mem_total_gb,
                    'name': gpu_name,
                    'utilization': gpu_util,
                    'usage_percent': (mem_used_gb / mem_total_gb) * 100
                })
        
        return gpu_info
    except subprocess.CalledProcessError:
        print("错误: 无法运行 nvidia-smi")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)



def monitor_gpu_memory(threshold_gb=30, check_interval=60, config_file='feishu_config.json'):
    """
    监控GPU显存使用并发送飞书提醒
    
    Args:
        threshold_gb: 可用显存阈值（GB），当可用显存达到或超过此值时发送提醒
        check_interval: 检查间隔（秒）
        config_file: 飞书配置文件路径
    """
    notifier = FeishuNotifier(config_file)
    
    print(f"🔍 开始监控GPU显存...")
    print(f"📊 可用显存阈值: {threshold_gb} GB (达到此值时提醒可以跑实验)")
    print(f"⏱️  检查间隔: {check_interval} 秒")
    print(f"📱 飞书webhook已配置")
    print(f"⌨️  按 Ctrl+C 停止监控\n")
    
    alert_sent = {}  # 记录每个GPU是否已发送提醒
    
    try:
        while True:
            gpu_info = get_gpu_memory_usage()
            
            # 检查每个GPU
            for gpu in gpu_info:
                gpu_id = gpu['id']
                
                if gpu['free_gb'] >= threshold_gb:
                    # 可用显存达到或超过阈值
                    if not alert_sent.get(gpu_id, False):
                        # 首次达到阈值，发送飞书消息
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n[{timestamp}] 🎉 GPU {gpu_id} 可用显存充足: {gpu['free_gb']:.2f} GB (>= {threshold_gb} GB)")
                        print("📤 正在发送飞书提醒...")
                        
                        if notifier.send_alert([gpu], threshold_gb):
                            print("✅ 飞书消息发送成功 - 可以开始跑实验了！")
                            alert_sent[gpu_id] = True
                        else:
                            print("❌ 飞书消息发送失败")
                else:
                    # 可用显存低于阈值，重置提醒状态
                    if alert_sent.get(gpu_id, False):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n[{timestamp}] GPU {gpu_id} 可用显存降至阈值以下: {gpu['free_gb']:.2f} GB (< {threshold_gb} GB)")
                        alert_sent[gpu_id] = False
            
            # 显示当前状态
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = " | ".join([
                f"GPU{gpu['id']}: 可用{gpu['free_gb']:.1f}GB (已用{gpu['usage_percent']:.0f}%)"
                for gpu in gpu_info
            ])
            print(f"[{timestamp}] {status}", end='\r')
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='监控GPU显存，当可用显存充足时通过飞书机器人提醒可以跑实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用步骤:
  1. 首次运行会生成 feishu_config.json 配置文件
  2. 在飞书群中添加自定义机器人，获取webhook地址
  3. 编辑配置文件，填入webhook地址
  4. 再次运行开始监控

示例:
  # 使用默认配置(可用显存达到30GB时提醒，60秒检查间隔)
  python gpu_memory_alert.py
  
  # 可用显存达到40GB时提醒
  python gpu_memory_alert.py --threshold 40
  
  # 每30秒检查一次
  python gpu_memory_alert.py --interval 30
  
  # 使用自定义配置文件
  python gpu_memory_alert.py --config my_config.json
        """
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=30,
        help='可用显存阈值（GB），当可用显存达到或超过此值时发送提醒，默认30GB'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=60,
        help='检查间隔（秒），默认60秒'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='feishu_config.json',
        help='飞书配置文件路径'
    )
    
    args = parser.parse_args()
    
    monitor_gpu_memory(
        threshold_gb=args.threshold,
        check_interval=args.interval,
        config_file=args.config
    )
