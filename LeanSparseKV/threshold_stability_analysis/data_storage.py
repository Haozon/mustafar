#!/usr/bin/env python3
"""
阈值数据存储和管理系统 (ThresholdDataStorage)

负责阈值数据的序列化、加载、查询和管理功能。
"""

import os
import json
import pickle
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from data_collector import ThresholdRecord, CollectionConfig

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """数据存储配置"""
    storage_dir: str
    use_database: bool = True
    use_json_backup: bool = True
    use_pickle_backup: bool = True
    compression: bool = True
    auto_backup: bool = True

class ThresholdDataStorage:
    """
    阈值数据存储和管理系统
    
    支持多种存储格式：SQLite数据库、JSON文件、Pickle文件
    提供数据查询、统计和管理功能
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化存储系统
        
        Args:
            config: 存储配置
        """
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据库连接
        self.db_path = self.storage_dir / "thresholds.db"
        self.conn = None
        
        if config.use_database:
            self._init_database()
        
        logger.info(f"数据存储系统初始化完成: {self.storage_dir}")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # 启用字典式访问
        
        # 创建表结构
        cursor = self.conn.cursor()
        
        # 阈值记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threshold_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                layer_id INTEGER NOT NULL,
                head_id INTEGER,
                quantile_name TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                collection_timestamp TEXT NOT NULL,
                model_config TEXT NOT NULL,
                bootstrap_iteration INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 数据集元信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                total_samples INTEGER,
                avg_sequence_length REAL,
                vocabulary_diversity REAL,
                task_complexity_score REAL,
                domain_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 收集会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                model_path TEXT NOT NULL,
                collection_config TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_records INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_layer ON threshold_records(dataset_name, layer_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quantile ON threshold_records(quantile_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sample_size ON threshold_records(sample_size)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bootstrap ON threshold_records(bootstrap_iteration)')
        
        self.conn.commit()
        logger.info("数据库初始化完成")
    
    def save_threshold_records(self, records: List[ThresholdRecord], session_id: str = None) -> str:
        """
        保存阈值记录
        
        Args:
            records: 阈值记录列表
            session_id: 会话ID（可选）
            
        Returns:
            session_id: 会话ID
        """
        if not records:
            logger.warning("没有记录需要保存")
            return session_id
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"保存 {len(records)} 条阈值记录到会话: {session_id}")
        
        # 保存到数据库
        if self.config.use_database and self.conn:
            self._save_to_database(records, session_id)
        
        # 保存到JSON文件
        if self.config.use_json_backup:
            self._save_to_json(records, session_id)
        
        # 保存到Pickle文件
        if self.config.use_pickle_backup:
            self._save_to_pickle(records, session_id)
        
        logger.info(f"阈值记录保存完成: {session_id}")
        return session_id
    
    def _save_to_database(self, records: List[ThresholdRecord], session_id: str):
        """保存到SQLite数据库"""
        cursor = self.conn.cursor()
        
        # 创建会话记录
        cursor.execute('''
            INSERT OR REPLACE INTO collection_sessions 
            (session_id, model_path, collection_config, start_time, total_records, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            records[0].model_config.get('model_path', ''),
            json.dumps(records[0].model_config),
            records[0].collection_timestamp.isoformat(),
            len(records),
            'completed'
        ))
        
        # 插入阈值记录
        for record in records:
            cursor.execute('''
                INSERT INTO threshold_records 
                (dataset_name, sample_size, layer_id, head_id, quantile_name, 
                 threshold_value, collection_timestamp, model_config, bootstrap_iteration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.dataset_name,
                record.sample_size,
                record.layer_id,
                record.head_id,
                record.quantile_name,
                record.threshold_value,
                record.collection_timestamp.isoformat(),
                json.dumps(record.model_config),
                record.bootstrap_iteration
            ))
        
        self.conn.commit()
        logger.info(f"已保存 {len(records)} 条记录到数据库")
    
    def _save_to_json(self, records: List[ThresholdRecord], session_id: str):
        """保存到JSON文件"""
        json_file = self.storage_dir / f"{session_id}.json"
        
        # 转换记录为可序列化格式
        serializable_records = []
        for record in records:
            record_dict = asdict(record)
            record_dict['collection_timestamp'] = record.collection_timestamp.isoformat()
            serializable_records.append(record_dict)
        
        data = {
            'session_id': session_id,
            'total_records': len(records),
            'created_at': datetime.now().isoformat(),
            'records': serializable_records
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存到JSON文件: {json_file}")
    
    def _save_to_pickle(self, records: List[ThresholdRecord], session_id: str):
        """保存到Pickle文件"""
        pickle_file = self.storage_dir / f"{session_id}.pkl"
        
        data = {
            'session_id': session_id,
            'total_records': len(records),
            'created_at': datetime.now(),
            'records': records
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"已保存到Pickle文件: {pickle_file}")
    
    def load_threshold_records(self, 
                             session_id: str = None,
                             dataset_name: str = None,
                             layer_id: int = None,
                             quantile_name: str = None) -> List[ThresholdRecord]:
        """
        加载阈值记录
        
        Args:
            session_id: 会话ID（可选）
            dataset_name: 数据集名称（可选）
            layer_id: 层ID（可选）
            quantile_name: 分位点名称（可选）
            
        Returns:
            records: 阈值记录列表
        """
        if self.config.use_database and self.conn:
            return self._load_from_database(session_id, dataset_name, layer_id, quantile_name)
        elif session_id:
            # 尝试从文件加载
            return self._load_from_files(session_id)
        else:
            logger.error("无法加载记录：需要数据库或会话ID")
            return []
    
    def _load_from_database(self, 
                          session_id: str = None,
                          dataset_name: str = None,
                          layer_id: int = None,
                          quantile_name: str = None) -> List[ThresholdRecord]:
        """从数据库加载记录"""
        cursor = self.conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if session_id:
            # 通过会话ID查找
            cursor.execute('SELECT model_path FROM collection_sessions WHERE session_id = ?', (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                logger.error(f"会话不存在: {session_id}")
                return []
        
        if dataset_name:
            conditions.append("dataset_name = ?")
            params.append(dataset_name)
        
        if layer_id is not None:
            conditions.append("layer_id = ?")
            params.append(layer_id)
        
        if quantile_name:
            conditions.append("quantile_name = ?")
            params.append(quantile_name)
        
        # 构建SQL查询
        query = "SELECT * FROM threshold_records"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY dataset_name, layer_id, quantile_name"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 转换为ThresholdRecord对象
        records = []
        for row in rows:
            record = ThresholdRecord(
                dataset_name=row['dataset_name'],
                sample_size=row['sample_size'],
                layer_id=row['layer_id'],
                head_id=row['head_id'],
                quantile_name=row['quantile_name'],
                threshold_value=row['threshold_value'],
                collection_timestamp=datetime.fromisoformat(row['collection_timestamp']),
                model_config=json.loads(row['model_config']),
                bootstrap_iteration=row['bootstrap_iteration']
            )
            records.append(record)
        
        logger.info(f"从数据库加载了 {len(records)} 条记录")
        return records
    
    def _load_from_files(self, session_id: str) -> List[ThresholdRecord]:
        """从文件加载记录"""
        # 优先尝试Pickle文件
        pickle_file = self.storage_dir / f"{session_id}.pkl"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"从Pickle文件加载了 {len(data['records'])} 条记录")
            return data['records']
        
        # 尝试JSON文件
        json_file = self.storage_dir / f"{session_id}.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为ThresholdRecord对象
            records = []
            for record_dict in data['records']:
                record_dict['collection_timestamp'] = datetime.fromisoformat(
                    record_dict['collection_timestamp']
                )
                record = ThresholdRecord(**record_dict)
                records.append(record)
            
            logger.info(f"从JSON文件加载了 {len(records)} 条记录")
            return records
        
        logger.error(f"找不到会话文件: {session_id}")
        return []
    
    def get_dataset_statistics(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            dataset_name: 数据集名称（可选，为None时返回所有数据集）
            
        Returns:
            statistics: 统计信息字典
        """
        if not self.config.use_database or not self.conn:
            logger.error("需要数据库支持才能获取统计信息")
            return {}
        
        cursor = self.conn.cursor()
        
        if dataset_name:
            # 单个数据集统计
            cursor.execute('''
                SELECT 
                    dataset_name,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT layer_id) as num_layers,
                    COUNT(DISTINCT quantile_name) as num_quantiles,
                    COUNT(DISTINCT sample_size) as num_sample_sizes,
                    MIN(threshold_value) as min_threshold,
                    MAX(threshold_value) as max_threshold,
                    AVG(threshold_value) as avg_threshold
                FROM threshold_records 
                WHERE dataset_name = ?
                GROUP BY dataset_name
            ''', (dataset_name,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            else:
                return {}
        else:
            # 所有数据集统计
            cursor.execute('''
                SELECT 
                    dataset_name,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT layer_id) as num_layers,
                    COUNT(DISTINCT quantile_name) as num_quantiles,
                    COUNT(DISTINCT sample_size) as num_sample_sizes,
                    MIN(threshold_value) as min_threshold,
                    MAX(threshold_value) as max_threshold,
                    AVG(threshold_value) as avg_threshold
                FROM threshold_records 
                GROUP BY dataset_name
            ''')
            
            rows = cursor.fetchall()
            statistics = {}
            for row in rows:
                statistics[row['dataset_name']] = dict(row)
            
            return statistics
    
    def get_threshold_matrix(self, 
                           dataset_name: str,
                           quantile_name: str,
                           sample_size: int = None) -> np.ndarray:
        """
        获取阈值矩阵
        
        Args:
            dataset_name: 数据集名称
            quantile_name: 分位点名称
            sample_size: 样本大小（可选）
            
        Returns:
            threshold_matrix: 阈值矩阵 [layers x samples/bootstrap_iterations]
        """
        if not self.config.use_database or not self.conn:
            logger.error("需要数据库支持才能获取阈值矩阵")
            return np.array([])
        
        cursor = self.conn.cursor()
        
        # 构建查询条件
        conditions = ["dataset_name = ?", "quantile_name = ?"]
        params = [dataset_name, quantile_name]
        
        if sample_size is not None:
            conditions.append("sample_size = ?")
            params.append(sample_size)
        
        query = f'''
            SELECT layer_id, bootstrap_iteration, threshold_value
            FROM threshold_records 
            WHERE {" AND ".join(conditions)}
            ORDER BY layer_id, bootstrap_iteration
        '''
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"没有找到匹配的阈值数据: {dataset_name}, {quantile_name}")
            return np.array([])
        
        # 转换为DataFrame以便处理
        df = pd.DataFrame(rows)
        
        # 创建矩阵
        layers = sorted(df['layer_id'].unique())
        iterations = sorted(df['bootstrap_iteration'].dropna().unique()) if df['bootstrap_iteration'].notna().any() else [None]
        
        if iterations == [None]:
            # 没有Bootstrap数据，每层只有一个值
            matrix = np.zeros((len(layers), 1))
            for i, layer_id in enumerate(layers):
                layer_data = df[df['layer_id'] == layer_id]
                if not layer_data.empty:
                    matrix[i, 0] = layer_data['threshold_value'].iloc[0]
        else:
            # 有Bootstrap数据
            matrix = np.zeros((len(layers), len(iterations)))
            for i, layer_id in enumerate(layers):
                for j, iteration in enumerate(iterations):
                    layer_iter_data = df[(df['layer_id'] == layer_id) & 
                                       (df['bootstrap_iteration'] == iteration)]
                    if not layer_iter_data.empty:
                        matrix[i, j] = layer_iter_data['threshold_value'].iloc[0]
        
        logger.info(f"获取阈值矩阵: {matrix.shape} ({dataset_name}, {quantile_name})")
        return matrix
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有收集会话
        
        Returns:
            sessions: 会话信息列表
        """
        if not self.config.use_database or not self.conn:
            # 从文件系统列出
            sessions = []
            for file_path in self.storage_dir.glob("session_*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    sessions.append({
                        'session_id': data['session_id'],
                        'total_records': data['total_records'],
                        'created_at': data['created_at'],
                        'source': 'file'
                    })
                except Exception as e:
                    logger.warning(f"无法读取会话文件 {file_path}: {e}")
            return sessions
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT session_id, model_path, start_time, end_time, total_records, status, created_at
            FROM collection_sessions 
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        sessions = [dict(row) for row in rows]
        
        logger.info(f"找到 {len(sessions)} 个收集会话")
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其相关数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            success: 是否成功删除
        """
        success = True
        
        # 从数据库删除
        if self.config.use_database and self.conn:
            cursor = self.conn.cursor()
            
            # 删除阈值记录（通过时间戳关联）
            cursor.execute('''
                DELETE FROM threshold_records 
                WHERE collection_timestamp IN (
                    SELECT start_time FROM collection_sessions WHERE session_id = ?
                )
            ''', (session_id,))
            
            # 删除会话记录
            cursor.execute('DELETE FROM collection_sessions WHERE session_id = ?', (session_id,))
            
            self.conn.commit()
            logger.info(f"从数据库删除会话: {session_id}")
        
        # 删除文件
        for ext in ['.json', '.pkl']:
            file_path = self.storage_dir / f"{session_id}{ext}"
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"删除文件: {file_path}")
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {e}")
                    success = False
        
        return success
    
    def export_to_csv(self, output_file: str, dataset_name: str = None) -> bool:
        """
        导出数据到CSV文件
        
        Args:
            output_file: 输出文件路径
            dataset_name: 数据集名称（可选）
            
        Returns:
            success: 是否成功导出
        """
        try:
            records = self.load_threshold_records(dataset_name=dataset_name)
            
            if not records:
                logger.warning("没有数据可导出")
                return False
            
            # 转换为DataFrame
            data = []
            for record in records:
                data.append({
                    'dataset_name': record.dataset_name,
                    'sample_size': record.sample_size,
                    'layer_id': record.layer_id,
                    'head_id': record.head_id,
                    'quantile_name': record.quantile_name,
                    'threshold_value': record.threshold_value,
                    'collection_timestamp': record.collection_timestamp.isoformat(),
                    'bootstrap_iteration': record.bootstrap_iteration,
                    'model_type': record.model_config.get('model_type', ''),
                    'num_layers': record.model_config.get('num_layers', ''),
                    'num_heads': record.model_config.get('num_heads', '')
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"数据已导出到: {output_file} ({len(records)} 条记录)")
            return True
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return False
    
    def close(self):
        """关闭存储系统"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()