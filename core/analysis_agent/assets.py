"""Local scoped assets. SQLite transactions bind metadata and Parquet/PNG payloads.

Scope identities must come from a trusted controller, not model arguments.
"""
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from threading import RLock

import pandas as pd
from utils.analysis_datasets import DatasetStore, DatasetInfo, Condition
from utils.analysis_charts import ChartPreview


class AssetDB:
    def __init__(self, root, owner, conversation, max_scope_bytes=None):
        if not owner or not conversation:
            raise ValueError('Trusted owner and conversation are required')
        key=sha256(json.dumps([owner,conversation]).encode()).hexdigest()
        self.directory=Path(root)/key
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.max_scope_bytes=max_scope_bytes
        self.lock=RLock()
        self.conn=sqlite3.connect(self.directory/'assets.sqlite',check_same_thread=False)
        self.conn.execute('CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, kind TEXT, metadata TEXT, payload BLOB)')
        self.conn.commit()

    def put(self,key,kind,metadata,payload):
        if self.max_scope_bytes is not None:
            current=sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
            if current+len(payload)>self.max_scope_bytes:
                raise MemoryError('이 대화의 저장 공간 한도를 초과합니다. 기존 결과를 정리하거나 더 작은 데이터 범위를 요청해주세요.')
        with self.lock,self.conn:
            self.conn.execute('INSERT INTO assets VALUES (?,?,?,?)',
                (key,kind,json.dumps(metadata,ensure_ascii=False),payload))

    def get(self,key,kind):
        with self.lock:
            row=self.conn.execute('SELECT metadata,payload FROM assets WHERE id=? AND kind=?',(key,kind)).fetchone()
        if row is None: raise KeyError('로컬 자산이 없거나 만료되었습니다. 자동 재조회하지 않습니다.')
        return json.loads(row[0]),row[1]

    def require(self,key,kind):
        """Check an asset reference without reading a potentially large payload."""
        with self.lock:
            row=self.conn.execute('SELECT 1 FROM assets WHERE id=? AND kind=?',(key,kind)).fetchone()
        if row is None: raise KeyError('로컬 자산이 없거나 만료되었습니다. 자동 재조회하지 않습니다.')

    def metadata(self,kind):
        with self.lock:
            return {key:json.loads(meta) for key,meta in self.conn.execute(
                'SELECT id,metadata FROM assets WHERE kind=?',(kind,)).fetchall()}

    def close(self):self.conn.close()


class FrameCache(Mapping):
    def __init__(self,db,budget):
        if budget<0:raise ValueError('Cache budget must be nonnegative')
        self.db,self.budget=db,budget
        self.cache=OrderedDict()
        self.bytes=0
        self.lock=RLock()

    def __iter__(self):return iter(self.db.metadata('dataset'))
    def __len__(self):return len(self.db.metadata('dataset'))
    def __getitem__(self,key):
        with self.lock:
            if key in self.cache:
                frame,size=self.cache.pop(key);self.cache[key]=(frame,size)
                return frame.copy(deep=True)
            _,payload=self.db.get(key,'dataset')
            frame=pd.read_parquet(BytesIO(payload))
            size=int(frame.memory_usage(index=True,deep=True).sum())
            while self.cache and self.bytes+size>self.budget:
                _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
            if size<=self.budget:
                self.cache[key]=(frame,size);self.bytes+=size
                return frame.copy(deep=True)
            # This newly decoded frame is not retained by the store. Returning
            # it directly preserves isolation without a second full allocation.
            return frame


class PersistentDatasets(DatasetStore):
    def __init__(self,db,budget=64*1024*1024,max_columns=None,max_frame_bytes=None):
        self.db=db
        self.frames=FrameCache(db,budget)
        self.max_columns=max_columns
        self.max_frame_bytes=max_frame_bytes

    @property
    def metadata(self):
        result={}
        for key,value in self.db.metadata('dataset').items():
            value['columns']=tuple(value['columns'])
            value['conditions']=tuple(Condition(**c) for c in value['conditions'])
            result[key]=DatasetInfo(**value)
        return result

    def register(self,frame,*,source,**provenance):
        if self.max_columns is not None and len(frame.columns)>self.max_columns:
            raise ValueError(f'데이터 컬럼 수가 운영 한도({self.max_columns})를 초과합니다.')
        frame_bytes=int(frame.memory_usage(index=True,deep=True).sum())
        if self.max_frame_bytes is not None and frame_bytes>self.max_frame_bytes:
            raise MemoryError(f'데이터 메모리 크기가 운영 한도({self.max_frame_bytes} bytes)를 초과합니다.')
        info=DatasetStore().register(frame,source=source,**provenance)
        buffer=BytesIO()
        frame.to_parquet(buffer,index=True)
        self.db.put(info.id,'dataset',asdict(info),buffer.getvalue())
        return info


class PersistentCharts(MutableMapping):
    def __init__(self,db):self.db=db
    def __iter__(self):return iter(self.db.metadata('chart'))
    def __len__(self):return len(self.db.metadata('chart'))
    def __getitem__(self,key):
        value,payload=self.db.get(key,'chart')
        value['columns']=tuple(value['columns'])
        return ChartPreview(**value,image=payload)
    def __setitem__(self,key,card):
        if key!=card.id:raise ValueError('Chart ID mismatch')
        self.db.require(card.dataset_id,'dataset')
        value=asdict(card);value.pop('image')
        self.db.put(key,'chart',value,card.image)
    def __delitem__(self,key):raise TypeError('Immutable chart assets')
