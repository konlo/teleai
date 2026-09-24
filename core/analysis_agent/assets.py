"""Local scoped assets. SQLite binds metadata, legacy BLOBs and chart payloads.

New remote datasets are staged as Parquet files, then published after validation.

Scope identities must come from a trusted controller, not model arguments.
"""
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
        self.conn.execute('CREATE TABLE IF NOT EXISTS dataset_previews (id TEXT PRIMARY KEY, rows_json TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS selection (slot TEXT PRIMARY KEY, dataset_id TEXT NOT NULL)')
        self.conn.commit()

    def put(self,key,kind,metadata,payload,*,preview=None):
        if self.max_scope_bytes is not None:
            current=sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
            if current+len(payload)>self.max_scope_bytes:
                raise MemoryError('이 대화의 저장 공간 한도를 초과합니다. 기존 결과를 정리하거나 더 작은 데이터 범위를 요청해주세요.')
        with self.lock,self.conn:
            self.conn.execute('INSERT INTO assets VALUES (?,?,?,?)',
                (key,kind,json.dumps(metadata,ensure_ascii=False),payload))
            if preview is not None:
                self.conn.execute('INSERT INTO dataset_previews VALUES (?,?)',
                    (key,json.dumps(preview,ensure_ascii=False)))

    def put_staged_dataset(self, key, metadata, staged_path, *, preview):
        """Publish a verified Parquet candidate without loading it as a BLOB."""
        destination = self.directory / f'{key}.parquet'
        if staged_path.parent != self.directory or destination.exists():
            raise ValueError('저장 후보 경로가 유효하지 않습니다.')
        with self.lock:
            current = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
            if self.max_scope_bytes is not None and current > self.max_scope_bytes:
                raise MemoryError('이 대화의 저장 공간 한도를 초과합니다. 기존 결과를 보존했습니다.')
            os.replace(staged_path, destination)
            try:
                with self.conn:
                    self.conn.execute('INSERT INTO assets VALUES (?,?,?,NULL)',
                        (key, 'dataset', json.dumps(metadata, ensure_ascii=False)))
                    self.conn.execute('INSERT INTO dataset_previews VALUES (?,?)',
                        (key, json.dumps(preview, ensure_ascii=False)))
            except BaseException:
                destination.unlink(missing_ok=True)
                raise

    def dataset_file(self, key):
        with self.lock:
            row = self.conn.execute('SELECT payload FROM assets WHERE id=? AND kind=?',
                                    (key, 'dataset')).fetchone()
        if row is None:
            raise KeyError('로컬 자산이 없거나 만료되었습니다. 자동 재조회하지 않습니다.')
        if row[0] is not None:
            return None  # Legacy SQLite BLOB.
        path = self.directory / f'{key}.parquet'
        if not path.is_file():
            raise FileNotFoundError('저장된 데이터 파일이 없습니다. 자동 재조회하지 않습니다.')
        return path

    def dataset_preview(self,key):
        """Read the small display sample without decoding the stored frame."""
        with self.lock:
            row=self.conn.execute('SELECT rows_json FROM dataset_previews WHERE id=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

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

    def selected_dataset_id(self):
        with self.lock:
            row=self.conn.execute("SELECT dataset_id FROM selection WHERE slot='active'").fetchone()
        return row[0] if row else ''

    def select_dataset(self,dataset_id):
        with self.lock,self.conn:
            present=self.conn.execute('SELECT 1 FROM assets WHERE id=? AND kind=?',
                                      (dataset_id,'dataset')).fetchone()
            if not present:
                raise KeyError('선택할 로컬 데이터가 없습니다. 자동 재조회하지 않습니다.')
            self.conn.execute("INSERT OR REPLACE INTO selection VALUES ('active',?)",(dataset_id,))

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
            path=self.db.dataset_file(key)
            if path is None:
                _,payload=self.db.get(key,'dataset')
                frame=pd.read_parquet(BytesIO(payload))
            else:
                frame=pd.read_parquet(path)
            size=int(frame.memory_usage(index=True,deep=True).sum())
            while self.cache and self.bytes+size>self.budget:
                _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
            if size<=self.budget:
                self.cache[key]=(frame,size);self.bytes+=size
                return frame.copy(deep=True)
            # This newly decoded frame is not retained by the store. Returning
            # it directly preserves isolation without a second full allocation.
            return frame

    def project(self, key, columns):
        """Read only requested Parquet columns; leave the full-frame cache cold."""
        path = self.db.dataset_file(key)
        if path is None:
            _, payload = self.db.get(key, 'dataset')
            return pd.read_parquet(BytesIO(payload), columns=list(columns))
        return pd.read_parquet(path, columns=list(columns))


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
            value['parent_ids']=tuple(value.get('parent_ids', ()))
            result[key]=DatasetInfo(**value)
        return result

    def register(self,frame,*,source,**provenance):
        if self.max_columns is not None and len(frame.columns)>self.max_columns:
            raise ValueError(f'데이터 컬럼 수가 운영 한도({self.max_columns})를 초과합니다.')
        frame_bytes=int(frame.memory_usage(index=True,deep=True).sum())
        if self.max_frame_bytes is not None and frame_bytes>self.max_frame_bytes:
            raise MemoryError(f'데이터 메모리 크기가 운영 한도({self.max_frame_bytes} bytes)를 초과합니다.')
        info=DatasetStore(metadata=self.metadata).register(frame,source=source,**provenance)
        buffer=BytesIO()
        frame.to_parquet(buffer,index=True)
        sample=frame.head(5)
        def display(value):
            try:
                if bool(pd.isna(value)):
                    return None
            except (TypeError,ValueError):
                pass
            return str(value)[:256]
        preview=[{str(column):display(value)
                  for column,value in row.items()} for row in sample.to_dict(orient='records')]
        self.db.put(info.id,'dataset',asdict(info),buffer.getvalue(),preview=preview)
        return info

    def inspect(self, dataset_id):
        info = self.metadata[dataset_id]
        path = self.db.dataset_file(dataset_id)
        if path is None:
            frame = self.frames[dataset_id]
            dtypes = frame.dtypes.astype(str).to_dict()
        else:
            schema = pq.ParquetFile(path).schema_arrow
            dtypes = {}
            for field in schema:
                if field.name in info.columns:
                    try:
                        dtypes[field.name] = str(pd.Series(dtype=field.type.to_pandas_dtype()).dtype)
                    except (TypeError, NotImplementedError):
                        dtypes[field.name] = str(field.type)
        return {'dataset': asdict(info), 'dtypes': dtypes,
                'preview': self.db.dataset_preview(dataset_id) or []}

    def register_batches(self, batches, *, columns, source, max_rows,
                         final_provenance=None, **provenance):
        """Stage bounded remote batches on disk and publish only after full validation.

        The first batch determines Arrow types. A later incompatible type fails
        closed, leaving every previously published dataset and selection intact.
        """
        if self.max_columns is not None and len(columns) > self.max_columns:
            raise ValueError('데이터 컬럼 수가 운영 한도를 초과합니다.')
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('데이터 컬럼 이름이 유효하지 않습니다.')
        staged = self.db.directory / f'.{uuid4().hex}.staging.parquet'
        writer = None
        count = 0
        estimated_bytes = 0
        preview = []
        try:
            for batch in batches:
                if tuple(batch.columns) != tuple(columns):
                    raise ValueError('조회 결과 컬럼 구조가 일치하지 않습니다.')
                if count + len(batch) > max_rows:
                    raise ValueError('조회 후보가 행 한도를 초과했습니다.')
                estimated_bytes += int(batch.memory_usage(index=True, deep=True).sum())
                if self.max_frame_bytes is not None and estimated_bytes > self.max_frame_bytes:
                    raise MemoryError('조회 결과가 데이터 메모리 한도를 초과해 발행하지 않았습니다.')
                if len(preview) < 5:
                    preview.extend(self._preview_rows(batch.head(5 - len(preview))))
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(staged, table.schema)
                else:
                    table = table.cast(writer.schema, safe=True)
                writer.write_table(table)
                count += len(batch)
                if self.db.max_scope_bytes is not None:
                    used = sum(path.stat().st_size for path in self.db.directory.iterdir()
                               if path.is_file())
                    if used > self.db.max_scope_bytes:
                        raise MemoryError('이 대화의 저장 공간 한도를 초과합니다. 기존 결과를 보존했습니다.')
            if writer is None:
                writer = pq.ParquetWriter(staged, pa.Table.from_pandas(
                    pd.DataFrame(columns=columns), preserve_index=False).schema)
            writer.close()
            writer = None
            with staged.open('rb') as candidate_file:
                os.fsync(candidate_file.fileno())
            if final_provenance is not None:
                provenance.update(final_provenance())
            base = DatasetStore(metadata=self.metadata).register(
                pd.DataFrame(columns=columns), source=source, **provenance)
            info = replace(base, rows=count)
            self.db.put_staged_dataset(info.id, asdict(info), staged, preview=preview)
            return info
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                staged.unlink(missing_ok=True)

    @staticmethod
    def _preview_rows(frame):
        def display(value):
            try:
                if bool(pd.isna(value)):
                    return None
            except (TypeError, ValueError):
                pass
            return str(value)[:256]
        return [{str(column): display(value) for column, value in row.items()}
                for row in frame.to_dict(orient='records')]


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
