import pandas as pd
from datetime import datetime
import pytz
import streamlit as st

MAX_CACHE_ROWS = 500

class CacheManager:
    def __init__(self):
        self.cache = {}
        self.pending_writes = {}
        self._initialized = True

    @staticmethod
    def get_instance():
        if '_cache_manager' not in st.session_state:
            st.session_state._cache_manager = CacheManager()
        return st.session_state._cache_manager

    def update(self, table_name, records):
        if not records:
            return
        if isinstance(records, list):
            df = pd.DataFrame(records)
        elif isinstance(records, pd.DataFrame):
            df = records
        else:
            return
        if table_name in self.cache:
            self.cache[table_name] = pd.concat([self.cache[table_name], df]).drop_duplicates().tail(MAX_CACHE_ROWS)
        else:
            self.cache[table_name] = df.tail(MAX_CACHE_ROWS)

    def get(self, table_name, filters=None):
        df = self.cache.get(table_name, pd.DataFrame())
        if df.empty or not filters:
            return df
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]
        return df

    def queue_write(self, table_name, records, conflict_cols=None):
        if table_name not in self.pending_writes:
            self.pending_writes[table_name] = []
        self.pending_writes[table_name].append({
            'records': records,
            'conflict_cols': conflict_cols,
            'queued_at': datetime.now(pytz.UTC)
        })
        self.update(table_name, records)

    def get_pending(self, table_name=None):
        if table_name:
            return self.pending_writes.get(table_name, [])
        return self.pending_writes

    def clear_pending(self, table_name):
        if table_name in self.pending_writes:
            del self.pending_writes[table_name]

    def has_pending(self):
        return any(len(v) > 0 for v in self.pending_writes.values())
