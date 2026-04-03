from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import libsql


TURSO_DATABASE_URL_ENV = "TURSO_DATABASE_URL"
TURSO_AUTH_TOKEN_ENV = "TURSO_AUTH_TOKEN"


@dataclass(frozen=True)
class TursoConfig:
    database: Path
    sync_url: str = ""
    auth_token: str = ""

    @property
    def sync_enabled(self) -> bool:
        return bool(self.sync_url and self.auth_token)

    @property
    def backend_name(self) -> str:
        return "turso_embedded_replica" if self.sync_enabled else "turso_local_libsql"


class LabRow(Mapping[str, object]):
    def __init__(self, column_names: Sequence[str], values: Sequence[object]):
        keys = [str(name or "") for name in column_names]
        vals = tuple(values)
        self._keys = tuple(keys)
        self._values = vals
        self._mapping = {keys[idx]: vals[idx] for idx in range(min(len(keys), len(vals)))}

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._values[key]
        return self._mapping[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def get(self, key: str, default=None):
        return self._mapping.get(str(key), default)

    def as_dict(self) -> dict[str, object]:
        return dict(self._mapping)

    def as_tuple(self) -> tuple[object, ...]:
        return tuple(self._values)


class LabCursor:
    def __init__(self, raw_cursor):
        self._raw = raw_cursor

    def _column_names(self) -> list[str]:
        description = getattr(self._raw, "description", None) or []
        names: list[str] = []
        for item in description:
            try:
                names.append(str(item[0]))
            except Exception:
                names.append("")
        return names

    def _wrap_row(self, row):
        if row is None or isinstance(row, LabRow):
            return row
        if isinstance(row, tuple):
            return LabRow(self._column_names(), row)
        return row

    def fetchone(self):
        return self._wrap_row(self._raw.fetchone())

    def fetchall(self):
        return [self._wrap_row(row) for row in self._raw.fetchall()]

    def fetchmany(self, size: int | None = None):
        if size is None:
            rows = self._raw.fetchmany()
        else:
            rows = self._raw.fetchmany(size)
        return [self._wrap_row(row) for row in rows]

    def execute(self, sql: str, parameters: Iterable[object] | None = None):
        if parameters is None:
            self._raw.execute(sql)
        else:
            self._raw.execute(sql, tuple(parameters))
        return self

    def executemany(self, sql: str, seq_of_parameters: Iterable[Iterable[object]]):
        self._raw.executemany(sql, [tuple(item) for item in seq_of_parameters])
        return self

    @property
    def description(self):
        return getattr(self._raw, "description", None)

    @property
    def rowcount(self):
        return getattr(self._raw, "rowcount", -1)

    @property
    def lastrowid(self):
        return getattr(self._raw, "lastrowid", None)

    def close(self) -> None:
        close_fn = getattr(self._raw, "close", None)
        if callable(close_fn):
            close_fn()

    def __iter__(self) -> Iterator[LabRow]:
        while True:
            row = self.fetchone()
            if row is None:
                break
            yield row


class LabConnection:
    def __init__(self, raw_connection, config: TursoConfig):
        self._raw = raw_connection
        self._config = config

    @property
    def backend_name(self) -> str:
        return self._config.backend_name

    @property
    def sync_enabled(self) -> bool:
        return self._config.sync_enabled

    @property
    def database_path(self) -> Path:
        return self._config.database

    def execute(self, sql: str, parameters: Iterable[object] | None = None) -> LabCursor:
        if parameters is None:
            cursor = self._raw.execute(sql)
        else:
            cursor = self._raw.execute(sql, tuple(parameters))
        return LabCursor(cursor)

    def executemany(self, sql: str, seq_of_parameters: Iterable[Iterable[object]]) -> LabCursor:
        cursor = self._raw.executemany(sql, [tuple(item) for item in seq_of_parameters])
        return LabCursor(cursor)

    def executescript(self, script: str) -> LabCursor:
        cursor = self._raw.executescript(script)
        return LabCursor(cursor)

    def cursor(self) -> LabCursor:
        return LabCursor(self._raw.cursor())

    def commit(self) -> None:
        self._raw.commit()
        if self.sync_enabled:
            self._raw.sync()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()

    def sync(self) -> None:
        if self.sync_enabled:
            self._raw.sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        finally:
            self.close()
        return False

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


def connect_backend(config: TursoConfig, timeout: float = 30.0) -> LabConnection:
    kwargs = {"timeout": float(timeout)}
    if config.sync_enabled:
        kwargs["sync_url"] = str(config.sync_url)
        kwargs["auth_token"] = str(config.auth_token)
    raw = libsql.connect(str(config.database), **kwargs)
    return LabConnection(raw, config)
