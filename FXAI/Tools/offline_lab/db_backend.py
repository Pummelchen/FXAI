from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import libsql


TURSO_DATABASE_URL_ENV = "TURSO_DATABASE_URL"
TURSO_AUTH_TOKEN_ENV = "TURSO_AUTH_TOKEN"

_CONNECTION_CONFIGS: dict[int, "TursoConfig"] = {}


@dataclass(frozen=True)
class TursoConfig:
    database: Path
    sync_url: str = ""
    auth_token: str = ""

    @property
    def sync_enabled(self) -> bool:
        return bool(self.sync_url and self.auth_token)

    @property
    def partial_sync_config(self) -> bool:
        return bool(self.sync_url) != bool(self.auth_token)

    @property
    def sync_mode(self) -> str:
        return "embedded_replica" if self.sync_enabled else "local_only"

    @property
    def backend_name(self) -> str:
        return "turso_embedded_replica" if self.sync_enabled else "turso_local"

    def validate(self) -> None:
        if self.partial_sync_config:
            raise ValueError(
                "partial Turso configuration: set both "
                f"{TURSO_DATABASE_URL_ENV} and {TURSO_AUTH_TOKEN_ENV}, or neither"
            )


def connection_config(connection: libsql.Connection) -> TursoConfig | None:
    return _CONNECTION_CONFIGS.get(id(connection))


def register_connection_config(connection: libsql.Connection, config: TursoConfig) -> None:
    _CONNECTION_CONFIGS[id(connection)] = config


def sync_backend(connection: libsql.Connection) -> None:
    config = connection_config(connection)
    if config is not None and config.sync_enabled:
        connection.sync()


def commit_backend(connection: libsql.Connection) -> None:
    connection.commit()
    sync_backend(connection)


def close_backend(connection: libsql.Connection) -> None:
    try:
        connection.close()
    finally:
        _CONNECTION_CONFIGS.pop(id(connection), None)


def connect_backend(config: TursoConfig, timeout: float = 30.0) -> libsql.Connection:
    config.validate()
    kwargs = {"timeout": float(timeout)}
    if config.sync_enabled:
        kwargs["sync_url"] = str(config.sync_url)
        kwargs["auth_token"] = str(config.auth_token)
    connection = libsql.connect(str(config.database), **kwargs)
    register_connection_config(connection, config)
    return connection
