from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import libsql


TURSO_DATABASE_URL_ENV = "TURSO_DATABASE_URL"
TURSO_AUTH_TOKEN_ENV = "TURSO_AUTH_TOKEN"
TURSO_ENCRYPTION_KEY_ENV = "TURSO_ENCRYPTION_KEY"
TURSO_SYNC_INTERVAL_ENV = "TURSO_SYNC_INTERVAL_SECONDS"
TURSO_DATABASE_NAME_ENV = "TURSO_DATABASE_NAME"
TURSO_ORGANIZATION_ENV = "TURSO_ORGANIZATION"
TURSO_API_TOKEN_ENV = "TURSO_API_TOKEN"
TURSO_GROUP_ENV = "TURSO_GROUP"
TURSO_LOCATION_ENV = "TURSO_LOCATION"
TURSO_CONFIG_PATH_ENV = "TURSO_CONFIG_PATH"

_CONNECTION_CONFIGS: dict[int, "TursoConfig"] = {}


@dataclass(frozen=True)
class TursoConfig:
    database: Path
    sync_url: str = ""
    auth_token: str = ""
    encryption_key: str = ""
    sync_interval_seconds: float = 0.0
    database_name: str = ""
    organization_slug: str = ""
    api_token: str = ""
    group_name: str = ""
    location_name: str = ""
    cli_config_path: str = ""

    @property
    def sync_enabled(self) -> bool:
        return bool(self.sync_url and self.auth_token)

    @property
    def partial_sync_config(self) -> bool:
        return bool(self.sync_url) != bool(self.auth_token)

    @property
    def partial_platform_api_config(self) -> bool:
        return bool(self.organization_slug) != bool(self.api_token)

    @property
    def sync_mode(self) -> str:
        return "embedded_replica" if self.sync_enabled else "local_only"

    @property
    def backend_name(self) -> str:
        return "turso_embedded_replica" if self.sync_enabled else "turso_local"

    @property
    def encryption_enabled(self) -> bool:
        return bool(self.encryption_key)

    @property
    def platform_api_enabled(self) -> bool:
        return bool(self.organization_slug and self.api_token)

    @property
    def cli_configured(self) -> bool:
        return bool(self.database_name or self.organization_slug or self.group_name or self.location_name)

    def validate(self) -> None:
        if self.partial_sync_config:
            raise ValueError(
                "partial Turso configuration: set both "
                f"{TURSO_DATABASE_URL_ENV} and {TURSO_AUTH_TOKEN_ENV}, or neither"
            )
        if self.partial_platform_api_config:
            raise ValueError(
                "partial Turso platform configuration: set both "
                f"{TURSO_ORGANIZATION_ENV} and {TURSO_API_TOKEN_ENV}, or neither"
            )
        if self.sync_interval_seconds < 0.0:
            raise ValueError(
                f"{TURSO_SYNC_INTERVAL_ENV} must be zero or positive"
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
        if config.sync_interval_seconds > 0.0:
            kwargs["sync_interval"] = float(config.sync_interval_seconds)
    if config.encryption_enabled:
        kwargs["encryption_key"] = str(config.encryption_key)
    connection = libsql.connect(str(config.database), **kwargs)
    register_connection_config(connection, config)
    return connection
