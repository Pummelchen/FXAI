from __future__ import annotations

from . import cli_campaigns as _cli_campaigns
from . import cli_commands as _cli_commands
from .cli_campaigns import *
from .cli_commands import *
from .cli_parser import build_parser, main

_ORIG_CMD_BEST_PARAMS = _cli_commands.cmd_best_params
_ORIG_CMD_SEED_DEMO = _cli_commands.cmd_seed_demo
_ORIG_CMD_TURSO_AUDIT_SYNC = _cli_commands.cmd_turso_audit_sync
_ORIG_CMD_CONTROL_LOOP = _cli_commands.cmd_control_loop
_WRAPPER_FUNCS: dict[str, object] = {}


def _sync_command_dependencies() -> None:
    for name, value in globals().items():
        if name.startswith("_"):
            continue
        if name in _WRAPPER_FUNCS and value is _WRAPPER_FUNCS[name]:
            continue
        if hasattr(_cli_commands, name):
            setattr(_cli_commands, name, value)
        if hasattr(_cli_campaigns, name):
            setattr(_cli_campaigns, name, value)


def cmd_best_params(args):
    _sync_command_dependencies()
    return _ORIG_CMD_BEST_PARAMS(args)


def cmd_seed_demo(args):
    _sync_command_dependencies()
    return _ORIG_CMD_SEED_DEMO(args)


def cmd_turso_audit_sync(args):
    _sync_command_dependencies()
    return _ORIG_CMD_TURSO_AUDIT_SYNC(args)


def cmd_control_loop(args):
    _sync_command_dependencies()
    return _ORIG_CMD_CONTROL_LOOP(args)


_WRAPPER_FUNCS = {
    "cmd_best_params": cmd_best_params,
    "cmd_seed_demo": cmd_seed_demo,
    "cmd_turso_audit_sync": cmd_turso_audit_sync,
    "cmd_control_loop": cmd_control_loop,
}

__all__ = [name for name in globals() if not name.startswith("_")]
