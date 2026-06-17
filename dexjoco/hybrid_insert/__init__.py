"""VLA + classical insert controller for DexJoCo bimanual assembly."""

from .assembly_contacts import AssemblyContactLabeler, AssemblyOutcome
from .config import HybridInsertConfig
from .controller import HybridInsertController
from .integration import EvalHybridInsert, get_raw_env, state_to_dual_arm_action44

SUPPORTED_TASKS = frozenset({"bimanual_assembly"})

__all__ = [
    "AssemblyContactLabeler",
    "AssemblyOutcome",
    "HybridInsertConfig",
    "HybridInsertController",
    "EvalHybridInsert",
    "SUPPORTED_TASKS",
    "get_raw_env",
    "state_to_dual_arm_action44",
]
