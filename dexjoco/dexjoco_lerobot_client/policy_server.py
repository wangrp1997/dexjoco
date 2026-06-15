"""DexJoCo policy server with fixes for queue-based LeRobot policies (Diffusion, DiT, ...)."""

import logging
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Any

import draccus
import grpc
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.utils import populate_queues
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import ACTION, OBS_IMAGES

# Policies validated on DexJoCo in addition to LeRobot async_inference defaults.
_DEXJOCO_EXTRA_POLICIES = ("multi_task_dit",)


def _supported_policies() -> set[str]:
    return set(SUPPORTED_POLICIES) | set(_DEXJOCO_EXTRA_POLICIES)


def _uses_observation_queues(policy) -> bool:
    if not hasattr(policy, "_queues"):
        return False
    return int(getattr(policy.config, "n_obs_steps", 1)) > 1


def _prepare_queued_policy_batch(policy, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mirror ``select_action`` queue handling before ``predict_action_chunk``."""
    batch = dict(observation)
    batch.pop(ACTION, None)

    if not _uses_observation_queues(policy):
        return batch

    image_features = getattr(policy.config, "image_features", None) or {}
    if image_features and OBS_IMAGES not in batch:
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_features], dim=-4)

    policy._queues = populate_queues(policy._queues, batch)
    return batch


class DexJoCoPolicyServer(PolicyServer):
    """Policy server that fills observation queues for temporal policies."""

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
            self.logger.debug("Policy observation queues reset")
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        # Parent PolicyServer binds SUPPORTED_POLICIES at import time; patch that module.
        import lerobot.async_inference.policy_server as lerobot_policy_server

        original_supported = list(lerobot_policy_server.SUPPORTED_POLICIES)
        extended = list(_supported_policies())
        lerobot_policy_server.SUPPORTED_POLICIES = extended
        try:
            response = super().SendPolicyInstructions(request, context)
        finally:
            lerobot_policy_server.SUPPORTED_POLICIES = original_supported

        if self.policy is not None and _uses_observation_queues(self.policy):
            self.logger.info(
                "Loaded queue-based policy (%s, n_obs_steps=%s)",
                self.policy_type,
                self.policy.config.n_obs_steps,
            )
        return response

    def _get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        batch = _prepare_queued_policy_batch(self.policy, observation)
        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start DexJoCo's patched PolicyServer."""
    logging.info(pformat(asdict(cfg)))

    policy_server = DexJoCoPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"DexJoCo PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")
