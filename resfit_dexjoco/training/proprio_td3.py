"""Proprio-only residual TD3 agent (adapted from ResFiT QAgent / Actor / Critic)."""

from __future__ import annotations

import copy

import torch
from torch import nn

from resfit_dexjoco.training.resfit_mlp import ActorConfig, TruncatedNormal, build_fc, initialize_layer_weights


class ProprioResidualActor(nn.Module):
    """ResFiT Actor without vision: input = state + base_action."""

    def __init__(self, state_dim: int, action_dim: int, cfg: ActorConfig):
        super().__init__()
        self.cfg = cfg
        policy_in_dim = state_dim + action_dim
        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            action_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
        if cfg.actor_last_layer_init_scale is not None:
            final_layer = None
            for module in reversed(list(self.policy.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break
            if final_layer is not None:
                initialize_layer_weights(
                    final_layer,
                    cfg.actor_last_layer_init_distribution,
                    cfg.actor_last_layer_init_scale,
                )

    def forward(self, state: torch.Tensor, base_action: torch.Tensor, std: float):
        policy_input = torch.cat([state, base_action], dim=-1)
        mu = self.policy(policy_input) * self.cfg.action_scale
        return TruncatedNormal(mu, std)


class ProprioCritic(nn.Module):
    """Twin-Q critic Q(state, combined_action)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = build_fc(in_dim, hidden_dim, 1, num_layer=2, layer_norm=1, dropout=0.0)
        self.q2 = build_fc(in_dim, hidden_dim, 1, num_layer=2, layer_norm=1, dropout=0.0)

    def forward(self, state: torch.Tensor, combined_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, combined_action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, combined_action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, combined_action], dim=-1))


class ProprioResidualTD3:
    """TD3 residual learner aligned with ResFiT update rules (proprio-only)."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-4,
        critic_tau: float = 0.005,
        gamma: float = 0.99,
        stddev: float = 0.05,
        stddev_clip: float = 0.3,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        actor_config: ActorConfig | None = None,
    ):
        self.device = device
        self.gamma = gamma
        self.stddev = stddev
        self.stddev_clip = stddev_clip
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        actor_cfg = actor_config or ActorConfig(
            action_scale=0.1,
            actor_last_layer_init_scale=0.0,
            hidden_dim=1024,
        )

        self.actor = ProprioResidualActor(state_dim, action_dim, actor_cfg).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = ProprioCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.critic_tau = critic_tau

    def act(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        *,
        eval_mode: bool = False,
        explore_std: float | None = None,
    ) -> torch.Tensor:
        std = 0.0 if eval_mode else (self.stddev if explore_std is None else explore_std)
        dist = self.actor(state, base_action, std)
        return dist.mean if eval_mode else dist.sample(clip=self.stddev_clip)

    @torch.no_grad()
    def _target_residual(self, next_state: torch.Tensor, next_base_action: torch.Tensor) -> torch.Tensor:
        noise = (
            torch.randn_like(next_base_action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        residual = self.actor_target(next_state, next_base_action, std=0.0).mean
        return (next_base_action + residual + noise).clamp(-1.0, 1.0) - next_base_action

    def update(self, batch, *, actor_update: bool = True) -> dict[str, float]:
        state = torch.as_tensor(batch.state, device=self.device)
        base_action = torch.as_tensor(batch.base_action, device=self.device)
        combined_action = torch.as_tensor(batch.combined_action, device=self.device)
        reward = torch.as_tensor(batch.reward, device=self.device).reshape(-1, 1)
        next_state = torch.as_tensor(batch.next_state, device=self.device)
        next_base_action = torch.as_tensor(batch.next_base_action, device=self.device)
        done = torch.as_tensor(batch.done, device=self.device).reshape(-1, 1).float()
        discount = self.gamma * (1.0 - done)

        with torch.no_grad():
            next_residual = self._target_residual(next_state, next_base_action)
            next_combined = torch.clamp(next_base_action + next_residual, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_state, next_combined)
            target_q = torch.min(target_q1, target_q2)
            y = reward + discount * target_q

        q1, q2 = self.critic(state, combined_action)
        critic_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": float(critic_loss.item())}
        if actor_update:
            residual = self.act(state, base_action, eval_mode=False, explore_std=0.0)
            combined = torch.clamp(base_action + residual, -1.0, 1.0)
            actor_loss = -self.critic.q1_forward(state, combined).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            metrics["actor_loss"] = float(actor_loss.item())

            for target, source in (
                (self.critic_target, self.critic),
                (self.actor_target, self.actor),
            ):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.critic_tau * sp.data + (1.0 - self.critic_tau) * tp.data)

        return metrics
