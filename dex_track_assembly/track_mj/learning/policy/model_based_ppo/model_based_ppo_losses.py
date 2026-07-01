"""Losses for model-based PPO."""

from typing import Any, List, Tuple

from brax.training import types
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp

from track_mj.learning.policy.model_based_ppo import model_based_ppo_networks as ppo_networks


@flax.struct.dataclass
class ModelBasedPPONetworkParams:
    policy: Params
    value: Params
    world_model: Params
    history_encoder: Params


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
    truncation_mask = 1 - truncation
    values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), acc

    (_, _), vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs_minus_v_xs, values)
    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount * (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_world_model_loss(
    trained_params: List[Params],
    normalizer_params: Any,
    data: types.Transition,
    ppo_network: ppo_networks.ModelBasedPPONetworks,
    world_model_obs_key: str = "world_state",
    world_model_gyro_weight: float = 1.0,
    world_model_gvec_weight: float = 1.0,
    world_model_joint_pos_weight: float = 1.0,
    world_model_joint_vel_weight: float = 1.0,
    world_model_root_height_weight: float = 1.0,
    world_model_autoregressive: bool = False,
) -> Tuple[jnp.ndarray, types.Metrics]:
    history_encoder_apply = ppo_network.history_encoder.apply
    world_model_apply = ppo_network.world_model.apply
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    if world_model_autoregressive:
        def _scan(carry, step_idx):
            history_obs, world_model_obs = carry
            embedding = history_encoder_apply(normalizer_params, trained_params[1], history_obs)
            predicted = world_model_apply(
                normalizer_params,
                trained_params[0],
                world_model_obs,
                embedding,
                data.action[step_idx],
            )
            done = data.discount[step_idx] < 0.1
            history_dim = predicted.shape[-1] - 1
            new_history_obs = jnp.where(
                done[..., jnp.newaxis],
                data.next_observation["history_state"][step_idx],
                jnp.concatenate([history_obs[..., history_dim:], predicted[..., :-1]], axis=-1),
            )
            new_world_model_obs = jnp.where(
                done[..., jnp.newaxis],
                data.next_observation[world_model_obs_key][step_idx],
                predicted,
            )
            return (new_history_obs, new_world_model_obs), predicted

        init_history_obs = data.observation["history_state"][0]
        init_world_model_obs = data.observation[world_model_obs_key][0]
        _, all_predicted_observations = jax.lax.scan(
            _scan,
            (init_history_obs, init_world_model_obs),
            jnp.arange(data.action.shape[0]).astype(jnp.int32),
        )
    else:
        embedding = history_encoder_apply(normalizer_params, trained_params[1], data.observation)
        all_predicted_observations = world_model_apply(
            normalizer_params,
            trained_params[0],
            data.observation,
            embedding,
            data.action,
        )

    done = data.discount < 0.1
    target = data.next_observation[world_model_obs_key]
    loss_gyro = world_model_gyro_weight * jnp.mean(jnp.abs(all_predicted_observations[..., :3] - target[..., :3]).sum(axis=-1) * (1 - done))
    loss_gvec = world_model_gvec_weight * (1 - jnp.mean(jnp.sum(all_predicted_observations[..., 3:6] * target[..., 3:6], axis=-1) * (1 - done)))
    loss_joint_pos = world_model_joint_pos_weight * jnp.mean(jnp.abs(all_predicted_observations[..., 6:35] - target[..., 6:35]).sum(axis=-1) * (1 - done))
    loss_joint_vel = world_model_joint_vel_weight * jnp.mean(jnp.abs(all_predicted_observations[..., 35:64] - target[..., 35:64]).sum(axis=-1) * (1 - done))
    loss_root_height = world_model_root_height_weight * jnp.mean(jnp.abs(all_predicted_observations[..., 64:] - target[..., 64:]).sum(axis=-1) * (1 - done))

    world_model_loss = loss_gyro + loss_gvec + loss_joint_pos + loss_joint_vel + loss_root_height
    return world_model_loss, {
        "world_model_loss": world_model_loss,
        "world_model_loss_gyro": loss_gyro,
        "world_model_loss_gvec": loss_gvec,
        "world_model_loss_joint_pos": loss_joint_pos,
        "world_model_loss_joint_vel": loss_joint_vel,
        "world_model_loss_root_height": loss_root_height,
    }


def compute_inverse_dynamics_model_loss(
    trained_params: List[Params],
    normalizer_params: Any,
    data: types.Transition,
    ppo_network: ppo_networks.ModelBasedPPONetworks,
    world_model_obs_key: str = "world_state",
) -> Tuple[jnp.ndarray, types.Metrics]:
    history_encoder_apply = ppo_network.history_encoder.apply
    world_model_apply = ppo_network.world_model.apply
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    embedding = history_encoder_apply(normalizer_params, trained_params[1], data.observation)
    predicted_action = world_model_apply(
        normalizer_params,
        trained_params[0],
        data.observation,
        embedding,
        data.next_observation[world_model_obs_key],
    )
    inverse_dynamics_model_loss = jnp.mean(jnp.square(predicted_action - data.action).sum(axis=-1))
    return inverse_dynamics_model_loss, {
        "inverse_dynamics_model_loss": inverse_dynamics_model_loss,
    }


def compute_ppo_loss_with_world_model(
    trained_params: List[Params],
    fixed_params: List[Params],
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.ModelBasedPPONetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    use_inverse_dynamics_model: bool = False,
    train_history_encoder_in_policy: bool = True,
    reference_world_model_obs_key: str = "ref_world_state",
    supervised_loss_weight: float = 1.0,
    ppo_loss_weight: float = 1.0,
    world_model_gyro_weight: float = 1.0,
    world_model_gvec_weight: float = 1.0,
    world_model_joint_pos_weight: float = 1.0,
    world_model_joint_vel_weight: float = 1.0,
    world_model_root_height_weight: float = 1.0,
) -> Tuple[jnp.ndarray, types.Metrics]:
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    history_encoder_apply = ppo_network.history_encoder.apply
    world_model_apply = ppo_network.world_model.apply

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    if not train_history_encoder_in_policy:
        embedding = history_encoder_apply(normalizer_params, fixed_params[1], data.observation)
    else:
        embedding = history_encoder_apply(normalizer_params, trained_params[2], data.observation)
    policy_logits = policy_apply(normalizer_params, trained_params[0], data.observation, embedding)

    if supervised_loss_weight > 0.0:
        if use_inverse_dynamics_model:
            mean_actions = parametric_action_distribution.mode(policy_logits)
            predicted_action = world_model_apply(
                normalizer_params,
                fixed_params[0],
                data.observation,
                embedding,
                data.next_observation[reference_world_model_obs_key],
            )
            policy_supervised_loss = supervised_loss_weight * jnp.mean(jnp.square(predicted_action - mean_actions).sum(axis=-1))
        else:
            sampled_actions = parametric_action_distribution.sample(policy_logits, rng)
            predicted_next_observation = world_model_apply(
                normalizer_params,
                fixed_params[0],
                data.observation,
                embedding,
                sampled_actions,
            )
            target = data.next_observation[reference_world_model_obs_key]
            loss_gyro = jnp.mean(jnp.abs(predicted_next_observation[..., :3] - target[..., :3]).sum(axis=-1))
            loss_gvec = 1 - jnp.mean(jnp.sum(predicted_next_observation[..., 3:6] * target[..., 3:6], axis=-1))
            loss_joint_pos = jnp.mean(jnp.abs(predicted_next_observation[..., 6:35] - target[..., 6:35]).sum(axis=-1))
            loss_joint_vel = jnp.mean(jnp.abs(predicted_next_observation[..., 35:64] - target[..., 35:64]).sum(axis=-1))
            loss_root_height = jnp.mean(jnp.abs(predicted_next_observation[..., 64:] - target[..., 64:]).sum(axis=-1))
            policy_supervised_loss = supervised_loss_weight * (
                loss_gyro * world_model_gyro_weight
                + loss_gvec * world_model_gvec_weight
                + loss_joint_pos * world_model_joint_pos_weight
                + loss_joint_vel * world_model_joint_vel_weight
                + loss_root_height * world_model_root_height_weight
            )
    else:
        policy_supervised_loss = 0.0

    if ppo_loss_weight > 0.0:
        baseline = value_apply(normalizer_params, trained_params[1], data.observation, embedding)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        if not train_history_encoder_in_policy:
            terminal_embedding = history_encoder_apply(normalizer_params, fixed_params[1], terminal_obs)
        else:
            terminal_embedding = history_encoder_apply(normalizer_params, trained_params[2], terminal_obs)
        bootstrap_value = value_apply(normalizer_params, trained_params[1], terminal_obs, terminal_embedding)

        rewards = data.reward * reward_scaling
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.discount) * (1 - truncation)

        target_action_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras["policy_extras"]["raw_action"]
        )
        behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

        vs, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2)) * ppo_loss_weight

        v_error = vs - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5 * ppo_loss_weight
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
        entropy_loss = entropy_cost * -entropy * ppo_loss_weight
    else:
        policy_loss = 0.0
        v_loss = 0.0
        entropy_loss = 0.0

    total_loss = policy_loss + v_loss + entropy_loss + policy_supervised_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "policy_supervised_loss": policy_supervised_loss,
    }
