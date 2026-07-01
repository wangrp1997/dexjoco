"""Model-based PPO network factory."""

from typing import Sequence, Tuple

import flax
from flax import linen
from brax.training import distribution
from brax.training import types
from brax.training.types import PRNGKey

from track_mj.learning.policy.model_based_ppo import brax_networks as networks


@flax.struct.dataclass
class ModelBasedPPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    history_encoder: networks.FeedForwardNetwork
    world_model: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: ModelBasedPPONetworks):
    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        policy_network = ppo_networks.policy_network
        history_encoder = ppo_networks.history_encoder
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            history_embedding = history_encoder.apply(params[0], params[4], observations)
            logits = policy_network.apply(params[0], params[1], observations, history_embedding)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def make_model_based_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    embedding_size: int,
    world_model_output_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    history_encoder_hidden_layer_sizes: Sequence[int] = (256,) * 2,
    history_encoder_use_conv: bool = True,
    history_encoder_num_filters=(128, 256),
    history_encoder_kernel_sizes=((5,), (3,)),
    history_encoder_strides=((1,), (1,)),
    history_len: int = 15,
    world_model_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    use_inverse_dynamics_model: bool = False,
    use_adapter: bool = True,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
    history_encoder_obs_key: str = "history_state",
    world_model_obs_key: str = "world_state",
    joint_vel_scale: float = 0.05,
    dt: float = 0.02,
) -> ModelBasedPPONetworks:
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    if use_adapter:
        policy_network = networks.make_policy_adapter_network(
            parametric_action_distribution.param_size,
            observation_size,
            embedding_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation,
            obs_key=policy_obs_key,
        )
    else:
        policy_network = networks.make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            embedding_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation,
            obs_key=policy_obs_key,
        )

    value_network = networks.make_value_network(
        observation_size,
        embedding_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )
    if history_encoder_use_conv:
        history_encoder = networks.make_history_encoder_conv(
            observation_size,
            embedding_size,
            history_len,
            preprocess_observations_fn=preprocess_observations_fn,
            num_filters=history_encoder_num_filters,
            kernel_sizes=history_encoder_kernel_sizes,
            strides=history_encoder_strides,
            hidden_layer_sizes=history_encoder_hidden_layer_sizes,
            activation=activation,
            obs_key=history_encoder_obs_key,
        )
    else:
        history_encoder = networks.make_history_encoder_mlp(
            observation_size,
            embedding_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=history_encoder_hidden_layer_sizes,
            activation=activation,
            obs_key=history_encoder_obs_key,
        )

    if use_inverse_dynamics_model:
        world_model = networks.make_inverse_dynamics_model(
            observation_size,
            embedding_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=world_model_hidden_layer_sizes,
            activation=activation,
            obs_key=world_model_obs_key,
        )
    else:
        world_model = networks.make_world_model(
            observation_size,
            embedding_size,
            action_size,
            world_model_output_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=world_model_hidden_layer_sizes,
            activation=activation,
            obs_key=world_model_obs_key,
            joint_vel_scale=joint_vel_scale,
            dt=dt,
        )

    return ModelBasedPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        history_encoder=history_encoder,
        world_model=world_model,
        parametric_action_distribution=parametric_action_distribution,
    )
