"""Network definitions for model-based PPO."""

import dataclasses
from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import types
from brax.training.acme import running_statistics
from flax import linen
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
        return hidden


class MLPWithAdapter(linen.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    def setup(self):
        if self.layer_norm or self.activate_final:
            raise NotImplementedError("Layer norm and activate_final are not supported for MLPWithAdapter")

        self.base_layers = [
            linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)
            for i, hidden_size in enumerate(self.layer_sizes)
        ]
        self.adapter_layers = [
            linen.Dense(
                hidden_size,
                name=f"adapter_{i}",
                kernel_init=jax.nn.initializers.zeros,
                use_bias=self.bias,
                bias_init=jax.nn.initializers.zeros,
            )
            for i, hidden_size in enumerate(self.layer_sizes)
        ]

    def __call__(self, data: jnp.ndarray, extra_data: jnp.ndarray):
        base_hidden = self.base_layers[0](data)
        adapter_hidden = self.adapter_layers[0](extra_data)

        for i in range(1, len(self.base_layers)):
            out = self.activation(base_hidden + adapter_hidden)

            base_hidden = self.base_layers[i](out)
            adapter_hidden = self.adapter_layers[i](out)

        return base_hidden + adapter_hidden


class ConvMLP(linen.Module):
    num_filters: Sequence[int]
    kernel_sizes: Sequence[Tuple]
    strides: Sequence[Tuple]
    history_len: int
    hidden_layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    use_bias: bool = True
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        hidden = hidden.reshape(*hidden.shape[:-1], self.history_len, -1)
        for i, (num_filter, kernel_size, stride) in enumerate(zip(self.num_filters, self.kernel_sizes, self.strides)):
            hidden = linen.Conv(
                num_filter,
                kernel_size=kernel_size,
                strides=stride,
                padding="VALID",
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                name=f"conv_{i}",
            )(hidden)
            hidden = self.activation(hidden)

        hidden = hidden.reshape(*hidden.shape[:-2], -1)

        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.use_bias,
            )(hidden)
            if i != len(self.hidden_layer_sizes) - 1:
                hidden = self.activation(hidden)
        return hidden


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    embedding_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = "state",
) -> FeedForwardNetwork:
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs, embedding):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        network_input = jnp.concatenate([obs, embedding], axis=-1)
        return policy_module.apply(policy_params, network_input)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size + embedding_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_policy_adapter_network(
    param_size: int,
    obs_size: types.ObservationSize,
    embedding_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = "state",
) -> FeedForwardNetwork:
    policy_module = MLPWithAdapter(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs, embedding):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs, embedding)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size))
    dummy_embedding = jnp.zeros((1, embedding_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs, dummy_embedding), apply=apply)


def make_value_network(
    obs_size: types.ObservationSize,
    embedding_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "privileged_state",
) -> FeedForwardNetwork:
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, value_params, obs, history_embedding):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        network_input = jnp.concatenate([obs, history_embedding], axis=-1)
        return jnp.squeeze(value_module.apply(value_params, network_input), axis=-1)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size + embedding_size))
    return FeedForwardNetwork(init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_history_encoder_mlp(
    obs_size: types.ObservationSize,
    embedding_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "history_state",
) -> FeedForwardNetwork:
    history_encoder_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [embedding_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, history_encoder_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return history_encoder_module.apply(history_encoder_params, obs)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size))
    return FeedForwardNetwork(init=lambda key: history_encoder_module.init(key, dummy_obs), apply=apply)


def make_history_encoder_conv(
    obs_size: types.ObservationSize,
    embedding_size: int,
    history_len: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    num_filters=(128, 256),
    kernel_sizes=((5,), (3,)),
    strides=((1,), (1,)),
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "history_state",
) -> FeedForwardNetwork:
    history_encoder_module = ConvMLP(
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        history_len=history_len,
        hidden_layer_sizes=list(hidden_layer_sizes) + [embedding_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, history_encoder_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return history_encoder_module.apply(history_encoder_params, obs)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size))
    return FeedForwardNetwork(init=lambda key: history_encoder_module.init(key, dummy_obs), apply=apply)


def make_world_model(
    obs_size: types.ObservationSize,
    embedding_size: int,
    action_size: int,
    output_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "world_state",
    joint_vel_scale: float = 0.05,
    dt: float = 0.02,
) -> FeedForwardNetwork:
    world_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [output_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, world_model_params, obs, history_embedding, action):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        network_input = jnp.concatenate([obs, history_embedding, action], axis=-1)
        predicted_delta_obs = world_module.apply(world_model_params, network_input)

        scaled_new_joint_vel = obs[..., 35:64] + predicted_delta_obs[..., 3:32]
        new_joint_pos = obs[..., 6:35] + scaled_new_joint_vel / joint_vel_scale * dt
        new_root_height = obs[..., 64:] + predicted_delta_obs[..., 32:]
        scaled_new_gyro = obs[..., :3] + predicted_delta_obs[..., :3]

        old_gvec = obs[..., 3:6]
        new_gvec = old_gvec - jnp.cross(scaled_new_gyro / joint_vel_scale, old_gvec) * dt
        new_gvec_norm = new_gvec / (jnp.linalg.norm(new_gvec + 1e-4, axis=-1, keepdims=True))

        return jnp.concatenate([scaled_new_gyro, new_gvec_norm, new_joint_pos, scaled_new_joint_vel, new_root_height], axis=-1)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, state_size + embedding_size + action_size))
    return FeedForwardNetwork(init=lambda key: world_module.init(key, dummy_obs), apply=apply)


def make_inverse_dynamics_model(
    obs_size: types.ObservationSize,
    embedding_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "world_state",
) -> FeedForwardNetwork:
    inverse_dynamics_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, inverse_dynamics_params, obs, history_embedding, next_obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(obs[obs_key], normalizer_select(processor_params, obs_key))
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        network_input = jnp.concatenate([obs, history_embedding, next_obs], axis=-1)
        return inverse_dynamics_module.apply(inverse_dynamics_params, network_input)

    state_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, 2 * state_size + embedding_size))
    return FeedForwardNetwork(init=lambda key: inverse_dynamics_module.init(key, dummy_obs), apply=apply)


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState,
    obs_key: str,
) -> running_statistics.RunningStatisticsState:
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
    )
