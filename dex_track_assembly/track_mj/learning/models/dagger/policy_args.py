from dataclasses import dataclass, asdict, fields, field
from ml_collections import config_dict

@dataclass
class PolicyArgs:
    policy_type: str = "mlp"

    # ===== model general config =====
    obs_dim: int = 0
    aux_obs_dim: int = 0
    act_dim: int = 0
    bf16: bool = False
    load_path: str = ""
    policy_obs_key: str = "state"
    policy_auxiliary_obs_key: str = "auxiliary_state"
    output_residual_action: bool = True
    # ===== BC loss config =====
    # Supported values:
    #   - "mse"
    #   - "mae"
    #   - "mse+mae"
    loss_type: str = "mse"
    mse_loss_coef: float = 1.0
    mae_loss_coef: float = 1.0

    # ===== MLP config =====
    # mlp_hidden_dim: list[int] = field(default_factory=lambda: [4096, 4096, 2048, 2048, 1024, 1024, 512])
    mlp_hidden_dim: list[int] = field(default_factory=lambda: [1024, 1024, 512, 512, 256])
    # mlp_hidden_dim: list[int] = field(default_factory=lambda: [512, 512, 256, 256, 128])
    mlp_activate_final: bool = False

    def __post_init__(self):
        normalized_loss_type = self.loss_type.lower().replace(" ", "")
        valid_loss_types = {"mse", "mae", "mse+mae"}
        if normalized_loss_type not in valid_loss_types:
            raise ValueError(f"Unsupported loss_type '{self.loss_type}'. Supported: {sorted(valid_loss_types)}")
        self.loss_type = normalized_loss_type

        if self.mse_loss_coef < 0 or self.mae_loss_coef < 0:
            raise ValueError("mse_loss_coef and mae_loss_coef must be non-negative.")
        if self.loss_type == "mse+mae" and (self.mse_loss_coef + self.mae_loss_coef) <= 0:
            raise ValueError("For loss_type='mse+mae', at least one of mse_loss_coef or mae_loss_coef must be > 0.")
    
    def to_config_dict(self) -> config_dict.ConfigDict:
        return config_dict.create(**asdict(self))
    
    @classmethod
    def from_config_dict(cls, cfg: config_dict.ConfigDict) -> "PolicyArgs":
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in field_names}
        return cls(**filtered)


@dataclass
class ONNXPolicyArgs(PolicyArgs):
    onnx_dir: str = ""