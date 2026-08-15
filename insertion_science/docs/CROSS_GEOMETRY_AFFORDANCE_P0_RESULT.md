# Cross-Geometry Contact-Affordance P0 Result

- 完成：2026-08-15T09:09:11Z
- 判定：`fail_stop_affordance_direction`
- 通过：False
- 停止该方向：True
- 摘要：contact-affordance 未在全部 family held-out 上稳定优于 tip/lat/axis，或 shuffle/instance 负对照未过；停止该方向。

- 样本数：192；标签：{'blocked': 96, 'jam': 94, 'free': 2}；feasible_rate=0.010
- mean aff−tip：0.062（门槛 0.02）
- mean shuffle drop：-0.010（门槛 0.05）
- instance aff−tip：0.000

## 重要限制

- `free` 仅 2/192，二分类探针接近全负类；多数 fold 的高 accuracy 不能解读为“表示已学会接触约束”。
- 在预注册通过线下仍判定失败：affordance 未在全部 family fold 稳定优于 tip，shuffle 未下降。
- **停止 Contact-Affordance 方向**；不据此宣称仿真插孔不可解。

## Family LOO folds

```json
{
  "folds": [
    {
      "held_family": "rectangular_12mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "raw_relation": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance": {
          "accuracy": 0.96875,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        }
      },
      "aff_minus_tip": -0.03125,
      "shuffle_drop": -0.03125
    },
    {
      "held_family": "rectangular_16mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 0.78125,
          "f1": 0.2222222222222222,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "raw_relation": {
          "accuracy": 0.78125,
          "f1": 0.2222222222222222,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "contact_affordance": {
          "accuracy": 1.0,
          "f1": 1.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 1.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        }
      },
      "aff_minus_tip": 0.21875,
      "shuffle_drop": 0.0
    },
    {
      "held_family": "rectangular_8mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "raw_relation": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        }
      },
      "aff_minus_tip": 0.0,
      "shuffle_drop": 0.0
    },
    {
      "held_family": "round_12mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "raw_relation": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance": {
          "accuracy": 0.96875,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        }
      },
      "aff_minus_tip": -0.03125,
      "shuffle_drop": -0.03125
    },
    {
      "held_family": "round_16mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 0.78125,
          "f1": 0.2222222222222222,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "raw_relation": {
          "accuracy": 0.78125,
          "f1": 0.2222222222222222,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "contact_affordance": {
          "accuracy": 1.0,
          "f1": 1.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 1.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.00625,
          "test_pos_rate": 0.03125
        }
      },
      "aff_minus_tip": 0.21875,
      "shuffle_drop": 0.0
    },
    {
      "held_family": "round_8mm",
      "n_train": 160,
      "n_test": 32,
      "reps": {
        "tip_lat_axis": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "raw_relation": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        },
        "contact_affordance_shuffled": {
          "accuracy": 1.0,
          "f1": 0.0,
          "degenerate_train": false,
          "n_train": 160,
          "n_test": 32,
          "train_pos_rate": 0.0125,
          "test_pos_rate": 0.0
        }
      },
      "aff_minus_tip": 0.0,
      "shuffle_drop": 0.0
    }
  ],
  "families": [
    "rectangular_12mm",
    "rectangular_16mm",
    "rectangular_8mm",
    "round_12mm",
    "round_16mm",
    "round_8mm"
  ]
}
```

## Instance holdout

```json
{
  "tip_lat_axis": {
    "accuracy": 1.0,
    "f1": 0.0,
    "degenerate_train": true,
    "n_train": 64,
    "n_test": 64,
    "train_pos_rate": 0.0,
    "test_pos_rate": 0.0
  },
  "raw_relation": {
    "accuracy": 1.0,
    "f1": 0.0,
    "degenerate_train": true,
    "n_train": 64,
    "n_test": 64,
    "train_pos_rate": 0.0,
    "test_pos_rate": 0.0
  },
  "contact_affordance": {
    "accuracy": 1.0,
    "f1": 0.0,
    "degenerate_train": true,
    "n_train": 64,
    "n_test": 64,
    "train_pos_rate": 0.0,
    "test_pos_rate": 0.0
  },
  "aff_minus_tip": 0.0
}
```

## 决策

- **立即停止** Cross-Geometry Contact-Affordance 方向。
- 不训练 generalist insertion policy。
- 下一步应审查阶段接口与数据支持，而非宣称仿真不可解。
