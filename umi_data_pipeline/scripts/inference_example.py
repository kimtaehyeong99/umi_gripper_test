#!/usr/bin/env python3
"""
UMI Policy Inference Example

학습된 체크포인트로 action을 추론하는 예제입니다.

Usage:
    python3 inference_example.py --ckpt /path/to/checkpoint.ckpt --zarr /path/to/dataset.zarr.zip
"""

import sys
import os
import argparse
import numpy as np
import torch
import dill

# Add detached-umi-policy to path
UMI_POLICY_PATH = os.path.expanduser("~/umi_ws/src/detached-umi-policy")
sys.path.insert(0, UMI_POLICY_PATH)

import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


class UMIPolicyInference:
    """UMI Diffusion Policy 추론 클래스"""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        """
        Args:
            ckpt_path: 체크포인트 파일 경로 (.ckpt)
            device: 추론 디바이스 ("cuda" or "cpu")
        """
        print(f"Loading checkpoint: {ckpt_path}")

        # Load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']

        print(f"Config: {self.cfg.name}")
        print(f"Policy: {self.cfg.policy._target_}")

        # Create workspace and load model
        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # Get policy (use EMA if available)
        self.policy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")

        # Set inference mode
        self.policy.num_inference_steps = 16  # DDIM steps
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()

        # Get observation steps from config
        self.n_obs_steps = self.cfg.n_obs_steps
        print(f"n_obs_steps: {self.n_obs_steps}")
        print(f"Model loaded successfully!")

    def predict_action(self, obs_dict_np: dict) -> np.ndarray:
        """
        관측값으로부터 action을 예측합니다.

        Args:
            obs_dict_np: 관측값 딕셔너리
                - camera0_rgb: [T, C, H, W] float32 (0~1 normalized, TCHW format)
                - robot0_eef_pos: [T, 3] float32
                - robot0_eef_rot_axis_angle: [T, 7] float32 (rot6d + gripper)
                - robot0_gripper_width: [T, 1] float32
                여기서 T = n_obs_steps (보통 2)

        Returns:
            action: [n_action_steps, action_dim] 예측된 action
        """
        with torch.no_grad():
            # Convert to torch tensors and add batch dimension
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # Predict
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().cpu().numpy()

        return action


def load_sample_from_zarr(zarr_path: str, n_obs_steps: int = 2):
    """Zarr 데이터셋에서 샘플 관측값을 로드합니다."""
    import zarr

    print(f"Loading sample from: {zarr_path}")
    z = zarr.open(zarr_path, 'r')

    # Load first n_obs_steps frames as sample
    sample_idx = 0

    # Images: [T, H, W, C] uint8 -> [T, C, H, W] float32 (normalized)
    images = z['data/camera0_rgb'][sample_idx:sample_idx + n_obs_steps]
    images = np.moveaxis(images, -1, 1)  # THWC -> TCHW
    images = images.astype(np.float32) / 255.0

    # Poses
    eef_pos = z['data/robot0_eef_pos'][sample_idx:sample_idx + n_obs_steps]
    eef_rot = z['data/robot0_eef_rot_axis_angle'][sample_idx:sample_idx + n_obs_steps]
    gripper = z['data/robot0_gripper_width'][sample_idx:sample_idx + n_obs_steps]

    obs_dict = {
        'camera0_rgb': images.astype(np.float32),
        'robot0_eef_pos': eef_pos.astype(np.float32),
        'robot0_eef_rot_axis_angle': eef_rot.astype(np.float32),
        'robot0_gripper_width': gripper.astype(np.float32),
    }

    print(f"Sample observation shapes:")
    for k, v in obs_dict.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    return obs_dict


def main():
    parser = argparse.ArgumentParser(description='UMI Policy Inference Example')
    parser.add_argument('--ckpt', '-c', required=True, help='Checkpoint path (.ckpt)')
    parser.add_argument('--zarr', '-z', default=None, help='Zarr dataset for sample observation')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Load policy
    policy = UMIPolicyInference(args.ckpt, args.device)

    if args.zarr:
        # Load sample observation from zarr
        obs_dict = load_sample_from_zarr(args.zarr, policy.n_obs_steps)
    else:
        # Create dummy observation for testing
        print("\nCreating dummy observation (use --zarr for real data)")
        obs_dict = {
            'camera0_rgb': np.random.rand(policy.n_obs_steps, 3, 224, 224).astype(np.float32),
            'robot0_eef_pos': np.random.rand(policy.n_obs_steps, 3).astype(np.float32),
            'robot0_eef_rot_axis_angle': np.random.rand(policy.n_obs_steps, 6).astype(np.float32),
            'robot0_gripper_width': np.random.rand(policy.n_obs_steps, 1).astype(np.float32),
        }

    # Predict action
    print("\nPredicting action...")
    action = policy.predict_action(obs_dict)

    print(f"\n=== Predicted Action ===")
    print(f"Shape: {action.shape}")
    print(f"Action (first step):")
    print(f"  Position: {action[0, :3]}")
    print(f"  Rotation (6D): {action[0, 3:9]}")
    print(f"  Gripper: {action[0, 9]}")


if __name__ == '__main__':
    main()
