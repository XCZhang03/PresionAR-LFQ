import torch
import os
from pathlib import Path
import json

def save_stage_model_ckpt(model_path):
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "pytorch_model.bin"
    
    model_dir = model_path.parent
    model = torch.load(model_path, map_location="cpu")
    
    base_model = {}
    for k, v in model.items():
        if k.startswith('base_model.'):
            base_model[k[len('base_model.'):]] = v
    torch.save(base_model, model_dir / "base_model.bin")

    for i in range(3):
        stage_i_model = {}
        prefix = f'cond_models.{i}.'
        for k, v in model.items():
            if k.startswith(prefix):
                stage_i_model[k[len(prefix):]] = v
        if stage_i_model:
            torch.save(stage_i_model, model_dir / f"stage_{i+1}_model.bin")


def compose_stage_model_checkpoint(ar_model_path, **kwargs):
    ar_model_path = Path(ar_model_path)
    if ar_model_path.is_dir():
        ar_model_path = ar_model_path / "pytorch_model.bin"
    
    ar_model_dir = ar_model_path.parent
    ar_model = torch.load(ar_model_path, map_location="cpu")
    
    for stage in range(4):
        stage_model_path = kwargs.get(f'stage_{stage}_model_path', None)
        if stage_model_path is None:
            continue
        stage_model_path = Path(stage_model_path)
        if stage_model_path.is_dir():
            stage_model_path = stage_model_path / "pytorch_model.bin"
        stage_model = torch.load(stage_model_path, map_location="cpu")
        prefix = 'base_model.' if stage == 0 else f'cond_models.{stage-1}.'
        for k, v in stage_model.items():
            ar_model[prefix + k] = v
        print(f"Integrated stage {stage} model from {stage_model_path}")

    save_dir = Path(f"{ar_model_dir}/composed_model")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(ar_model, save_dir / "pytorch_model.bin")
    with open(save_dir / "model_sources.json", "w") as f:
        json.dump(kwargs, f, indent=2)
    print(f"Composed model saved to {save_dir / 'pytorch_model.bin'}")

if __name__ == "__main__":
    model_path = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-resume/checkpoints/checkpoint_50/ema_model/pytorch_model.bin"
    stage_0_model_path = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/maskbit_generator_10bit/ft-2lvl-test/checkpoints/checkpoint_36/ema_model/pytorch_model.bin"
    # save_stage_model_ckpt(model_path)
    compose_stage_model_checkpoint(model_path, stage_0_model_path=stage_0_model_path)