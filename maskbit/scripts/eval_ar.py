import os
import argparse
import math
from pathlib import Path
import pprint

import torch
from omegaconf import OmegaConf
import tqdm

from modeling.conv_vqgan import FTConvVQModel
# from modeling.bert import Bert, LFQBert
from modeling.rqgan import RQModel
from modeling.cond_bert import CondBert, CondLFQBert
from modeling.ar import ResAR
from modeling.modules import sample, conditional_sample, residual_sample
from utils.adm_eval_suite import Evaluator
from data import SimpleImagenet

from utils.script_utils import get_model_kwargs, get_sampling_kwargs, get_config

import tensorflow.compat.v1 as tf

import wandb

TRAIN_SET_STATISTICS_256 = "train_imagenet256_stats.npz"
TRAIN_SET_STATISTICS_512 = "train_imagenet512_stats.npz"




@torch.no_grad()
def get_tokenizer(config, tokenizer_path):
    tokenizer_cls = {
        "rqgan": RQModel,
        "ft-vqgan+": FTConvVQModel,
    }
    tokenizer_model = tokenizer_cls[config.model.vq_model.model_class](config.model.vq_model)
    tokenizer_model.load_pretrained(tokenizer_path)
    tokenizer_model.eval()
    tokenizer_model.requires_grad_(False)
    return tokenizer_model


@torch.no_grad()
def get_generator(config, generator_path=None):
    ar_model = ResAR(config)
    if generator_path is not None:
        ar_model.load_pretrained(generator_path)
    ar_model.eval()
    ar_model.requires_grad_(False)
    return ar_model


def eval(
    config,
    device: str = "cuda:0",
):
    # config = OmegaConf.load(config_file)
    # config = get_config()
    batchsize = config.training.per_gpu_batch_size
    tokenizer_path = config.experiment.vqgan_checkpoint
    generator_path = config.experiment.ar_checkpoint

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # if config.model.vq_model.quantizer_type == "lookup-free":
    #     num_codebook_entries = 2 ** config.model.vq_model.token_size
    #     config.model.vq_model.codebook_size = num_codebook_entries
    #     config.model.mlm_model.mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))
    # else:
    #     num_codebook_entries = config.model.vq_model.codebook_size
    #     config.model.mlm_model.mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))

    tokenizer_model = get_tokenizer(config, tokenizer_path).to(device)
    generator_model = get_generator(config, generator_path).to(device)
    
    ##################################
    # EVALUATION STUFF.              #
    ##################################
    with torch.no_grad():

        tokenizer_model.eval()
        generator_model.eval()
        total_samples = 50_000

        generated_list = []

        tf_config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        tf_config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=tf_config))

        print("warming up TensorFlow...")
        evaluator.warmup()
        tot_samples = 0
        n_classes = 1000
        # This is important due to how the Inception Score is computed in the tensorflow suite.
        # The computation is done batchwise, hence we need to shuffle everithing to have a good representation per batch.
         # This is important due to how the Inception Score is computed in the tensorflow suite.
        # The computation is done batchwise, hence we need to shuffle everithing to have a good representation per batch.
        labels = torch.randperm(n_classes, dtype=torch.int, device=device)
        labels = labels.repeat(int(total_samples // n_classes))
        print("Running generation...")
        for i in tqdm.tqdm(range(total_samples//batchsize), desc="Generating samples", position=0):
            y = labels[batchsize*i: batchsize*(i+1)].long()
            generated_samples, _ = residual_sample(
                generator_model,
                tokenizer_model,
                labels=y,
            )
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)
            generated_samples = (generated_samples * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            generated_list.append(generated_samples)
            tot_samples += generated_samples.shape[0]
        print(f"Generated {tot_samples} samples.")
        # if res == 256:
        #     stat_file = TRAIN_SET_STATISTICS_256
        # elif res == 512:
        #     stat_file = TRAIN_SET_STATISTICS_512
        # else:
        #     raise ValueError("res must be 256 or 512")
        stat_file = TRAIN_SET_STATISTICS_256
        
        current_file_dir = Path(__file__).parent

        stats_file_path = (
            current_file_dir 
            / Path("..") 
            / "metrics" 
            / "stats" 
            / stat_file
        ).resolve()

        print("Running evaluation...")

        ref_stats = evaluator.read_statistics(stats_file_path, None)

        sample_acts = evaluator.compute_activations(generated_list)
        sample_stats = evaluator.compute_statistics(sample_acts)

        eval_scores = {
            "InceptionScore": evaluator.compute_inception_score(sample_acts),
            "FID": sample_stats.frechet_distance(ref_stats),
        }

    print("EVALUATION")
    pprint.pprint(eval_scores)
    return eval_scores

def main():
    from utils.logger import setup_logger

    config = get_config()
    work_dir = os.environ.get('WORKSPACE', './runs')
    output_dir = os.path.join(work_dir, "outputs", f"eval-{config.experiment.name}", config.experiment.run_name)

    logger = setup_logger(name="eval_ar", log_level="INFO", output_dir=output_dir, use_accelerate=False)

    import uuid
    run_id_file = os.path.join(output_dir, "wandb_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
    else:
        run_id = str(uuid.uuid4())
        with open(run_id_file, "w") as f:
            f.write(run_id)
    
    wandb.init(
        project=f"eval-{config.experiment.name}",
        name=config.experiment.run_name,
        config=OmegaConf.to_container(config, resolve=True),
        resume="allow",
        id=run_id,
    )

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    keys = ["num_steps", "guidance_scale"]

    from copy import deepcopy
    from collections.abc import Iterable
    from itertools import product
    for key in keys:
        if not isinstance(config.model.cond_model.get(key, None), Iterable):
            config.model.cond_model[key] = [config.model.cond_model[key]]
    # Get all combinations of values for the keys
    values_list = [config.model.cond_model[key] for key in keys]
    for values in product(*values_list):
        config_copy = deepcopy(config)
        for i, key in enumerate(keys):
            config_copy.model.cond_model[key] = values[i]
        logger.info(f"Running eval with {dict(zip(keys, values))}")
        eval_scores = eval(config_copy)
        # wandb.log({**dict(zip(keys, values)), **eval_scores})
        logger.info(f"Eval scores: {eval_scores}")
    
    


if __name__ == "__main__":
    main()
