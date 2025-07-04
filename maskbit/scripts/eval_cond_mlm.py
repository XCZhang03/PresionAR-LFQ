import os
import argparse
import math
from pathlib import Path
import pprint

import torch
from omegaconf import OmegaConf
import tqdm

# from modeling.conv_vqgan import ConvVQModel
# from modeling.bert import Bert, LFQBert
from modeling.rqgan import RQModel
from modeling.cond_bert import CondBert, CondLFQBert
from modeling.modules import sample, conditional_sample
from utils.adm_eval_suite import Evaluator
from data import SimpleImagenet

from utils.script_utils import get_model_kwargs, get_sampling_kwargs, get_config

import tensorflow.compat.v1 as tf

TRAIN_SET_STATISTICS_256 = "train_imagenet256_stats.npz"
TRAIN_SET_STATISTICS_512 = "train_imagenet512_stats.npz"




@torch.no_grad()
def get_tokenizer(config, tokenizer_path):
    tokenizer_model = RQModel(config.model.vq_model, legacy=False)
    tokenizer_model.load_pretrained(tokenizer_path)
    tokenizer_model.eval()
    tokenizer_model.requires_grad_(False)
    return tokenizer_model


@torch.no_grad()
def get_generator(config, generator_path):
    stage2_model_cls = {
        "cond_bert": CondBert,
        "cond_lfq_bert": CondLFQBert,
    }[config.model.mlm_model.model_cls]
    
    generator_model = stage2_model_cls(
        **get_model_kwargs(config)
    )
    # rename_dict = {"token_emb": "input_proj"}
    rename_dict = {}
    generator_model.load_pretrained(generator_path, rename_keys=rename_dict)
    generator_model.eval()
    generator_model.requires_grad_(False)
    return generator_model


def main(
    device: str = "cuda:0",
):
    # config = OmegaConf.load(config_file)
    config = get_config()
    batchsize = config.training.per_gpu_batch_size
    tokenizer_path = config.experiment.vqgan_checkpoint
    generator_path = config.experiment.mlm_checkpoint

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
    config.model.mlm_model.mask_token = generator_model.mask_token

    ##################################
    # DATLOADER                      #
    ##################################

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleImagenet(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=batchsize,
        global_batch_size=batchsize,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resolution=preproc_config.resolution,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        use_aspect_ratio_aug=preproc_config.use_aspect_ratio_aug,
        use_random_crop=preproc_config.use_random_crop,
        min_scale=preproc_config.min_scale,
        interpolation=preproc_config.interpolation,
    )
    eval_dataloader = dataset.eval_dataloader

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
        # This is important due to how the Inception Score is computed in the tensorflow suite.
        # The computation is done batchwise, hence we need to shuffle everithing to have a good representation per batch.
        for batch in tqdm.tqdm(eval_dataloader, desc="Generating samples", position=0):
            total_samples += batch["image"].shape[0]
            images = batch["image"].to(
                device, memory_format=torch.contiguous_format, non_blocking=True
            )
            class_tokens = batch["class_id"].to(
                device, memory_format=torch.contiguous_format, non_blocking=True
            )
            context, encoder_dict = tokenizer_model.encode(images, num_levels=config.model.mlm_model.stage)
            generated_samples, _ = conditional_sample(
                generator_model,
                tokenizer_model,
                context=context,
                num_samples=class_tokens.shape[0],
                labels=class_tokens,
                **get_sampling_kwargs(config)
            )
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)
            generated_samples = (generated_samples * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            generated_list.append(generated_samples)
        print(f"Generated {total_samples} samples.")
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
    print(f"Results for {config.model.vq_model.token_size} bits with {config.model.mlm_model.num_steps} steps.")
    pprint.pprint(eval_scores)
    return eval_scores



if __name__ == "__main__":
    main()
