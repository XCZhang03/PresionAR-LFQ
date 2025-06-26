from omegaconf import OmegaConf
import os
import glob



def get_config():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    ## overrides
    if conf.experiment.get('eval_every', None) is not None:
        conf.experiment.eval_loss_every = conf.experiment.eval_every
        conf.experiment.eval_gen_every = conf.experiment.eval_every
    if conf.model.vq_model.get("codebook_size", None) is None:
        conf.model.vq_model.codebook_size = [variants ** conf.model.vq_model.token_size for variants in conf.model.vq_model.variants]
    
    return conf

def get_save_iteration(project_dir):
    output_dir = os.path.join(project_dir, "checkpoints")
    if os.path.exists(output_dir):
        checkpoint_list = list(glob.glob(os.path.join(output_dir, "checkpoint*")))
        if len(checkpoint_list) > 0:
            fn = lambda x: int(x.split('/')[-1].split('_')[-1])
            checkpoint_indices = [fn(x) for x in checkpoint_list]
            return max(checkpoint_indices) + 1
    return 0


def get_sampling_kwargs(config):
    model_cls_name = config.model.mlm_model.get("model_cls", None)
    if model_cls_name == "cond_bert":
        sampling_kwargs = dict(
            stage=config.model.mlm_model.stage,
            softmax_temperature=config.model.mlm_model.softmax_temperature,
            randomize_temperature=config.model.mlm_model.randomize_temperature,
            mask_schedule_strategy=config.model.mlm_model.gen_mask_schedule_strategy,
            num_steps=config.model.mlm_model.num_steps,
            guidance_scale=config.model.mlm_model.guidance_scale,
            mask_token=config.model.mlm_model.mask_token,
            patch_size=int(config.dataset.preprocessing.resolution // (2**(config.model.vq_model.num_resolutions - 1))),
            guidance_annealing=config.model.mlm_model.guidance_annealing,
            scale_pow=config.model.mlm_model.get("scale_pow", 4.0),
            use_sampling_annealing=config.model.mlm_model.get("use_sampling_annealing", False),
            codebook_splits=config.model.mlm_model.codebook_splits,
            codebook_size=config.model.vq_model.codebook_size[config.model.mlm_model.stage],
            bits=config.model.vq_model.token_size,
            variants=config.model.vq_model.variants[config.model.mlm_model.stage],
        )
    elif model_cls_name == "lfq_bert":
        sampling_kwargs = dict(
            softmax_temperature=config.model.mlm_model.softmax_temperature,
            randomize_temperature=config.model.mlm_model.randomize_temperature,
            mask_schedule_strategy=config.model.mlm_model.gen_mask_schedule_strategy,
            num_steps=config.model.mlm_model.num_steps,
            guidance_scale=config.model.mlm_model.guidance_scale,
            mask_token=config.model.mlm_model.mask_token,
            patch_size = int(config.dataset.preprocessing.resolution // (2**(config.model.vq_model.num_resolutions - 1))),
            guidance_annealing=config.model.mlm_model.guidance_annealing,
            scale_pow=config.model.mlm_model.get("scale_pow", 4.0),
            use_sampling_annealing=config.model.mlm_model.get("use_sampling_annealing", False),
            codebook_size=config.model.vq_model.codebook_size[0],
            codebook_splits=config.model.mlm_model.codebook_splits,
        )
    else:
        raise ValueError(f"Unsupported model class: {model_cls_name}")
    return sampling_kwargs

def get_model_kwargs(config):
    model_cls_name = config.model.mlm_model.get("model_cls", None)
    if model_cls_name == "cond_bert":
        model_kwargs = dict(
            stage=config.model.mlm_model.stage,
            img_size=config.dataset.preprocessing.resolution,
            token_size=config.model.vq_model.token_size,
            variants=config.model.vq_model.variants[config.model.mlm_model.stage],
            codebook_size=config.model.vq_model.codebook_size[config.model.mlm_model.stage],
            hidden_dim=config.model.mlm_model.hidden_dim,
            depth=config.model.mlm_model.depth,
            heads=config.model.mlm_model.heads,
            mlp_ratio=config.model.mlm_model.mlp_ratio,
            codebook_splits=config.model.mlm_model.codebook_splits,
            input_stride=2**(config.model.vq_model.num_resolutions - 1),
            dropout=config.model.mlm_model.dropout,
            drop_path=config.model.mlm_model.drop_path,
            context_conditioning=config.model.mlm_model.context_conditioning,
            label_conditioning=config.model.mlm_model.label_conditioning,
            attn_l2_norm=config.model.mlm_model.attn_l2_norm,
            tie_embeddings=config.model.mlm_model.get("tie_embeddings", False),
            tie_context_pos_embeddings=config.model.mlm_model.get("tie_context_pos_embeddings", False),
        )
    elif model_cls_name == "lfq_bert":
        model_kwargs = dict(
            img_size=config.dataset.preprocessing.resolution,
            hidden_dim=config.model.mlm_model.hidden_dim,
            codebook_size=config.model.vq_model.codebook_size[0],
            codebook_splits=config.model.mlm_model.codebook_splits,
            depth=config.model.mlm_model.depth,
            heads=config.model.mlm_model.heads,
            mlp_dim=config.model.mlm_model.mlp_dim,
            dropout=config.model.mlm_model.dropout,
            input_stride=2**(config.model.vq_model.num_resolutions - 1),
            use_prenorm=config.model.mlm_model.use_prenorm,
        )
    else:
        raise ValueError(f"Unsupported model class: {model_cls_name}")
    return model_kwargs