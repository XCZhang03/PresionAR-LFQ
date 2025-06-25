"""This file contains the training script for single stage MLM conditional generation."""
import json
import math
import os
import time
from pathlib import Path
import pprint
from typing import Text
import glob

import tqdm

from accelerate.utils import DistributedType, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils.dataclasses import ProjectConfiguration

from data import SimpleImagenet
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from utils.logger import setup_logger
from utils.meter import AverageMeter
from modeling.modules import EMAModel, MLMLoss, get_mask_tokens, residual_sample, combine_factorized_tokens, split_factorized_tokens
from modeling.rqgan import RQModel
from modeling.ar import ResAR
from evaluator import GeneratorEvaluator

from utils.viz_utils import make_viz_reconstructed_stage_two, make_viz_generated_stage_two
from utils.script_utils import get_config, get_save_iteration
from torchinfo import summary


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    work_dir = os.environ.get('WORKSPACE', './runs')

    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    output_dir = os.path.join(work_dir, "outputs", config.experiment.name, config.experiment.run_name)
    # if config.experiment.resume:
    #     ## remember to change the dir after all training
    #     output_dir = os.path.join(work_dir, "outputs", config.experiment.name, f"{config.model.vq_model.num_quantizers}level", "current")   # do not set time if experiment may be resumed
    # else:
    #     output_dir = os.path.join(work_dir, "outputs", config.experiment.name, f"{config.model.vq_model.num_quantizers}level", cur_time)
    config.experiment.logging_dir = str(Path(output_dir) / "logs")

    if config.experiment.logger not in ("wandb", "tensorboard"):
        raise ValueError(f"{config.experiment.logger} is not supported. Please choose `wandb` or `tensorboard`.")

    project_config = ProjectConfiguration(
        project_dir=output_dir,
        automatic_checkpoint_naming=True,
        total_limit=2,
        iteration=get_save_iteration(output_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.experiment.logger,
        split_batches=False,
        project_config=project_config,
    )


    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.per_gpu_batch_size
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(name="ResAR", log_level="INFO", output_dir=config.experiment.logging_dir)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        #### save the run_id for resume
        if config.experiment.logger == "wandb":
            import uuid
            run_id_file = os.path.join(output_dir, "wandb_run_id.txt")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r") as f:
                    run_id = f.read().strip()
            else:
                run_id = str(uuid.uuid4())
                with open(run_id_file, "w") as f:
                    f.write(run_id)
            
            accelerator.init_trackers(
                project_name=config.experiment.name,
                config=OmegaConf.to_container(config, resolve=True),  # Convert OmegaConf to a dictionary
                init_kwargs={
                    "wandb": {
                        "id": run_id,
                        "resume": "allow",
                        "name": config.experiment.run_name,
                    }
                }
                )
        elif config.experiment.logger == "tensorboard":
            accelerator.init_trackers(
                project_name=config.experiment.name,
                # config=OmegaConf.to_container(config, resolve=True),  # Convert OmegaConf to a dictionary
            )
        
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
    
    if accelerator.is_local_main_process:
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}", main_process_only=False)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Creating model and optimizer")

    vqgan_model = RQModel(config.model.vq_model)
    config.model.vq_model.codebook_size = vqgan_model.codebook_size
    vqgan_model.load_pretrained(config.experiment.vqgan_checkpoint)
    vqgan_model.eval()

    
    ar_model = ResAR(config)
    model_cls = ResAR
    cur_config = ar_model._cur_config
    cur_config.model.mlm_model.mask_token = ar_model._cur_model.mask_token
    stage = cur_config.model.mlm_model.stage
    codebook_size = cur_config.model.vq_model.codebook_size[stage]
    splits = cur_config.model.mlm_model.codebook_splits
    bits = cur_config.model.vq_model.token_size
    variants = cur_config.model.vq_model.variants[stage]
    
    if accelerator.is_local_main_process:
        logger.info(f"Current config:\n{OmegaConf.to_yaml(cur_config)}", main_process_only=False)
    logger.info(f"Masktoken: {cur_config.model.mlm_model.mask_token}")
    logger.info(f"Training mlm model with stage {stage} in {ar_model.num_stages} stages.")
    
    # Create the EMA model
    if config.training.use_ema:
        ema_model = EMAModel(ar_model.parameters(), decay=0.999, model_cls=model_cls,
            config=config,
        )

        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"), model_cls=model_cls,
                config=config,
            )
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    loss_config = config.losses.mlm
    loss_module = MLMLoss(loss_config.label_smoothing, loss_config.sum_splits)

    # Print Model:
    vqgan_summary_str = summary(
        vqgan_model,
        input_size=(1,3,256,256),
        depth=5,
        col_width=15,
        col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"),
        verbose=0
    )
    patch_size = int(config.dataset.preprocessing.resolution // (2**(config.model.vq_model.num_resolutions - 1)))
    mlm_summary_str = summary(
        ar_model,
        input_data=[torch.randint(0, cur_config.model.mlm_model.mask_token, (1, patch_size * patch_size, cur_config.model.mlm_model.codebook_splits)), torch.randn(1, config.model.vq_model.token_size, patch_size, patch_size), torch.ones(1, dtype=int)],
        depth=7,
        col_width=15,
        col_names=("input_size", "output_size", "num_params", "params_percent",),
        verbose=0
    )
    logger.info(vqgan_summary_str)
    logger.info(mlm_summary_str)

    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    optimizer = optimizer_cls(
        list(ar_model.parameters()),
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleImagenet(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
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
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    num_batches = train_dataloader.num_batches


    ##################################
    # EVALUATION STUFF.              #
    ##################################
    evaluator = GeneratorEvaluator(
        device=accelerator.device,
        enable_fid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=False,
    )
    
    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    ar_model, optimizer, lr_scheduler = accelerator.prepare(
        ar_model, optimizer, lr_scheduler)

    if config.training.use_ema:
        ema_model.to(accelerator.device)
    vqgan_model.to(device=accelerator.device)

    if config.training.overfit_batch:
        num_update_steps_per_epoch = config.training.overfit_batch_num
        new_train_dataloader = []
        for i, batch in enumerate(train_dataloader):
            if i >= num_update_steps_per_epoch:
                break
            new_train_dataloader.append(batch)
        train_dataloader = new_train_dataloader
        logger.info(f"Overfitting on {len(train_dataloader)} batch(es).")
    else:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0
    first_epoch = 0

    if config.experiment.resume:
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(output_dir, "checkpoints", "checkpoint*")))
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('_')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:  # len(local_ckpt_list) == 1
                checkpoint_paths = local_ckpt_list
            
            resume_lr_scheduler = config.experiment.get("resume_lr_scheduler", True)
            dont_resume_optimizer = config.experiment.get("dont_resume_optimizer", False)
            if not resume_lr_scheduler:
                logger.info("Not resuming the lr scheduler.")
                accelerator._schedulers = []  # very hacky, but we don't want to resume the lr scheduler
            if dont_resume_optimizer:
                logger.info("Not resuming the optimizer.")
                accelerator._optimizers = []  # very hacky, but we don't want to resume the optimizer
                grad_scaler = accelerator.scaler
                accelerator.scaler = None

            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            if not resume_lr_scheduler:
                accelerator._schedulers = [lr_scheduler]
            if dont_resume_optimizer:
                accelerator._optimizers = [optimizer]
                accelerator.scaler = grad_scaler

            first_epoch = global_step // num_update_steps_per_epoch
        elif len(local_ckpt_list) > 1:
            raise ValueError("There should only be one checkpoint folder.")


    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    loss_average_m = AverageMeter()
    accuracy_m = AverageMeter()
    masked_accuracy_m = AverageMeter()
    end = time.time()
    last_eval_at_step = -1
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for current_epoch in range(first_epoch, num_train_epochs):
        ar_model.train()
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        for batch in train_dataloader:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            class_tokens = batch["class_id"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            with torch.no_grad():
                context, encoder_dict = vqgan_model.encode(images, num_levels=cur_config.model.mlm_model.stage)
                input_tokens = encoder_dict["min_encoding_indices"][cur_config.model.mlm_model.stage]
                input_tokens = input_tokens.reshape(input_tokens.shape[0], -1)
            fnames = batch["__key__"]
            data_time_m.update(time.time() - end)

            with accelerator.accumulate([ar_model]):
                input_tokens = split_factorized_tokens(input_tokens, codebook_size=codebook_size, splits=splits, bits=bits, variants=variants)

                masked_tokens, masks = get_mask_tokens(
                    input_tokens,
                    cur_config.model.mlm_model.mask_token,
                    mode=cur_config.model.mlm_model.train_mask_schedule_strategy
                )

                # forward
                drop_label_mask = torch.rand_like(class_tokens, dtype=torch.float) < cur_config.model.mlm_model.class_label_dropout
                logits = ar_model(masked_tokens, context, class_tokens, drop_label_mask)
                maskgit_loss, loss_dict = loss_module(logits, input_tokens, masks)

                # Gather the losses across all processes for logging (if we use distributed training).
                mlm_logs = {}
                for k, v in loss_dict.items():
                    mlm_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(maskgit_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ar_model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(ar_model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            loss_average_m.update(mlm_logs['train/mlm_loss'])
            accuracy_m.update(mlm_logs['train/correct_tokens'])
            masked_accuracy_m.update(mlm_logs['train/masked_correct_tokens'])
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_model.step(ar_model.parameters())

                # wait for both generator and discriminator to settle
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_m.val
                    )
                    logger.info(
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Step: {global_step + 1} "
                        f"Loss: {loss_average_m.avg:0.4f} "
                        f"Accuracy: {accuracy_m.avg:0.4f} "
                        f"Masked Accuracy: {masked_accuracy_m.avg:0.4f} "
                    )
                    logs = {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "time/data_time": data_time_m.val,
                        "time/batch_time": batch_time_m.val,
                        "train/avg_mlm_loss": loss_average_m.avg,
                    }
                    logs.update(mlm_logs)
                    accelerator.log(logs, step=global_step + 1)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                    loss_average_m.reset()
                    accuracy_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(ar_model, output_dir, accelerator, global_step + 1)

                    # Wait for everyone to save their checkpoint
                    accelerator.wait_for_everyone()

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    # Store the model parameters temporarily and load the EMA parameters to perform inference.
                    if config.training.get("use_ema", False):
                        ema_model.store(ar_model.parameters())
                        ema_model.copy_to(ar_model.parameters())

                    context = context[:config.training.num_generated_images]
                    class_tokens = class_tokens[:config.training.num_generated_images]

                    # Generate images
                    generate_images(
                        ar_model,
                        vqgan_model,
                        class_tokens,
                        config,
                        accelerator,
                        global_step+1,
                        output_dir,
                    )

                    # Reconstruct images
                    predicted_tokens = torch.argmax(logits, -1).view(input_tokens.shape)
                    reconstructed_and_predicted_images(
                        vqgan_model,
                        cur_config,
                        context,
                        input_tokens[:config.training.num_generated_images],
                        predicted_tokens[:config.training.num_generated_images],
                        accelerator,
                        global_step+1,
                    )

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(ar_model.parameters())
                
                if (global_step + 1) % config.experiment.eval_loss_every == 0:
                    # or all val images.
                    logger.info(f"Computing losses on the validation set.")

                    if config.training.get("use_ema", False):
                        ema_model.store(ar_model.parameters())
                        ema_model.copy_to(ar_model.parameters())
                    
                    eval_loss_scores = eval_loss(
                        ar_model,
                        vqgan_model,
                        eval_dataloader,
                        loss_module,
                        cur_config,
                    )
                    logger.info(f"EVALUATION Step: {global_step + 1}")
                    logger.info(pprint.pformat(eval_loss_scores))
                    if accelerator.is_main_process:
                        eval_log = {'eval/'+k: v for k, v in eval_loss_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(ar_model.parameters())
                    accelerator.wait_for_everyone()

                # Evaluate reconstruction
                if (global_step + 1) % config.experiment.eval_gen_every == 0:
                    # or all val images.
                    logger.info(f"Computing metrics on the validation set.")

                    if config.training.get("use_ema", False):
                        ema_model.store(ar_model.parameters())
                        ema_model.copy_to(ar_model.parameters())

                    eval_scores = eval_generation(
                        ar_model,
                        vqgan_model,
                        eval_dataloader,
                        evaluator,
                        config,
                    )

                    logger.info(f"EVALUATION Step: {global_step + 1}")
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(ar_model.parameters())
                    accelerator.wait_for_everyone()
                    last_eval_at_step = global_step

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(ar_model, output_dir, accelerator, global_step)
    if global_step - last_eval_at_step > 1000:
        logger.info(f"Computing metrics on the validation set.")

        if config.training.get("use_ema", False):
            ema_model.store(ar_model.parameters())
            ema_model.copy_to(ar_model.parameters())

        eval_scores = eval_generation(
            ar_model,
            vqgan_model,
            eval_dataloader,
            evaluator,
            config,
        )

        logger.info(f"EVALUATION Step: {global_step + 1} ")
        logger.info(pprint.pformat(eval_scores))
        if accelerator.is_main_process:
            eval_log = {'eval/'+k: v for k, v in eval_scores.items()}
            accelerator.log(eval_log, step=global_step + 1)

        accelerator.wait_for_everyone()

    accelerator.end_training()

@torch.no_grad()
def eval_loss(
        ar_model,
        vqgan_model,
        eval_loader,
        loss_module,
        config,
):
    ar_model.eval()
    loss_average_m = AverageMeter()
    accuracy_m = AverageMeter()
    masked_accuracy_m = AverageMeter()

    for batch in tqdm.tqdm(eval_loader):
        images = batch["image"].to(
                ar_model.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        class_tokens = batch["class_id"].to(
                ar_model.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        num_samples = class_tokens.shape[0]
        context, encoder_dict = vqgan_model.encode(images, num_levels=config.model.mlm_model.stage)
        input_tokens = encoder_dict["min_encoding_indices"][config.model.mlm_model.stage]
        input_tokens = input_tokens.reshape(input_tokens.shape[0], -1)
        input_tokens = split_factorized_tokens(input_tokens, codebook_size=config.model.vq_model.codebook_size[config.model.mlm_model.stage], splits=config.model.mlm_model.codebook_splits, bits=config.model.vq_model.token_size, variants=config.model.vq_model.variants[config.model.mlm_model.stage])

        masked_tokens, masks = get_mask_tokens(
            input_tokens,
            config.model.mlm_model.mask_token,
            mode=config.model.mlm_model.train_mask_schedule_strategy
        )

        # forward
        drop_label_mask = torch.rand_like(class_tokens, dtype=torch.float) < config.model.mlm_model.class_label_dropout
        logits = ar_model(masked_tokens, context, class_tokens, drop_label_mask)
        maskgit_loss, loss_dict = loss_module(logits, input_tokens, masks)

        loss_average_m.update(loss_dict['mlm_loss'].item())
        accuracy_m.update(loss_dict['correct_tokens'].item())
        masked_accuracy_m.update(loss_dict['masked_correct_tokens'].item())

    return {
        "loss": loss_average_m.avg,
        "accuracy": accuracy_m.avg,
        "masked_accuracy": masked_accuracy_m.avg,
    }


@torch.no_grad()
def eval_generation(
    ar_model,
    vqgan_model,
    eval_loader,
    evaluator,
    config,
):
    ar_model.eval()
    evaluator.reset_metrics()

    for batch in tqdm.tqdm(eval_loader):
        images = batch["image"].to(
                ar_model.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        class_tokens = batch["class_id"].to(
                ar_model.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        generated_samples, _ = residual_sample(
            ar_model,
            vqgan_model,
            config=config,
            labels=class_tokens,
        )
        
        generated_samples = torch.clamp(generated_samples, 0.0, 1.0)

        evaluator.update(generated_samples)

    return evaluator.result()


@torch.no_grad()
def reconstructed_and_predicted_images(
    vqgan_model: torch.nn.Module,
    config,
    context: torch.Tensor,
    tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    accelerator,
    global_step: int,
):
    """Decode ground-truth and predicted tokens into images.

    Args:
        vqgan_model -> torch.nn.Module: The Stage-I model used for encoding/decoding.
        config: The config file.
        tokens -> torch.Tensor: The ground-truth tokens.
        predicted_tokens -> torch.Tensor: The rpedicted tokens.
        accelerator: The accelerator
        global_step -> int: The current training step.
    """
    logger = get_logger(name="ResAR", log_level="INFO")
    logger.info("Decoding images...")

    codebook_size = config.model.vq_model.codebook_size[config.model.mlm_model.stage]
    codebook_splits = config.model.mlm_model.codebook_splits
    bits = config.model.vq_model.token_size
    variants = config.model.vq_model.variants[config.model.mlm_model.stage]

    tokens = combine_factorized_tokens(tokens, codebook_size, codebook_splits, bits=bits, variants=variants)
    predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits, bits=bits, variants=variants)

    reconstructed = vqgan_model.decode_tokens(tokens, num_level=config.model.mlm_model.stage, context=context)
    predicted = vqgan_model.decode_tokens(predicted_tokens, num_level=config.model.mlm_model.stage, context=context)

    images_wandb, images_tensorboard = make_viz_reconstructed_stage_two(
        reconstructed, predicted
    )

    if config.experiment.logger == "wandb":
        accelerator.get_tracker("wandb").log_images(
            {"Train Decoded": images_wandb}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
        {"Train Decoded": images_tensorboard}, step=global_step
    )


@torch.no_grad()
def generate_images(
    ar_model: torch.nn.Module,
    vqgan_model: torch.nn.Module,
    labels: torch.Tensor,
    config,
    accelerator,
    global_step: int,
    output_dir: Text,
):
    """Generate images with the model.
    This function generates images with the model and saves them to the output directory.
    Args: 
        - model -> torch.nn.Module: The model to use for generating images.
        - vqgan_model -> torch.nn.Module: The VQGAN model to use for decoding tokens to images.
        - config: The configuration dictionary.
        - accelerator: The accelerator object.
        - global_step -> int: The current training step. This is used to create the output filename.
        - output_dir -> Text: The output directory.
    """
    logger = get_logger(name="ResAR", log_level="INFO")
    logger.info("Generating images...")


    generated_samples, _ = residual_sample(
        ar_model,
        vqgan_model,
        config=config,
        labels=labels,
    )
    
    images_wandb, images_tensorboard = make_viz_generated_stage_two(generated_samples)


    if config.experiment.logger == "wandb":
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_wandb]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_tensorboard.unsqueeze(0)}, step=global_step
        )

    # Log locally
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_wandb.save(path)


def save_checkpoint(model, output_dir, accelerator, global_step) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    logger = get_logger(name="ResAR", log_level="INFO")

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    state_dict = accelerator.get_state_dict(model)
    save_path = accelerator.save_state(save_path)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

        if global_step % 100_000 == 0:
            os.makedirs(f"{output_dir}/archive", exist_ok=True)
            import shutil
            shutil.copytree(save_path, f"{output_dir}/archive/checkpoint-{global_step}", dirs_exist_ok=True)
            print(f"Checkpoint {global_step} archived.")

    return save_path


def load_checkpoint(
    checkpoint_path: Path,
    accelerator
) -> int:
    """Load a checkpoint of the model.
    This function loads a checkpoint of the model from the specified path.

    Args:
        - checkpoint_path -> Path: The path to the checkpoint.
        - accelerator: The accelerator object.
    Returns:
        int: The training step of the loaded checkpoint.
    """
    logger = get_logger(name="ResAR", log_level="INFO")
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(
    model: torch.nn.Module,
    accelerator, 
    global_step: int
):
    """Log the norm of the gradients.
    This function logs the norm of the gradients for each parameter in the model.

    Args:
        - model -> torch.nn.Module: The model to log the gradients for.
        - accelerator: The accelerator object.
        - global_step -> int: The current training step. This is used as the logging step.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()