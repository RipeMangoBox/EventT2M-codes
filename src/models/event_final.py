import os
import os.path as osp
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MeanMetric, MetricCollection, MinMetric
import lightning.pytorch as L

from diffusers import UniPCMultistepScheduler, DDPMScheduler

from .metrics import TM2TMetrics, MMMetrics

from .utils.utils import occumpy_mem, replace_annotation_with_null, lengths_to_mask, replace_annotation_with_null_2
from .nets.ema import EMAModel

from src.data.humanml.scripts.motion_process import recover_from_ric

class EventMotionGeneration(L.LightningModule):
    def __init__(self,
                 text_encoder,
                denoiser,
                noise_scheduler,
                sample_scheduler,
                text_replace_prob,
                guidance_scale,
                dataset_name, # for evaluate model
                evaluator,
                optimizer, # for optimize model
                ema=False,
                lr_scheduler=None,
                debug=False,
                ocpm=False,
                step_num=10,
                 **kwargs):
        super(EventMotionGeneration, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['text_encoder', 'denoiser'])
        self.text_encoder = text_encoder
        self.denoiser = denoiser

        # self.best_fid = float('inf')

        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
            self.sample_scheduler.set_timesteps(1000)
        else:
            self.sample_scheduler = sample_scheduler
            self.sample_scheduler.set_timesteps(step_num)

        self.configure_evaluator_and_metrics(dataset_name, evaluator)

        if ema.use_ema:
            self.ema_denoiser = EMAModel(self.denoiser, decay=ema.ema_decay)
            self.ema_denoiser.set(self.denoiser)
        else:
            self.ema_denoiser = None

        self.is_mm_metric = False # only used during mm testing

        num_params = sum([x.numel() for x in self.denoiser.parameters() if x.requires_grad])
        print("number of trainable parameters: %.3fM" % (num_params / 1000_000))

        self.global_index = 0


    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.denoiser.parameters())
        if self.hparams.lr_scheduler is not None:
            lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer

    def configure_evaluator_and_metrics(self, dataset_name, evaluator):
        self.train_metrics = MetricCollection({"loss": MeanMetric()})

        from src.models.evaluator.T2M import T2MEvaluator
        self.t2m_evaluator = T2MEvaluator(dataset_name, evaluator.T2M_dir)
        self.t2m_metrics = TM2TMetrics(diversity_times=300 if dataset_name == "hml3d" else 100,
                               dist_sync_on_step=True)
        self.t2m_mm_metric = MMMetrics(mm_num_times=10, dist_sync_on_step=True)

    def on_train_start(self):
        if self.hparams.ocpm:
            occumpy_mem(self.device.index)

    def encode_decomposed_with_padding(self, decomposed, max_len=11):
        B = len(decomposed)

        flat_caps = [cap for caps in decomposed for cap in caps]
        seq_lens  = [len(caps) for caps in decomposed]

        with torch.no_grad():
            flat_emb = self.text_encoder(flat_caps, self.device)
        clip_dim = flat_emb['text_emb'].size(-1)

        emb = {}

        emb['text_emb'] = flat_emb['text_emb'].new_zeros(B, max_len, clip_dim)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)

        pos = 0
        for b, L in enumerate(seq_lens):
            if L == 0:
                continue
            emb['text_emb'][b, :L]  = flat_emb['text_emb'][pos:pos+L]
            mask[b, :L] = True
            pos += L

        return emb, mask

    def _step_network(self, batch, batch_idx):
        motion, length, text = batch["motion"], batch["motion_len"], batch["text"]
        
        decomposed = [[d['caption'] for d in t[1]] for t in text]
        text = [t[0] for t in text]

        text = replace_annotation_with_null(text, self.hparams.text_replace_prob)
        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device)

        decomposed = replace_annotation_with_null_2(decomposed, self.hparams.text_replace_prob)

        decomposed_embed, decomposed_mask = self.encode_decomposed_with_padding(
            decomposed, max_len=11
        )

        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                 (motion.size(0),), device=motion.device).long()
        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(motion, device=motion.device)
        x_t = self.noise_scheduler.add_noise(motion, noise, timestep)

        output = self.denoiser(x_t, padding_mask, timestep, text_embed, decomposed_embed, decomposed_mask)

        # calculate loss
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = motion
        else:
            raise ValueError(f"{prediction_type} not supported!")

        loss = F.mse_loss(output, target, reduction='none')[padding_mask].mean()

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        losses = self._step_network(batch, batch_idx)
        self.log(f"train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=False)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_denoiser is not None:
            if self.global_step <= self.hparams.ema.ema_start:
                self.ema_denoiser.set(self.denoiser)
            else:
                self.ema_denoiser.update(self.denoiser)

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.save_every_n_epochs == 0:
            self.trainer.save_checkpoint(os.path.join(self.hparams.ckpt_path, f"epoch-{self.current_epoch}.ckpt"))

    def on_validation_start(self) -> None:
        if self.hparams.debug:
            self.trainer.save_checkpoint(os.path.join(self.hparams.ckpt_path, f"epoch-{self.current_epoch}.ckpt"))

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return
        self.evaluate(batch, split='val', batch_idx=batch_idx)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.t2m_metrics.reset()
            return

        results = {}
        metric_output = self.t2m_metrics.compute(sanity_flag=False)
        results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})
        self.t2m_metrics.reset()
        results.update({"epoch": self.trainer.current_epoch, "step": self.global_step,})
        self.log_dict(results, sync_dist=True)
        
        # current_fid = float('inf')
        # if not self.trainer.sanity_checking and self.global_rank == 0:
        #     current_fid = results.get("Metrics/FID", None)
        # if current_fid < self.best_fid:
        #     self.best_fid = current_fid
        #     save_path = os.path.join(self.hparams.ckpt_path, f"best_fid_epoch-{self.current_epoch}.ckpt")
        #     self.trainer.save_checkpoint(save_path)
        #     print(f"Best FID updated: {self.best_fid:.4f}, checkpoint saved to {save_path}")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, split="test", batch_idx=batch_idx)

    def on_test_epoch_end(self):
        results = {}
        if self.is_mm_metric:
            results.update(self.t2m_mm_metric.compute(sanity_flag=self.trainer.sanity_checking))
            self.t2m_mm_metric.reset()
        else:
            metric_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})
            self.t2m_metrics.reset()
        if self.trainer.sanity_checking is False:
            self.log_dict(results, sync_dist=True, rank_zero_only=True)


    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        remove_keys = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                remove_keys.append(k)
        for k in remove_keys:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        keys_list = list(checkpoint['state_dict'].keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    def evaluate(self, batch, split='val', batch_idx=0):
        motion, length, text = batch["motion"], batch["motion_len"], batch["text"]

        decomposed = [[d['caption'] for d in t[1]] for t in text]
        text = [t[0] for t in text]

        word_embs, pos_ohot, text_len = batch["word_embs"], batch["pos_ohot"], batch["text_len"]
        padding_mask = lengths_to_mask(length, self.device)

        self.is_mm_metric = split == 'test' and self.trainer.test_dataloaders.dataset.is_mm

        if split == 'test':
            dataset = self.trainer.test_dataloaders.dataset
        else:
            dataset = self.trainer.val_dataloaders.dataset

        if self.is_mm_metric:
            pred_motion = []
            mm_repeats = 30
            for _ in range(mm_repeats):
                pred_motion.append(self.sample_motion(motion, length, text, decomposed))

            pred_motion = torch.cat(pred_motion, dim=0)
            length = length.repeat([mm_repeats])
            # pred_motion = dataset.to_hml3d_format(pred_motion)
            pred_motion_unnorm = dataset.inv_transform(pred_motion)
            # pred_motion_unnorm[~padding_mask] = 0

            t2m_motion_gen_emb = self.t2m_evaluator.extract_motion_embedding(pred_motion_unnorm, length)
            bz = len(batch["motion"])
            self.t2m_mm_metric.update(t2m_motion_gen_emb.view(mm_repeats, bz, -1).permute([1, 0, 2]))
        else:
            gt_motion_unnorm = dataset.inv_transform(motion)
            # gt_motion_unnorm[~padding_mask] = 0
            pred_motion = self.sample_motion(motion, length, text, decomposed)
            pred_motion_unnorm = dataset.inv_transform(pred_motion)
            # pred_motion_unnorm[~padding_mask] = 0

            t2m_text_emb = self.t2m_evaluator.extract_text_embedding(word_embs, pos_ohot, text_len, self.device)
            t2m_motion_gen_emb = self.t2m_evaluator.extract_motion_embedding(pred_motion_unnorm, length)
            t2m_motion_gt_emb = self.t2m_evaluator.extract_motion_embedding(gt_motion_unnorm, length)
            self.t2m_metrics.update(t2m_text_emb, t2m_motion_gen_emb, t2m_motion_gt_emb, length)

    @torch.no_grad()
    def sample_motion(self, gt_motion, length, text, decomposed):
        B, L, D = gt_motion.shape
        repeated_text = text.copy()
        repeated_text.extend([""] * B)
        text_embed = self.text_encoder(repeated_text, self.device)

        repeated_decomposed = decomposed.copy()
        repeated_decomposed.extend([[""] * 11] * B)

        decomposed_embed, decomposed_mask = self.encode_decomposed_with_padding(
            repeated_decomposed, max_len=11
        )

        time_steps = self.sample_scheduler.timesteps.to(self.device)
        pred_motion = torch.randn_like(gt_motion, device=self.device) * self.sample_scheduler.init_noise_sigma
        padding_mask = lengths_to_mask(length, self.device)

        if self.ema_denoiser is not None:
            denoiser = self.ema_denoiser.model
        else:
            denoiser = self.denoiser

        prediction_type = self.noise_scheduler.config.prediction_type
        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        for i, t in enumerate(time_steps):
            output = denoiser(pred_motion.repeat([2, 1, 1]), padding_mask.repeat([2, 1,]),
                                       t.repeat([2 * B]), text_embed, decomposed_embed, decomposed_mask)

            if prediction_type == "epsilon":
                cond_eps, uncond_eps = output.chunk(2)
            elif prediction_type == "sample":
                cond_x0, uncond_x0 = output.chunk(2)
                cond_eps, uncond_eps = self.obtain_eps_when_predicting_x_0(cond_x0, uncond_x0, t, pred_motion)
            else:
                raise ValueError(f"{prediction_type} not supported!")
            # pred_noise = (1 + self.hparams.guidance_scale) * cond_eps - self.hparams.guidance_scale * uncond_eps
            pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
            if isinstance(self.sample_scheduler, UniPCMultistepScheduler) or isinstance(self.sample_scheduler, DDPMScheduler):
                pred_motion = self.sample_scheduler.step(pred_noise, t, pred_motion).prev_sample.float()
            else:
                pred_motion = self.sample_scheduler.step(pred_noise, t, pred_motion, use_clipped_model_output=False).prev_sample.float()
            pred_motion[~padding_mask] = 0

        return pred_motion

    def obtain_eps_when_predicting_x_0(self, cond_x0, uncond_x0, timestep, x_t):
        scheduler = self.sample_scheduler
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        cond_eps = (x_t - alpha_prod_t ** 0.5 * cond_x0) / beta_prod_t ** 0.5
        uncond_eps = (x_t - alpha_prod_t ** 0.5 * uncond_x0) / beta_prod_t ** 0.5
        return cond_eps, uncond_eps
