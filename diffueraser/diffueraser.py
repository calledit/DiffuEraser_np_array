import gc
import copy
import cv2
import os
import numpy as np
import torch
from functools import partial
import torchvision
from einops import repeat
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig

from libs.unet_motion_model import MotionAdapter, UNetMotionModel
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        4,
        0.0,
    ],
}

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames

def read_mask(validation_mask, fps, n_total_frames, img_size, mask_dilation_iter, frames):
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        print("Error: Could not open mask video.")
        exit()
    mask_fps = cap.get(cv2.CAP_PROP_FPS)
    if mask_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    masks = []
    masked_images = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        if(idx >= n_total_frames):
            break
        mask = Image.fromarray(frame[...,::-1]).convert('L')
        if mask.size != img_size:
            mask = mask.resize(img_size, Image.NEAREST)
        mask = np.asarray(mask)
        m = np.array(mask > 0).astype(np.uint8)
        m = cv2.erode(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1)
        m = cv2.dilate(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=mask_dilation_iter)

        mask = Image.fromarray(m * 255)
        masks.append(mask)

        masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        masked_images.append(masked_image)

        idx += 1
    cap.release()

    return masks, masked_images

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
    if priori_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    os.remove(priori) # remove priori 

    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec', end_pts=video_length) # RGB
    fps = info['video_fps']
    n_total_frames = int(video_length * fps)
    n_clip = int(np.ceil(n_total_frames/nframes))

    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    max_size = max(frames[0].size)
    if(max_size<256):
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if(max_size>4096):
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")
    if max_size>max_img_size:
        ratio = max_size/max_img_size
        ratio_size = (int(frames[0].size[0]/ratio),int(frames[0].size[1]/ratio))
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    elif (frames[0].size[0]%8==0) and (frames[0].size[1]%8==0):
        img_size = frames[0].size
        resize_flag=False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    return frames, fps, img_size, n_clip, n_total_frames

def _arrays_to_pil(frames_np):
    """List[np.ndarray HxWx{1,3}] -> List[PIL.Image], RGB if 3-ch."""
    pil = []
    for a in frames_np:
        a = a.astype(np.uint8)
        if a.ndim == 2:
            pil.append(Image.fromarray(a, mode="L").convert("RGB"))
        elif a.ndim == 3 and a.shape[2] == 3:
            pil.append(Image.fromarray(a, mode="RGB"))
        elif a.ndim == 3 and a.shape[2] == 1:
            pil.append(Image.fromarray(a.squeeze(-1), mode="L").convert("RGB"))
        else:
            raise ValueError("Frames must be HxW, HxWx1 or HxWx3 uint8.")
    return pil

def _prepare_masks_from_arrays(mask_arrays, target_size, dilation_iter=4):
    """
    mask_arrays: List[np.ndarray] (len==1 or == #frames), any of: HxW, HxWx1, HxWx3; uint8/bool/float
    Returns: (masks_pil, masked_images_pil) where each mask is binary (0/255) after erode+ dilate,
             and masked_images = frame*(1-mask) will be created elsewhere (this function returns only masks).
    """
    def _to_gray_uint8(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8)).convert('L')
        else:
            if arr.dtype != np.uint8:
                arr = (arr > 0.0).astype(np.uint8) * 255
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            img = Image.fromarray(arr.astype(np.uint8))
        return img

    masks_pil = []
    if len(mask_arrays) == 0:
        raise ValueError("mask_arrays must be a non-empty list")
    for arr in mask_arrays:
        m = _to_gray_uint8(arr)
        if m.size != target_size:
            m = m.resize(target_size, Image.NEAREST)
        m_np = np.array(m)
        m_bin = (m_np > 0).astype(np.uint8)
        # gentle erode + dialate like original
        m_bin = cv2.erode(m_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        if dilation_iter > 0:
            m_bin = cv2.dilate(m_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=dilation_iter)
        masks_pil.append(Image.fromarray(m_bin * 255))
    return masks_pil

def _fit_size_multiple_of_8_and_limit(frames_pil, max_img_size):
    """
    Ensures frames are <= max_img_size on the longer side and multiples of 8.
    Returns (resized_frames, img_size).
    """
    if len(frames_pil) == 0:
        raise ValueError("frames list is empty")
    w0, h0 = frames_pil[0].size
    max_side = max(w0, h0)
    if max_side < 256:
        raise ValueError("Resolution must be at least 256x256.")
    if max_side > 4096:
        raise ValueError("Resolution must be <= 4096x4096.")
    if max_side > max_img_size:
        ratio = max_side / max_img_size
        w = int(w0 / ratio)
        h = int(h0 / ratio)
    else:
        w, h = w0, h0
    w = w - (w % 8)
    h = h - (h % 8)
    if (w, h) != (w0, h0):
        frames_pil = [im.resize((w, h)) for im in frames_pil]
    return frames_pil, (w, h)


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="Normal CFG 4-Step", mode="sd15", loaded=None):
        self.device = device

        ## load model
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, 
                subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    subfolder="tokenizer",
                    use_fast=False,
                )
        text_encoder_cls = import_model_class_from_model_name_or_path(base_model_path,revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
                base_model_path, subfolder="text_encoder"
            )
        self.brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet")
        self.unet_main = UNetMotionModel.from_pretrained(
            diffueraser_path, subfolder="unet_main",
        )

        ## set pipeline
        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            base_model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet
        ).to(self.device, torch.float16)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        self.noise_scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            self.pipeline.load_lora_weights(
                "wangfuyun/PCM_Weights", weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]
        self.guidance_scale = 0

    def forward(
        self,
        video_frames_np,          # List[np.ndarray] RGB uint8 HxWx3 (or HxW/HxWx1 -> will be converted to RGB)
        mask_frames_np,           # List[np.ndarray] (len==1 or == len(video)), any of HxW/HxWx1/HxWx3, uint8/bool/float
        priori_frames_np,         # List[np.ndarray] same size count as video (or will be clipped)
        *,
        max_img_size=1280,
        mask_dilation_iter=4,
        nframes=22,
        seed=None,
        guidance_scale=None,
        blended=True,
        progress=None,
    ):
        """
        Returns:
            comp_frames: List[np.ndarray] (uint8, H, W, 3) composed frames.
        """
        if not (isinstance(video_frames_np, (list, tuple)) and len(video_frames_np) > 0):
            raise ValueError("video_frames_np must be a non-empty list.")
        if not (isinstance(mask_frames_np, (list, tuple)) and len(mask_frames_np) > 0):
            raise ValueError("mask_frames_np must be a non-empty list.")
        if not (isinstance(priori_frames_np, (list, tuple)) and len(priori_frames_np) > 0):
            raise ValueError("priori_frames_np must be a non-empty list.")

        if (max_img_size < 256 or max_img_size > 1920):
            raise ValueError("max_img_size must be in [256, 1920].")
        
        if progress is not None:
            progress(52, "diffueraser preparing frames")
        
        # ------------- Convert inputs to PIL and unify sizes -------------
        frames = _arrays_to_pil(video_frames_np)
        prioris = _arrays_to_pil(priori_frames_np)

        # Fit video frames to multiple-of-8 and <= max_img_size
        frames, img_size = _fit_size_multiple_of_8_and_limit(frames, max_img_size)
        # Resize prioris to the exact same size
        prioris = [p.resize(img_size) if p.size != img_size else p for p in prioris]

        # Prepare masks (broadcast if length==1; then clip)
        if len(mask_frames_np) == 1:
            mask_frames = mask_frames_np * len(frames)
        else:
            mask_frames = mask_frames_np

        # Clip all to same effective length
        n_total_frames = min(len(frames), len(mask_frames), len(prioris))
        if n_total_frames < 22:
            raise ValueError("Effective video duration too short; need at least 22 frames.")

        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]
        mask_frames = mask_frames[:n_total_frames]

        # Build binary masks (0/255)
        validation_masks_input = _prepare_masks_from_arrays(mask_frames, img_size, dilation_iter=mask_dilation_iter)

        # Make masked images: frame * (1 - mask)
        validation_images_input = []
        for i in range(n_total_frames):
            m = np.array(validation_masks_input[i])  # HxW uint8 (0/255)
            mask01 = (m.astype(np.float32) / 255.0)[..., None]
            img_np = np.array(frames[i]).astype(np.float32)
            masked_np = img_np * (1.0 - mask01)
            validation_images_input.append(Image.fromarray(masked_np.astype(np.uint8)))

        # Shallow copies used later for composition
        validation_masks_input_ori = [m.copy() for m in validation_masks_input]
        resized_frames_ori = [f.copy() for f in frames]

        # ------------- DiffuEraser inference -------------
        print("DiffuEraser inference...")
        validation_prompt = ""
        guidance_scale_final = self.guidance_scale if guidance_scale is None else guidance_scale

        # Determine chunking like original read_video logic
        # emulate fps-driven segmentation using nframes cadence
        n_clip = int(np.ceil(n_total_frames / nframes))

        # Random generator
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        
        if progress is not None:
            progress(53, "diffueraser generating noice for sequence")
        # ---- random noise for the whole sequence ----
        tar_width, tar_height = img_size
        shape = (nframes, 4, tar_height // 8, tar_width // 8)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet_main is not None:
            prompt_embeds_dtype = self.unet_main.dtype
        else:
            prompt_embeds_dtype = torch.float16

        noise_pre = randn_tensor(shape, device=torch.device(self.device), dtype=prompt_embeds_dtype, generator=generator)
        real_video_length = n_total_frames
        noise = repeat(noise_pre, "t c h w -> (repeat t) c h w", repeat=n_clip)[:real_video_length, ...]
        if progress is not None:
            progress(55, "diffueraser preprocessing frames")
        # ---- prepare priori -> VAE latents ----
        images_preprocessed = []
        for image in prioris:
            image = self.image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
            image = image.to(device=torch.device(self.device), dtype=torch.float16)
            images_preprocessed.append(image)
        pixel_values = torch.cat(images_preprocessed)

        with torch.no_grad():
            pixel_values = pixel_values.to(dtype=torch.float16)
            latents = []
            bs = 4
            for i in range(0, pixel_values.shape[0], bs):
                latents.append(self.vae.encode(pixel_values[i : i + bs]).latent_dist.sample())
            latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        torch.cuda.empty_cache()

        timesteps = torch.tensor([0], device=self.device).long()
        
        
        if progress is not None:
            progress(58, "diffueraser doing sampled pre-inferance on subset of frames")
        # ---- Pre-inference (sampling some frames if long) ----
        if n_total_frames > nframes * 2:
            step = n_total_frames / nframes
            sample_index = [int(i * step) for i in range(nframes)]
            sample_index = sample_index[:22]

            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre, timesteps)
            latents_pre = noisy_latents_pre

            with torch.no_grad():
                latents_pre_out = self.pipeline(
                    num_frames=nframes,
                    prompt=validation_prompt,
                    images=validation_images_input_pre,
                    masks=validation_masks_input_pre,
                    num_inference_steps=self.num_inference_steps,
                    generator=generator,
                    guidance_scale=guidance_scale_final,
                    latents=latents_pre,
                    progress = partial(progress, 60, "diffueraser doing sampled pre-inferance on subset of frames", 70)
                ).latents
            torch.cuda.empty_cache()

            # Decode and replace corresponding input frames & masks
            def decode_latents(latents_x, weight_dtype=torch.float16):
                latents_x = 1 / self.vae.config.scaling_factor * latents_x
                video = []
                for t in range(latents_x.shape[0]):
                    video.append(self.vae.decode(latents_x[t:t+1, ...].to(weight_dtype)).sample)
                video = torch.concat(video, dim=0).float()
                return video

            with torch.no_grad():
                video_tensor_temp = decode_latents(latents_pre_out, weight_dtype=torch.float16)
                images_pre_out = self.image_processor.postprocess(video_tensor_temp, output_type="pil")

            black_image = Image.new('L', validation_masks_input[0].size, color=0)
            for i, idx in enumerate(sample_index):
                latents[idx] = latents_pre_out[i]
                validation_masks_input[idx] = black_image
                validation_images_input[idx] = images_pre_out[i]
                frames[idx] = images_pre_out[i]
        else:
            latents_pre_out = None
            sample_index = None

        gc.collect()
        torch.cuda.empty_cache()
        
        if progress is not None:
            progress(70, "diffueraser doing inferance on all frames")
            
        # ---- Frame-by-frame inference ----
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        latents = noisy_latents
        with torch.no_grad():
            images = self.pipeline(
                num_frames=nframes,
                prompt=validation_prompt,
                images=validation_images_input,
                masks=validation_masks_input,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents,
                progress = partial(progress, 70, "diffueraser doing full inferance", 90)
            ).frames
        images = images[:real_video_length]

        gc.collect()
        torch.cuda.empty_cache()

        # ------------- Compose & return frames (np.uint8) -------------
        binary_masks = validation_masks_input_ori
        if blended:
            mask_blurreds = []
            for i in range(len(binary_masks)):
                m = np.array(binary_masks[i])
                mask_blurred = cv2.GaussianBlur(m, (21, 21), 0) / 255.0
                binary_mask = 1.0 - (1.0 - (m / 255.0)) * (1.0 - mask_blurred)
                mask_blurreds.append(Image.fromarray((binary_mask * 255).astype(np.uint8)))
            binary_masks = mask_blurreds

        comp_frames = []
        for i in range(len(images)):
            mask = (np.expand_dims(np.array(binary_masks[i]), 2).repeat(3, axis=2).astype(np.float32)) / 255.0
            img = (np.array(images[i]).astype(np.uint8) * mask +
                   np.array(resized_frames_ori[i]).astype(np.uint8) * (1.0 - mask)).astype(np.uint8)
            comp_frames.append(img)  # np.uint8 HxWx3

        torch.cuda.empty_cache()
        return comp_frames
            



