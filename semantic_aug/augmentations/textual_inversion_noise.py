from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers_ import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import logging
from PIL import Image, ImageOps

from typing import Any, Tuple, Callable
from torch import autocast
from scipy.ndimage import maximum_filter

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."


def load_embeddings(embed_path: str,
                    model_path: str = "CompVis/stable-diffusion-v1-4"):

    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder")

    for token, token_embedding in torch.load(
            embed_path, map_location="cpu").items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
    
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = \
            token_embedding.to(embeddings.weight.dtype)

    return tokenizer, text_encoder.to('cuda')


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


class TextualInversionPer(GenerativeAugmentation):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, embed_path: str, 
                 model_path: str = "CompVis/stable-diffusion-v1-4",
                 prompt: str = "a photo of a {name}",
                 format_name: Callable = format_name,
                 strength: float = 0.5, 
                 guidance_scale: float = 7.5,
                 std: float = None,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 erasure_ckpt_path: str = None,
                 disable_safety_checker: bool = True,
                 **kwargs):

        super(TextualInversionPer, self).__init__()

        if TextualInversionPer.pipe is None:

            PipelineClass = (StableDiffusionImg2ImgPipeline 
                             if strength < 1.0 else
                             StableDiffusionPipeline)

            tokenizer, text_encoder = load_embeddings(
                embed_path, model_path=model_path)

            TextualInversionPer.pipe = PipelineClass.from_pretrained(
                model_path,
                variant="fp16", 
                torch_dtype=torch.float16
            ).to('cuda')

            self.pipe.tokenizer = tokenizer
            self.pipe.text_encoder = text_encoder

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.format_name = format_name

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_ckpt_path = erasure_ckpt_path
        self.erasure_word_name = None

        self.means = [-0.10514619201421738,
                    -0.10475793480873108,
                    -0.10457515716552734,
                    -0.10383006930351257,
                    -0.10434737056493759,
                    -0.10644923895597458,
                    -0.10559606552124023,
                    -0.1044950857758522,
                    -0.10507697612047195,
                    -0.10446768999099731]
        self.stds = [0.9949540495872498,
                    1.0103657245635986,
                    1.0084270238876343,
                    0.9984888434410095,
                    0.996153712272644,
                    1.001267910003662,
                    0.9912418723106384,
                    0.9947549104690552,
                    1.0044190883636475,
                    1.0062288045883179]

    def forward(self, image: Image.Image, label: int, 
                metadata: dict,
                ) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)
        name = self.format_name(metadata.get("label", ""))
        prompt = self.prompt.format(name=name)
        mean = self.means[label]
        std = self.stds[label]

        if self.strength < 1.0:  # use image as guidance
            kwargs = dict(
                image=canvas,
                prompt=[prompt], 
                strength=self.strength, 
                guidance_scale=self.guidance_scale,
                mean=mean,
                std=std,
            )
        else:
            kwargs = dict(
                prompt=[prompt], 
                guidance_scale=self.guidance_scale,
                mean=mean,
                std=std,
            )

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None 
                and outputs.nsfw_content_detected[0]
            )

        canvas = outputs.images[0].resize(
            image.size, Image.BILINEAR)

        return canvas, label