from .nodes import *

NODE_CLASS_MAPPINGS = {
    "ImageQuilting_Bmad": ImageQuilting,
    "LatentQuilting_Bmad": LatentQuilting,
    "ImageQuiltingSeamlessMB_Bmad": ImageMakeSeamlessMB,
    "LatentQuiltingSeamlessMB_Bmad": LatentMakeSeamlessMB,
    "ImageQuiltingSeamlessSB_Bmad": ImageMakeSeamlessSB,
    "LatentQuiltingSeamlessSB_Bmad": LatentMakeSeamlessSB,
    "GuessQuiltingBlockSize_Bmad": GuessNiceBlockSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageQuilting_Bmad": "Image Quilting",
    "LatentQuilting_Bmad": "Latent Quilting",
    "ImageQuiltingSeamlessMB_Bmad": "Image Seamless Quilting MP",
    "ImageQuiltingSeamlessSB_Bmad": "Image Seamless Quilting SP",
    "LatentQuiltingSeamlessMB_Bmad": "Latent Seamless Quilting MP",
    "LatentQuiltingSeamlessSB_Bmad": "Latent Seamless Quilting SP",
    "GuessQuiltingBlockSize_Bmad": "Guess Quilting Block Size",
}
