from .nodes import *

NODE_CLASS_MAPPINGS = {
    "ImageQuilting_Bmad": ImageQuilting,
    "LatentQuilting_Bmad": LatentQuilting,
    "ImageQuiltingSeamlessMB_Bmad": ImageMakeSeamlessMB,
    #"ImageQuiltingSeamlessSB_Bmad": ImageMakeSeamlessSB,
    #"GuessQuiltingBlockSize": GuessBlockSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageQuilting_Bmad": "Image Quilting",
    "LatentQuilting_Bmad": "Latent Quilting",
    "ImageQuiltingSeamlessMB_Bmad": "Image Seamless Quilting MP",
    #"ImageQuiltingSeamlessSB_Bmad": "Image Seamless Quilting SP",
    # "GuessQuiltingBlockSize": "Guess BlockSize (Quilting)"
}