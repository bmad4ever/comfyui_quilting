# comfyui_quilting
Image and latent quilting nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).


### Image quilting example workflow
![image quilting workflow](workflows/image_quilting.png)


### Latent quilting example workflow
![latent quilting workflow](workflows/latent_quilting.png)


## Arguments

### scale
The output will have the source dimensions scaled by this amount. 

###  block_size
The size of the blocks is given in pixels for images; for latent images, use the number of pixels divided by 8 instead.

All image nodes allow for the block size to be defined automatically by setting **block_size** to values within the range **[-1, 2]**. The meanings of these values are as follows:

* 0 to 2: Uses the same logic as the **Guess Nice Block Size** node with the option **simple_and_fast** disabled.
* -1: Enables **simple_and_fast** option, which uses a quick estimation.

When a batch of images is provided, a separate block size guess is computed for each image individually.

Note that the guessed block sizes are recalculated each time and are not stored for future executions. If caching is important, you can use the **Guess Nice Block Size** node instead. However, be aware that this node will not calculate individual block sizes for each image in a batch; it will only inspect the first image.

### overlap
Given as a percentage, indicates the portion of the block that overlaps with the next block when stitching.

### tolerance
When stitching, tolerance defines the margin of acceptable patches.

- Lower tolerance: Selects sets of patches that better fit their neighborhood but may result in too much repetition.
- Higher tolerance: Avoids repetition but may generate some not-so-seamless transitions between patches.

A tolerance of 1 allows for the selection of patches with an error value up to twice the minimum error, where the minimum error is defined as the error of the most seamless patch. The selection among these patches is random.

### parallelization_lvl (Parallelization Level)
Controls the level of parallel processing during the generation.

* 0: Runs the algorithm sequentially (no parallelization).

* 1: Segments the generation into 4 quadrants, which are generated in parallel.

* 2 or more: Generally not recommended for most use cases as it can be slower than using a lower parallelization level. Consider this setting for larger generations and patches, and also account for the available CPU cores.

    When using a parallelization level of 2 or more:

    * Each quadrant's process will use a number of subprocesses equal to the parallelization level to generate that quadrant. 
    * The generation is done via cascading rows, where a row can only be generated to the same extent as the previous row. Consequently, a process may stay idle waiting for the previous row generation to advance. 


**Changing the parallelization level will affect the output!**

The sides where the overlap occurs differ for each quadrant, 
so it is not possible to reproduce the same result as the sequential algorithm. Higher levels of parallelization do not suffer from this problem conceptually, 
however the current implementation won't generate the same output.

