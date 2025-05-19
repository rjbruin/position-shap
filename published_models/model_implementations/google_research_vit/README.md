# Google Research ViT

**Source**:
https://github.com/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb

Implementation derived from notebook `vit_jax_augreg.ipynb`, specifically the part for loading PyTorch models using `timm`.

Original model code:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

## Requirements

```
conda install tensorflow -c conda-forge
```

## Known issues

**TensorFlow GFile missing certificate**

On some Linux systems TensorFlow cannot find the right SSL certificate when
using `tf.io.gfile.GFile` with a file on a `gs://` server.

Fix:

```python
import os
os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-certificates.crt"
```

This fix is already built into the code. Note though that this fix *does not
work on the HPC*. My solution is to run the code on my local machine, so that
the checkpoint gets downloaded, then upload the checkpoint to the cluster
manually so that the code on the cluster doesn't have to download the file. The
code is also written this way: it won't apply the environment variable change if
the file doesn't have to be downloaded.