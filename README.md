torchzfp
=========

A utility library for zfp compression / decompression of Torch7 float tensors.

It (usually) gets better compression ratios than zlib (see https://github.com/jonathantompson/torchzlib for a Torch7 wrapper) for floating point types, particularly tensors that exhibit lots of spatial coherence. The caveat is that usually you need to set the ``accuracy`` value to lossy compression to see any gains.

You can read more about the method here: http://computation.llnl.gov/projects/floating-point-compression (based on the paper "Fixed-Rate Compressed Floating-Point Arrays" by Peter Lindstrom).

The main (and only) API entry point is a new class ```torch.ZFPTensor```.  This is a super simple class that creates a compressed ByteTensor of an input tensor and has a single ```decompress()``` method to return the original data.

The constructor signature is:

``` lua
torch.ZFPTensor(tensor)
```

Where ```tensor``` is the tensor to be compressed.

Usage:

```lua
require 'torchzfp'
require 'image'

data = image.lena():double()  -- Can be double or float.
accuracy = 1e-4
dataCompressed = torch.ZFPTensor(data, accuracy)  -- Compress data.
dataDecompressed = dataCompressed:decompress()
```

