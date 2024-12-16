# The QuaRot Add-On
This is a simple demo project to illustrate some key concepts from the [QuaRot](https://arxiv.org/abs/2404.00456) paper.

## Outline
At the time of this writing QuaRot is a novel quantization approach for end-to-end 4-bit LLM quantization. One fundamental concept making this approach efficient is _Incoherence Processing_. This demo aims to show the effects of incoherence processing by comparing the application of simple round-to-nearest (RTN) quantization to processed and unprocessed example weight matrices. For processing we use _Hadamard-Walsh Transformations_ as in the paper.

Actions performed by the demo:

1. Transform sample weight matrices using hadamard transformations
2. Quantize the original matrix and the processed matrix to 4-bit using RTN
3. Dequantize both matrices again.
4. Revert the transformation performed during incoherence processing
5. Compare both matrices to the original matrix to show incoherence processing results in better reconstruction

Optional: Show computational invariance
 
1. Implement a simple pre-RMSNorm FFN Part of a transformer block
2. Prepare the block as proposed in the paper and compare both results when using the same input
