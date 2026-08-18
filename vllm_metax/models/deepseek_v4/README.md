# Deepseek_v4 on MetaX

The quantization methods of deepseek v4 on metax is different with upstream's default implementation. Here we make a clear comparison.

## Deepseek v4 attention

|modules| upstream | Metax |
|-------| -------- | ----- |
| **kv_cache** | *fp8 / fp8_ds_mla* | *bf16* |


### indexer

|modules| upstream | Metax |
|-------| -------- | ----- |
| **kv_cache** | *fp8 / mxfp4* | *int8* |
| **fused_indexer_q_rope_quant** | *fp8* | *int8* |

#### Sparse Attention indexer

|modules| upstream | Metax |
|-------| -------- | ----- |
|**sparse_attn_indexer_impl**| *fp8* | *int8 / bf16* |



### swa 

TODO: to be determined.

### Compressor

# Note: Metax use full attn bf16 + indexer int8

|modules| upstream | Metax |
|-------| -------- | ----- |
|**_fused_kv_compress_norm_rope_insert_sparse_attn**| *fp8* | *bf16* |
|**_fused_kv_compress_norm_rope_insert_indexer_attn**| *fp8* | *int8* |

