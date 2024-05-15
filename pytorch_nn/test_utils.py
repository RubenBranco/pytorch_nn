from torch.nn import Module


def copy_weights(src: Module, dst: Module):
    for (src_name, src_param), (dst_name, dst_param) in zip(src.named_parameters(), dst.named_parameters()):
        try:
            dst_param.data.copy_(src_param.data)
        except RuntimeError:
            print(f"Failed to copy {src_name} to {dst_name}")
            raise


def copy_rnn_weights(src: Module, dst: Module):
    for src_name, src_params in src.named_parameters():
        try:
            param_prefix, layer_num = src_name.split(".")
            getattr(dst, f"{param_prefix}{layer_num}").data.copy_(src_params.data)
        except RuntimeError:
            print(f"Failed to copy {src_name}")
            raise
