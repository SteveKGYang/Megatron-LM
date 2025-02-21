import safetensors.torch
import sys
sys.path.append("megatron")
print(sys.path)
import os
import re
import torch
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from pretrain_gpt import model_provider
from megatron.training import get_args
import argparse

# from toolkits.model_checkpoints_convertor.utils import (
#     save_state_dict,
#     save_hfmodel
# )

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def add_model_args(parser=None):
    if parser is None:
        # create a new parser from scratch
        parser = argparse.ArgumentParser(description='moe parser')
    
    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=5
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--target-decoder-last-pipeline-num-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    parser.add_argument(
        "--load",
        type=str,
        default='/home/v-yucding/blob_ruizhe/blob_files/checkpoints/deepseek_moe_debug/moe_64exp_lite_dclm_1T_18node_0219_1',
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--qk-head-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--qk-pos-emb-head-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_attention-heads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
    )

    return parser


def load_megatron_model(args):
    # os.makedirs(args.save, exist_ok=True)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    # os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)

    # os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)

    # model = model_provider()

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    if args.num_experts is not None:
        pattern = r'weight(\d+)'
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)

    if (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.expert_model_parallel_size == args.tensor_model_parallel_size
    ):
        #assert args.num_layers % args.pipeline_model_parallel_size == 0
        if args.target_decoder_first_pipeline_num_layers is not None and args.target_decoder_last_pipeline_num_layers is not None:
            remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers - args.target_decoder_last_pipeline_num_layers
            remained_stages = args.pipeline_model_parallel_size - 1
            assert remained_layers % remained_stages == 0
            pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages) + [args.target_decoder_last_pipeline_num_layers]
        else:
            raise ValueError('not support yet')
            pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size

        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                ep_rank = tp_rank
                layer_offset = sum(pp_layers_per_stage[:pp_rank])
                for layer in range(pp_layers_per_stage[pp_rank]):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[(pp_rank, layer)] = pp_layer_id

                if args.expert_model_parallel_size > 1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                            ep_rank)
                elif args.expert_model_parallel_size == 1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                            False)
                print(f'load {checkpoint_name} with pp_rank {pp_rank} tp_rank {tp_rank} ep_rank {ep_rank}')
                split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                # print(split_state.keys())
                for k, v in split_state.items():
                    try:
                        if 'experts' in k:
                            local_expert_rank = int(re.findall(pattern, k)[0])
                            expert_rank = local_expert_rank + num_local_experts * ep_rank
                            k = k.replace(f'experts.{local_expert_rank}', f'experts.{expert_rank}')
                            # print(k)
                            # print(v.size())
                        # else:
                        #     continue
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                        # print(tgt)
                        if 'linear_proj' in k or 'linear_q_down_proj' in k or 'linear_q_up_proj'in k or 'linear_kv_up_proj' in k or 'linear_kv_down_proj' in k or\
                                'decoder.layers.0.mlp.linear_fc1' in k or 'decoder.layers.1.mlp.linear_fc1' in k or 'decoder.layers.2.mlp.linear_fc1' in k or \
                                'decoder.layers.0.mlp.linear_fc2' in k or 'decoder.layers.1.mlp.linear_fc2' in k or 'decoder.layers.2.mlp.linear_fc2' in k or \
                                'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k:
                            if ep_rank ==0:
                                mid_state[tgt].append(v)
                        else:
                            mid_state[tgt].append(v)
                    except:
                        pass
                        # if "word_embeddings" in k:
                        #     if ep_rank ==0 and pp_rank == 0:
                        #         mid_state[k].append(v)
                        # elif "output_layer" in k or "final_layernorm" in k:
                        #     if ep_rank ==0 and pp_rank == args.pipeline_model_parallel_size - 1:
                        #         mid_state[k].append(v)
                        # else:
                        #     raise ValueError(f"{k} is missing! ")
                break

        for k, v in mid_state.items():
            if 'extra_state' in k:
                continue
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'input_layernorm' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_down_proj' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_up_proj' in k and 'layer_norm_weight' not in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_q_up_proj.layer_norm_weight' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_kv_down_proj' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_kv_up_proj' in k and 'layer_norm_weight' not in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_kv_up_proj.layer_norm_weight' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                print(len(viewed))
                print(f'{k} view size', viewed[0].size())
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
                # print(f'{k} original size', x.size())
                print(f'{k} convert size', target_v.size())
            elif 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            else:
                # raise ValueError(f"{k} is missing!")
                pass
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')

    # model.load_state_dict(state_dict, strict=False)
    # return model

if __name__ == "__main__":
    # parser = get_args()
    parser = add_model_args()
    args = parser.parse_args()
    # initialize_megatron(args)
    model = load_megatron_model(args)