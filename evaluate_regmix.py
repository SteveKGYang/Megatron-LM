# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce, partial
from logging import logMultiprocessing
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

import lm_eval
from lm_eval.api.model import LM, CacheHook
# from lm_eval import evaluator, tasks, utils

from tqdm import tqdm
import torch.nn.functional as F

# from lm_eval.api.registry import ALL_TASKS
import numpy as np
import time
import warnings

import torch
from megatron.training import get_args, print_rank_0
from megatron.training import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import get_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from megatron.training.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
import pickle
import json

from megatron.training.initialize import initialize_megatron
import megatron
from megatron.training.arguments import parse_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint

from pretrain_gpt import model_provider
from text_generation_utils import generate_samples_from_prompt

from torch.nn.parallel import DistributedDataParallel as torchDDP
from megatron.legacy.model import Float16Module
try:
    from msamp.megatron import FP8DistributedDataParallel as LocalDDP
except ImportError:
    from megatron.core.distributed import DistributedDataParallel as LocalDDP

try:
    from deepspeed.runtime.pipe import schedule
    from deepspeed.accelerator import get_accelerator
    from megatron.training import setup_model_and_optimizer
    from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
    from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint
    deepspeed_available = True
except ImportError:
    deepspeed_available = False

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
    

class EvalHarnessAdaptor(lm_eval.api.model.LM):
    def __init__(self, model, tokenizer, accelerator):
        args = get_args()
        self.accelerator = accelerator
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod
        self.max_gen_toks = 512

        self.generate = partial(
            generate_samples_from_prompt,
            tokenizer=self.tokenizer,
            model=self.model,
        )

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size

        self.cache_hook = lm_eval.api.model.CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        if deepspeed_available:
            self._device = get_accelerator().current_device_name()
        else:
            self._device = torch.cuda.current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        # currently 'Namespace' object has no attribute 'moe_expert_parallel_size'
        # if self.is_data_parallel and args.moe_expert_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
        #     raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    @property
    def rank(self):
        return torch.cuda.current_device()
        # return 0            # 0116: fix multi-gpu issue related to lm-eval
    
    @property
    def world_size(self):
        return self.accelerator.num_processes
        # return 1            # 0116: fix multi-gpu issue related to lm-eval


    def loglikelihood(self, requests):
        new_reqs = []
        for req in requests:
            context, continuation = req.args
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(lm_eval.utils.make_disjoint_window, lm_eval.utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = lm_eval.utils.Reorderer(requests, _collate)
            for chunk in lm_eval.models.utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        # answer = (float(logits.sum()), bool(max_equal))
                        answer = (float(logits.mean()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        args = get_args()
        config = core_transformer_config_from_args(args)

        if deepspeed_available:
            if args.deepspeed:
                if args.no_pipeline_parallel:
                    # self.model.set_batch_fn(self.create_model_inputs)
                    # round up to multiple of micro_batch_size
                    new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
                    padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                    # dummy data iterator for pipelining.
                    data_iterator = list((torch.stack(inp) for inp in lm_eval.utils.chunks(padded, args.micro_batch_size)))
                    self.model.micro_batches = len(data_iterator)
                    # output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)
                    output = []
                    for tokens in data_iterator:
                        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                                                                    tokens,
                                                                    self.EOT_TOKEN_ID,
                                                                    args.reset_position_ids,
                                                                    args.reset_attention_mask,
                                                                    args.eod_mask_loss)
                        a_output, *other_losses = self.model(tokens,
                            position_ids,
                            attention_mask,
                            tokentype_ids=None)
                        output.append(a_output)

                    if output is not None:
                        output = torch.cat(output, 0)[:len(inps)]
                    else:
                        output = None

                    # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                    if args.adaptive_seq_len:
                        self.model.total_loss = None
                else:
                    self.model.set_batch_fn(self.create_model_inputs)
                    # round up to multiple of micro_batch_size
                    new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
                    padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                    # dummy data iterator for pipelining.
                    data_iterator = list((torch.stack(inp) for inp in lm_eval.utils.chunks(padded, args.micro_batch_size)))
                    self.model.micro_batches = len(data_iterator)
                    output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)


                    if output is not None:
                        output = torch.cat(output, 0)[:len(inps)]
                    else:
                        output = None

                    # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                    if args.adaptive_seq_len:
                        self.model.total_loss = None
        else:
            # Since the shape of the micro-batch will change
            # We need set the correct shapes here
            # So that latter pipeline stages knows which shapes to expect.
            # Otherwise we will deadlock.

            args.micro_batch_size = len(inps)
            args.seq_length = len(inps[0])
            args.max_position_embeddings = args.seq_length

            tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
            input_tensor = recv_forward(tensor_shape, config)

            # Forward pass through the model.
            unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            output = self.model(*self.create_model_inputs(inps)[0])
            send_forward(output, config)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        # from megatron.training.tokenizer.gpt2_tokenization import GPT2Tokenizer
        # if isinstance(self.tokenizer, GPT2Tokenizer):
        #     return self.tokenizer.tokenize(text)
        # else:
        #     return self.tokenizer.tokenize(text)
        return self.tokenizer.tokenize(text)
        
    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        # self.model.module.inference_mode(use_cache=True)  # tell model to cache kv pairs
        res = []

        # get only the args from each Instance object
        reqs = [req.args for req in requests]

        def _collate(x):
            # toks = self.tokenizer.encode(x[0])
            toks = self.tokenizer.tokenize(x[0])
            return (len(toks), x[0])

        reord = lm_eval.utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in tqdm(
            reord.get_reordered(), "Running greedy generation"
        ):
            if isinstance(gen_kwargs, dict):
                import copy
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                )
            if not until:
                until = [self.tok_decode(self.eot_token_id)]
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            if "do_sample" in kwargs.keys():
                kwargs.pop("do_sample")

            stop_tokens = [self.tokenizer.tokenize(i) for i in until]
            # cont = self.generate(
            #     text=context,
            #     stop_tokens=stop_tokens,
            #     recompute=self.neox_args.recompute,
            #     maximum_tokens=max_gen_toks,
            #     **kwargs,
            # )
            cont = self.generate(
                text=context,
                stop_tokens=stop_tokens,
                eos_token_id=self.EOT_TOKEN_ID,
                maximum_tokens=max_gen_toks,
                **kwargs,
            )
            if cont:
                s = cont[0]["text"] or ""
            else:
                s = ""

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("generate_until", (context, until), s)

            res.append(s)

        # self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)



def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    # parse the megatorn args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = parse_args(extra_args_provider=extra_args_provider)

    ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                        tp_degree=args.tensor_model_parallel_size,
                                        pp_degree=args.pipeline_model_parallel_size,
                                        no_pp=args.no_pipeline_parallel)


    cp_args = ds_checkpoint.get_args()
    # Merge the current args with the checkpoint args.
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration', 'inference', 'random_ltd']

    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    cp_args.tokenizer_type = 'GPT2BPETokenizer'

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    megatron.arguments.parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._ensure_var_is_not_initialized = lambda *_args, **kwarg: None
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron(extra_args_provider=extra_args_provider)
    megatron.global_vars._GLOBAL_ARGS = args
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    # print final arguments.
    _print_args("eval_harness arguments", args)
    if args.deepspeed:
        
        assert deepspeed_available, "Deepspeed is not available, but args.--deepspeed is set."

        # Hack #3:
        # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
        # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
        # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
        # which does not contain these attention-specific keys.
        #
        # Deepspeed does however manage to load the model if we just turn off this sanity check.
        import deepspeed
        deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None


        cp_path = args.load
        args.load = None
        model, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
        model = model[0]
        zero_enabled = model._config.zero_enabled
        model._config.zero_enabled = False
        _, _ = model.load_checkpoint(cp_path, tag = '.', load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        model._config.zero_enabled = zero_enabled
    else:
        model = get_model(model_provider)[0]
        # Initialize megatron model using the parsed state dict.
        sd = _create_rank_checkpoint(ds_checkpoint, None, mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(), True)

        model.load_state_dict(sd['model'], strict=True)

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    # group.add_argument('--task-list', type=str, default = "hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--task-list', type=str, default = "lambada", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results-path', type=str, default = "./results_mmlu_h100.json", help='Path to where the results will be stored.')
    group.add_argument('--adaptive-seq-len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num-fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval-fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--trust-remote-code', action='store_true')
    return parser

# from megatron.arguments import parse_args

def main():
    start = time.time()

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    print(f"Deepspeed_available: {deepspeed_available}")
    if deepspeed_available:
        model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)
    else:
        
        # _print_args = megatron.arguments._print_args
        # megatron.arguments._print_args = lambda *_args, **kwarg: None
        args = parse_args(extra_args_provider=tasks_args)

        #! do not load rng and optimizer states
        initialize_megatron(extra_args_provider=tasks_args, args_defaults={'no_load_optim': True})
        args = get_args()
        # model, _, _ = setup_model_and_optimizer(model_provider, model_type=ModelType.encoder_or_decoder)      # 对于大模型，不需要load optimizer
        model = get_model(model_provider, wrap_with_ddp=True)       # must support warp_with_ddp=True (MS_AMP_DDP) to correctly load MS-AMP checkpoint

        if args.load is not None:
            _ = load_checkpoint(model, None, None, strict=True)

        assert len(model) == 1, "Above condition should have caught this"
        model = model[0]
        
    if deepspeed_available:
        if args.deepspeed and args.adaptive_seq_len:
            # adaptive_seq_len hack #1:
            # CL automatically enables reset_activation_shape() which allows us to change input shapes
            # and it also reshapes the attenion scores in attention_mask_func
            args.curriculum_learning_legacy = 1

    # task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')

    if args.trust_remote_code:
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    task_list = args.task_list.split(',')
    print(f"task_list: {task_list}")
    task_dict = lm_eval.tasks.get_task_dict(task_list)
    # print(f"task_dict: {task_dict}")

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer, accelerator)
    # results = lm_eval.evaluator.evaluate(adaptor, task_dict, False, args.num_fewshot, None)
    
    #!
    # if args.msamp:
    #     from megatron.core import tensor_parallel
    #     tensor_parallel.linear_with_grad_accumulation_and_async_allreduce.permit_fp4_computation = True     # todo 这里只需要把这个属性设置下就行，否则会报错找不到这个属性
    
    results = lm_eval.simple_evaluate(
        model=adaptor, 
        tasks=task_list, 
        num_fewshot=args.num_fewshot
    )
    
    print(results)

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        # print(json.dumps(results, indent=2))
        
        try:
            parent_dir = os.path.dirname(args.results_path)
            file_name = os.path.basename(args.results_path)
            
            if not file_name:           # user has provided a directory
                if not os.path.exists(args.results_path):
                    os.makedirs(args.results_path, exist_ok=True)
                file_path = os.path.join(args.results_path, "results.json")
            else:                       # user has provided a file path
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                file_path = args.results_path
        except Exception as e:
            warnings.warn(f"Invalid path: {args.results_path}, caught error message: {e}")
            warnings.warn("saving to ./results.json instead.")
            file_path = "./results.json"
            
        with open(file_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)
            
    end = time.time()
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()