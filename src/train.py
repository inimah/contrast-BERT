import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset
from datasets import concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available

import transformers
from datasets import load_dataset
from datasets import interleave_datasets # use interleave batch sampling method

from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    
    attribute: Optional[str] = field(
        default=None, 
        metadata={"help": "Attribute of dataset used as input: title, content, fact "}
    )
        
    
        
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The validation data file (.txt or .csv)."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The test data file (.txt or .csv)."}
    )
    
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    '''
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
    '''


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
        
    # SimCSE arguments:
    method: Optional[str] = field(
        default=None, 
        metadata={"help": "supervised, unsupervised"}
    )

    datapath: str = field(
        default=None, 
        metadata={"help": "Huggingface path to the data"}
    )
        
    pairsdatapath: str = field(
        default=None, 
        metadata={"help": "Huggingface path to the pairwise data for validation stage (FakePairs)"}
    )
        
    evalpath: str = field(
        default=None, 
        metadata={"help": "Huggingface path to the eval data (Fake news classification)"}
    )
        
    eval_attribute: Optional[str] = field(
        default=None, 
        metadata={"help": "Attribute of dataset used as input: title, content, fact "}
    )
        
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
        

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    sys.stdout.flush()

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    
    # read data based on attribute provided
    
    print("training_args.method:", training_args.method)
    sys.stdout.flush()
    
    if 'supervised' in training_args.method and training_args.method != 'unsupervised':
        # supervised training
        # used for both pairwise contrastive and triplets 
            
        if training_args.method == "supervised-pairs-filtered":
            # TRIPLETS subset
            data_files = {
                            "train_hoax_cnt": "train/train_hoax_cnt.csv", \
                            "train_hoax_fct": "train/train_hoax_fct.csv", \
                            "train_nonhoax_cnt": "train/train_nonhoax_cnt.csv", \
                            "train_nonhoax_fct": "train/train_nonhoax_fct.csv", \
                            "test_hoax_cnt": "test/test_hoax_cnt.csv", \
                            "test_hoax_fct": "test/test_hoax_fct.csv", \
                            "test_nonhoax_cnt": "test/test_nonhoax_cnt.csv", \
                            "test_nonhoax_fct": "test/test_nonhoax_fct.csv", \
                            }
            
            
        else:
            #elif training_args.method == "supervised-triplets-filtered":
            # TRIPLETS subset
            data_files = {
                            "train_hoax_cnt": "train/train_hoax_cnt.csv", \
                            "train_hoax_fct": "train/train_hoax_fct.csv", \
                            "train_nonhoax_cnt": "train/train_nonhoax_cnt.csv", \
                            "train_nonhoax_fct": "train/train_nonhoax_fct.csv", \
                            "test_hoax_cnt": "test/test_hoax_cnt.csv", \
                            "test_hoax_fct": "test/test_hoax_fct.csv", \
                            "test_nonhoax_cnt": "test/test_nonhoax_cnt.csv", \
                            "test_nonhoax_fct": "test/test_nonhoax_fct.csv", \
                            }
            
        
            
        # training set
        hoax_cnt_train = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="train_hoax_cnt", trust_remote_code=True, token="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")
        nonhoax_cnt_train = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="train_nonhoax_cnt", trust_remote_code=True, \
                                             token="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")
        hoax_fct_train = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="train_hoax_fct", trust_remote_code=True, token="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")
        nonhoax_fct_train = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="train_nonhoax_fct", trust_remote_code=True, \
                                             token="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")
        
        train_datasets = concatenate_datasets([hoax_cnt_train, nonhoax_cnt_train, hoax_fct_train, nonhoax_fct_train])
        
       
        
        # test set
        
        hoax_cnt_test = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="test_hoax_cnt", trust_remote_code=True, token="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")
        nonhoax_cnt_test = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="test_nonhoax_cnt", trust_remote_code=True, \
                                             token="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")
        hoax_fct_test = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="test_hoax_fct", trust_remote_code=True, token="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")
        nonhoax_fct_test = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="test_nonhoax_fct", trust_remote_code=True, \
                                             token="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")
        
        test_datasets = concatenate_datasets([hoax_cnt_test, nonhoax_cnt_test, hoax_fct_test, nonhoax_fct_test])
        
    else:
        #print("this is for unsupervised section")
        #sys.stdout.flush()
        
        # datasets for unsupervised training
        if data_args.attribute == 'title':
            
            
            data_files = {
                        "train_hoax_ttl": "train/hoax_ttl.csv", \
                        "test_hoax_ttl": "test/hoax_ttl.csv", \
                        "train_nonhoax_ttl": "train/nonhoax_ttl.csv", \
                        "test_nonhoax_ttl": "test/nonhoax_ttl.csv", \
                        
                        }
            
            hoax_ttl_train = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="train_hoax_ttl", trust_remote_code=True, token="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")
            nonhoax_ttl_train = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="train_nonhoax_ttl", trust_remote_code=True, \
                                             token="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")
            
            # for fully unsupervised
            train_datasets = concatenate_datasets([hoax_ttl_train, nonhoax_ttl_train])
            
            hoax_ttl_valid = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="valid_hoax_ttl", trust_remote_code=True, token="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")
            nonhoax_ttl_valid = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="valid_nonhoax_ttl", trust_remote_code=True, \
                                             token="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")

            dev_datasets = interleave_datasets([hoax_ttl_valid, nonhoax_ttl_valid], seed=42, stopping_strategy="all_exhausted")
            #dev_datasets = concatenate_datasets([hoax_ttl_valid, nonhoax_ttl_valid])
            
            hoax_ttl_test = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="test_hoax_ttl", trust_remote_code=True, token="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")
            nonhoax_ttl_test = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="test_nonhoax_ttl", trust_remote_code=True, \
                                             token="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")

            test_datasets = concatenate_datasets([hoax_ttl_test, nonhoax_ttl_test])
            
        elif data_args.attribute == 'title-content':
            
            
            data_files = {
                        "train_hoax_ttl": "train/hoax_ttl_cnt.csv", \
                        "test_hoax_ttl": "test/hoax_ttl_cnt.csv", \
                        "train_nonhoax_ttl": "train/nonhoax_ttl_cnt.csv", \
                        "test_nonhoax_ttl": "test/nonhoax_ttl_cnt.csv", \
                        
                        }
            
            hoax_ttl_train = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="train_hoax_ttl", trust_remote_code=True, token="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")
            nonhoax_ttl_train = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="train_nonhoax_ttl", trust_remote_code=True, \
                                             token="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")
            
            # for fully unsupervised
            train_datasets = concatenate_datasets([hoax_ttl_train, nonhoax_ttl_train])
            
            hoax_ttl_test = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="test_hoax_ttl", trust_remote_code=True, token="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")
            nonhoax_ttl_test = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="test_nonhoax_ttl", trust_remote_code=True, \
                                             token="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")

            test_datasets = concatenate_datasets([hoax_ttl_test, nonhoax_ttl_test])
            
        
        else:
            #elif data_args.attribute == 'title-fact':
            # use subset from triplets_filtered
            
            
            data_files = {
                        "train_hoax_fct": "train/hoax_ttl_fct.csv", \
                        "test_hoax_fct": "test/hoax_ttl_fct.csv", \
                        "train_nonhoax_fct": "train/nonhoax_ttl_fct.csv", \
                        "test_nonhoax_fct": "test/nonhoax_ttl_fct.csv", \
                        
                        }
            
            hoax_train = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="train_hoax_fct", trust_remote_code=True, token="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")
            nonhoax_train = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="train_nonhoax_fct", trust_remote_code=True, \
                                             token="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")
            
            # for fully unsupervised
            train_datasets = concatenate_datasets([hoax_train, nonhoax_train])
            
                      
            hoax_test = load_dataset(training_args.datapath, data_files=data_files, \
                                          split="test_hoax_fct", trust_remote_code=True, token="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")
            nonhoax_test = load_dataset(training_args.datapath, data_files=data_files, \
                                             split="test_nonhoax_fct", trust_remote_code=True, \
                                             token="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")

            test_datasets = concatenate_datasets([hoax_test, nonhoax_test])
        
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path or 'multilingual' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    #column_names = datasets["train"].column_names
    column_names = train_datasets.column_names
    
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives (triplets)
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to training_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=training_args.max_seq_length,
            truncation=True,
            padding="max_length" if training_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features

    if training_args.do_train:
        #train_dataset = datasets["train"].map(
        train_dataset = train_datasets.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
    if training_args.do_eval:
        
        test_dataset = test_datasets.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = default_data_collator if training_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)
    

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        '''
        best model is defined based on loss on evaluation dataset
        metric 1 : senteval transfer learning
        metric 2 : STS spearman . correlation to STS data
        metric 3 : sickr_spearman . correlation to SICK relatedness
        metric 4 : eval_avg_sts . avg (stsb_spearman + sickr_spearman)
        metric 5: In Domain fake news similarity
        '''
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True) # with senteval benchmarking
        

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
