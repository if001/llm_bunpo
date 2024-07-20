import wandb
import torch
import numpy as np
import argparse
import os
from typing import Optional
os.environ["TOKENIZERS_PARALLELISM"]="false"

from datasets import load_dataset, concatenate_datasets, DatasetDict
# from peft import LoraConfig, get_peft_model, TaskType
# from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

print("torch version: ", torch.version.cuda)

from hf_config import get_config
from hf_model import get_hf_models

from callbacks import ComputeThroughputCallback, TokenCountCallback
from prepare_dataset import prepare_dataset
from hinshi_encoder import build_hinshi_tokenize

MAX_TOKENS = 8 * 1000 * 1000 * 1000

MAX_LENGTH = 2048
BATCH_SIZE=1
GC_STEPS=1

LOGGING_STEPS=100
SAVE_STEPS=100
# NUM_GPUS=int(os.environ['WORLD_SIZE'])
# LOCAL_RANK = int(os.environ['LOCAL_RANK'])

import transformers
transformers.logging.set_verbosity_info()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    # parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument("--tokenizer", type=str, default="NovelAI/nerdstash-tokenizer-v2")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument('--dataset_ids', required=True, nargs="*", type=str, help='--dataset_ids izumi-lab/wikipedia-ja-20230720') 
    parser.add_argument('--max_steps', default=-1)
    parser.add_argument('--warmup_steps', default=300)
    args = parser.parse_args()
    print("args: ", args)
    return args

def make_dataset(dataset_ids):
    ds = []
    # print(datasets)
    for dataset_id in dataset_ids:
        dataset = load_dataset(dataset_id, split="train", num_proc=8)
        # ds_part = dataset.shuffle(seed=42).select(range(100))
        # ds_part = dataset.shuffle(seed=42)
        ds_part = dataset
        filtered_list = []
        for name in ds_part.column_names:
            if "text" != name:
                filtered_list.append(name)
        ds_part = ds_part.remove_columns(filtered_list)
        ds.append(ds_part)
    combined_dataset = concatenate_datasets(ds)
    print("dataset", combined_dataset)
    return combined_dataset.shuffle(seed=42).train_test_split(test_size=0.1)

def main():
    args = parse_arguments()
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.init(project=args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    encoder = build_hinshi_tokenize(tokenizer, rate=args.mask_rate)
    t = "朝日新聞によると、日本の菅義偉総務大臣は北朝鮮による日本人拉致問題に関する放送を、"
    id = encoder(t, add_special_tokens=True)
    print(id)
    print(tokenizer.decode(id))
    print()
    exit(0)
    config = get_config(args.model_name)
    model = get_hf_models(config)
    # model = AutoModelForCausalLM.from_pretrained(
    #         args.repo_id,
    #         # torch_dtype=torch.float16
    #         )
    print("--- model config ... ---")
    print(model.config)    
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("--- making dataset ... ---")
    dataset = make_dataset(args.dataset_ids)
    train_dataset = prepare_dataset(dataset["train"], tokenizer, encoder=encoder, add_special_tokens=False, append_concat_token=True)
    test_dataset = prepare_dataset(dataset["test"], tokenizer, encoder=encoder, add_special_tokens=False, append_concat_token=True)

    print("--- training start ... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GC_STEPS,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=1000,
        weight_decay=0.01,
        # optim="adamw_apex_fused",
        # optim="adafactor",
        logging_dir=args.output_dir,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        learning_rate=6.0e-5,
        # min_lr
        save_strategy="steps",
        save_total_limit=3,
        save_steps=SAVE_STEPS,
        report_to="wandb",
        # bf16=True,
        fp16=True,
        # ddp_backend="nccl",
        # half_precision_backend="apex",
        # deepspeed=args.ds_config_path,
        dataloader_pin_memory=True,
        dataloader_num_workers=16,
        # torch_compile=True,
        # num_workers=16,
        # fsdp="full_shard",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        max_steps=args.max_steps
    )
    print("parallel_mode: ", training_args.parallel_mode)
    print("world_size", training_args.world_size)

    computeThroughput = ComputeThroughputCallback(
        vocab_size=model.config.vocab_size,
        #seq_length=model.config.max_sequence_length,
        seq_length=model.config.max_position_embeddings,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        world_size=1,
        log_steps=LOGGING_STEPS,
    )
    tokenCounter = TokenCountCallback(max_token_count=MAX_TOKENS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[computeThroughput, tokenCounter]
    )

    trainer.train(resume_from_checkpoint=args.resume)
    print("train done..")

    model.save_pretrained(args.output_dir)
    print("save...") 
    
    for v in model.state_dict():
        print(v, model.state_dict()[v].shape)
    print("="*100)

    if args.upload_repo_id:
        print("--- push to hf ---")
        model.push_to_hub(args.upload_repo_id)
        print("upload done...") 
    wandb.finish()

if __name__ == "__main__":
    main()