import os
import random
import glob
from pathlib import Path
from itertools import chain
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import MistralCommonTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import DistributedDataParallelKwargs
from adam_atan2_pytorch import AdoptAtan2

from memory_models import MemoryMLP, MemoryAttention
from mac_transformer import MemoryAsContextTransformer

# ==================== Configuration ====================
NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
LOG_EVERY = 20  # Print loss every 200 steps
SAVE_EVERY = 50  # Save checkpoint every 500 steps
MAX_CHECKPOINTS = 3  # Keep only last 3 checkpoints
GENERATE_EVERY = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True

# Neural memory related
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True

# Data paths
DATA_DIR = "/workspace/llama/data/processeddata"
CACHE_DIR = "/workspace/titans/.datacathe"
TOKENIZER_PATH = "/workspace/llama/weights/mistralai/Ministral-3-14B-Instruct-2512"
CHECKPOINT_DIR = "./checkpoints"

# Performance
USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False
NUM_PROC = 4
BLOCK_SIZE = 2048

# ==================== Helper Functions ====================
def cycle(loader):
    """Infinite data loader"""
    while True:
        for data in loader:
            yield data

def decode_tokens(tokenizer, tokens):
    """Decode tokens to text"""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens, skip_special_tokens=True)

def cleanup_old_checkpoints(checkpoint_dir, max_keep=3):
    """Keep only the last N checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint directories
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    
    # Remove old checkpoints
    if len(checkpoints) > max_keep:
        for old_ckpt in checkpoints[:-max_keep]:
            if old_ckpt.exists():
                import shutil
                shutil.rmtree(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")

                

# define MIRAS loss
def lp_loss_fn(pred, target, p=3.0):
    return (pred - target).abs().pow(p).mean(dim=-1)

def huber_loss_fn(pred, target, delta=1.0):
    return F.huber_loss(pred, target, delta=delta, reduction='none').mean(dim=-1)

# ==================== Data Loading ====================
def load_and_prepare_data():
    """Load and prepare datasets"""
    print("Loading datasets...")
    
    # Scan for data files
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json")) + \
                 glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Found {len(parquet_files)} Parquet files")
    
    loaded_datasets = []
    
    # Load JSON files
    if json_files:
        print("Loading JSON files...")
        ds_json = load_dataset(
            "json", 
            data_files=json_files, 
            split="train", 
            cache_dir=CACHE_DIR
        )
        loaded_datasets.append(ds_json)
    
    # Load Parquet files
    if parquet_files:
        print("Loading Parquet files...")
        ds_parquet = load_dataset(
            "parquet", 
            data_files=parquet_files, 
            split="train", 
            cache_dir=CACHE_DIR
        )
        loaded_datasets.append(ds_parquet)
    
    # Merge datasets
    if loaded_datasets:
        print("Merging datasets...")
        final_dataset = concatenate_datasets(loaded_datasets)
        print(f"Total samples: {len(final_dataset)}")
        return final_dataset
    else:
        print("No data files found!")
        return Dataset.from_list([])

def tokenize_and_chunk_data(raw_datasets, tokenizer):
    """Tokenize and chunk data into fixed-size blocks"""
    print(f"Raw dataset size: {len(raw_datasets)}")
    
    # Tokenization function
    def tokenize_function(examples):
        output_texts = []
        for messages in examples["messages"]:
            ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                continue_final_message=True
            )
            output_texts.append(ids + [tokenizer.eos_token_id])
        return {"input_ids": output_texts}
    
    print("Tokenizing...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=raw_datasets.column_names,
        desc="Tokenizing",
    )
    
    # Group texts into chunks
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop remainder
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        
        # Split into chunks
        result = {
            k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        
        result["labels"] = result["input_ids"].copy()
        return result
    
    print(f"Chunking data (Block Size: {BLOCK_SIZE})...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        desc="Grouping texts",
    )
    
    print(f"Final training samples: {len(lm_datasets)}")
    return lm_datasets

# ==================== Main Training Function ====================
def main():
    # Initialize Accelerator with DDP (instead of FSDP to avoid scalar parameter issue)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY,
        mixed_precision="bf16",  # Use bf16 for better stability
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Set up logging only on main process
    if accelerator.is_main_process:
        print("=" * 60)
        print("Distributed Training with Accelerate + DDP")
        print("=" * 60)
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Batch size per device: {BATCH_SIZE}")
        print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATE_EVERY}")
        print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY * accelerator.num_processes}")
        print("=" * 60)
    
    # Load tokenizer
    tokenizer = MistralCommonTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Load and prepare data (all processes load the same data)
    # This is fine since datasets uses caching
    with accelerator.main_process_first():
        raw_datasets = load_and_prepare_data()
        lm_datasets = tokenize_and_chunk_data(raw_datasets, tokenizer)
    
    # Set format and split
    lm_datasets.set_format(type="torch", columns=["input_ids", "labels"])
    split_dataset = lm_datasets.train_test_split(test_size=0.05)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]
    
    # Create DataLoaders
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        return input_ids
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Create memory model
    if USE_MEM_ATTENTION_MODEL:
        neural_memory_model = MemoryAttention(dim=64)
    else:
        neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)
    
    # VARIANT = "TITANS" # L2 (Original)
    # VARIANT = "MONETA" # Lp
    VARIANT = "TITANS"   # Huber

    if VARIANT == "MONETA":
        loss_fn = partial(lp_loss_fn, p=3.0)
    elif VARIANT == "YAAD":
        loss_fn = partial(huber_loss_fn, delta=1.0)
    else:
        # default Titans L2
        loss_fn = lambda pred, target: (pred - target).pow(2).mean(dim=-1)

    
    # Create main model
    model = MemoryAsContextTransformer(
        num_tokens=tokenizer.vocab_size,
        dim=384,
        depth=8,
        segment_len=WINDOW_SIZE,
        num_persist_mem_tokens=NUM_PERSIST_MEM,
        num_longterm_mem_tokens=NUM_LONGTERM_MEM,
        neural_memory_layers=NEURAL_MEM_LAYERS,
        neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
        neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn=USE_FLEX_ATTN,
        sliding_window_attn=SLIDING_WINDOWS,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=4,
            attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm=NEURAL_MEM_QK_NORM,
            momentum=NEURAL_MEM_MOMENTUM,
            store_memory_loss_fn=loss_fn,
            momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
            use_accelerated_scan=USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
            spectral_norm_surprises=NEURAL_MEM_SPEC_NORM_SURPRISES
        )
    )
    
    print("Applying FSDP scalar parameter fix...")
    for name, param in model.named_parameters():
        if param.dim() == 0:
            # 将 0维 scalar 变为 1维 vector
            param.data = param.data.unsqueeze(0)
            print(f" -> Reshaped scalar param '{name}' from [] to [1]")
    
    # Create optimizer
    optimizer = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)
    
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    global_step = 0
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)
    total_training_steps = NUM_BATCHES // GRADIENT_ACCUMULATE_EVERY
   
    if accelerator.is_main_process:
        print(f"Starting training...\nTotal training steps: {total_training_steps}")
    
    for batch_idx in range(NUM_BATCHES):
        model.train()
        
        # Training step with gradient accumulation
        with accelerator.accumulate(model):
            batch = next(train_iter)
            loss = model(batch, return_loss=True)
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Increment global step after accumulation
        if accelerator.sync_gradients:
            global_step += 1
            
            # Log every LOG_EVERY steps
            if global_step % LOG_EVERY == 0:
                # Gather loss from all processes
                loss_value = accelerator.gather(loss).mean().item()
                percent = (global_step / total_training_steps) * 100
                if accelerator.is_main_process:
                    print(f"Step {global_step}/{total_training_steps} ({percent:.2f}%) | Training Loss: {loss_value:.4f}")
            
            # Validation
            if global_step % LOG_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = next(val_iter)
                    val_loss = model(val_batch, return_loss=True)
                    val_loss_value = accelerator.gather(val_loss).mean().item()
                    if accelerator.is_main_process:
                        print(f"Step {global_step}/{total_training_steps} | Validation Loss: {val_loss_value:.4f}")
            
            # Save checkpoint every SAVE_EVERY steps
            if global_step % SAVE_EVERY == 0:
                if accelerator.is_main_process:
                    checkpoint_path = Path(CHECKPOINT_DIR) / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save model
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(
                        {
                            "model": unwrapped_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                        },
                        checkpoint_path / "model.pt"
                    )
                    
                    print(f"Checkpoint saved at step {global_step}")
                    
                    # Cleanup old checkpoints
                    cleanup_old_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
                
                accelerator.wait_for_everyone()
            
            # Generate samples
            if SHOULD_GENERATE and global_step % GENERATE_EVERY == 0:
                
                def run_generation_task():
                    model.eval()
                    with torch.no_grad():
                        # Random sample from validation set
                        rand_idx = random.randint(0, len(val_ds) - 1)
                        inp = val_ds[rand_idx]['input_ids'][:PRIME_LENGTH]
                        
                        prime = decode_tokens(tokenizer, inp)
                        print(f"\n{'='*60}")
                        print(f"Generation at step {global_step}")
                        print(f"{'='*60}")
                        print(f"Prime: {prime}\n")
                        print('*' * 60)
                        
                        # Generate
                        # 注意：在 summon_full_params 上下文中，unwrap 后的模型参数是完整的 2D 矩阵
                        unwrapped_model = accelerator.unwrap_model(model)
                        sample = unwrapped_model.sample(
                            inp[None, ...].to(accelerator.device), 
                            GENERATE_LENGTH, 
                            use_cache=USE_FAST_INFERENCE
                        )
                        output_str = decode_tokens(tokenizer, sample[0])
                        print(f"Generated: {output_str}")
                        print('='*60 + "\n")

               
                if accelerator.distributed_type == DistributedType.FSDP:
                    
                    with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                        if accelerator.is_main_process:
                            run_generation_task()
                else:
                    
                    if accelerator.is_main_process:
                        run_generation_task()
    
    if accelerator.is_main_process:
        print("Training completed!")

if __name__ == "__main__":
    main()
