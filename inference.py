import torch
import os
from transformers import MistralCommonTokenizer
from mac_transformer import MemoryAsContextTransformer
from memory_models import MemoryMLP, MemoryAttention

# ==================== Configuration (保持与训练一致) ====================
# 模型超参数 (必须与训练代码完全一致，否则报错)
DIM = 384
DEPTH = 8
DIM_HEAD = 64
HEADS = 4

# 来自你的配置文件
WINDOW_SIZE = 32
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
USE_MEM_ATTENTION_MODEL = False

# Memory Kwargs
MEMORY_KWARGS = dict(
    dim_head=DIM_HEAD,
    heads=HEADS,
    attn_pool_chunks=True,  # STORE_ATTN_POOL_CHUNKS
    qk_rmsnorm=True,        # NEURAL_MEM_QK_NORM
    momentum=True,          # NEURAL_MEM_MOMENTUM
    momentum_order=1,       # NEURAL_MEM_MOMENTUM_ORDER
    default_step_transform_max_lr=1e-1, # NEURAL_MEM_MAX_LR
    use_accelerated_scan=True, # USE_ACCELERATED_SCAN
    per_parameter_lr_modulation=True, # MEMORY_MODEL_PER_LAYER_LEARNED_LR
    spectral_norm_surprises=True, # NEURAL_MEM_SPEC_NORM_SURPRISES
)

# 路径配置
TOKENIZER_PATH = "/workspace/llama/weights/mistralai/Ministral-3-14B-Instruct-2512"
# 修改这里为你想要测试的具体 checkpoint 文件路径
CHECKPOINT_PATH = "./checkpoints/checkpoint-5950/model.pt" 

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = MistralCommonTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None

    print("Initializing model structure...")
    
    
    if USE_MEM_ATTENTION_MODEL:
        neural_memory_model = MemoryAttention(dim=DIM_HEAD)
    else:
        neural_memory_model = MemoryMLP(dim=DIM_HEAD, depth=NEURAL_MEMORY_DEPTH)

    # 1. 初始化主模型 (修正 heads 参数)
    model = MemoryAsContextTransformer(
        num_tokens=tokenizer.vocab_size,
        dim=DIM,
        depth=DEPTH,
        segment_len=WINDOW_SIZE,
        num_persist_mem_tokens=NUM_PERSIST_MEM,
        num_longterm_mem_tokens=NUM_LONGTERM_MEM,
        neural_memory_layers=NEURAL_MEM_LAYERS,
        neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
        neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn=True,
        sliding_window_attn=True,
        dim_head=DIM_HEAD,
        heads=8, 
        ff_mult=4,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=MEMORY_KWARGS
    )
    
    print("Fixing parameter shapes and memory layout...")
    
    for name, param in model.named_parameters():
       
        if param.dim() == 0:
            param.data = param.data.unsqueeze(0)
        
        
        if "memory_model_parameters" in name:
            param.data = param.data.clone()

    # loading weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        try:
           
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if len(missing) > 0:
                print(f"[Warning] Missing keys: {len(missing)}")
            if len(unexpected) > 0:
                print(f"[Warning] Unexpected keys: {len(unexpected)}")
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading state dict: {e}")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}, using random initialization.")

    model.to(DEVICE)
    model.eval()
    return model, tokenizer

model,tokenizer = load_model()




def generate_text(model, tokenizer, prompt_text, max_new_tokens=256, temperature=0.7, top_k=50):
    """
    使用 MemoryAsContextTransformer 进行文本生成
    """
    model.eval() # 确保模型处于评估模式

    # 1. 数据预处理
    # 因为训练时使用了 apply_chat_template，推理时也必须保持一致的格式
    # 否则模型可能无法理解指令
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    # Mistral Tokenizer 处理
    # apply_chat_template 通常返回 list[int]
    try:
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    except Exception as e:
        print(f"Chat template failed, falling back to simple encode: {e}")
        input_ids = tokenizer.encode(prompt_text)

    # 转换为 Tensor [1, seq_len] 并移动到设备
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    input_len = input_tensor.shape[1]
    target_len = input_len + max_new_tokens
    
    print(f"Prompt Length: {input_len} tokens | Generating up to: {target_len} tokens")

    # 2. 模型推理 (Sampling)
    with torch.no_grad():
        # 调用 mac_transformer.py 中的 sample 方法
        # 该方法内部已经处理了 KV Cache 和 Neural Memory 的状态传递
        generated_tokens = model.sample(
            prompt=input_tensor,
            seq_len=target_len,        # 总长度 (Prompt + New Tokens)
            temperature=temperature,   # 温度参数，控制随机性
            use_cache=True,            # 开启 Cache 加速推理 (包括 KV Cache 和 Memory Cache)
            show_progress=True,        # 显示进度条
            filter_kwargs=dict(min_p=0.1) # 使用 Min-P 采样 (或者你可以改代码用 top-k)
        )

    # 3. 解码输出
    # model.sample 返回的通常是生成的*新* Token 部分 (根据 mac_transformer.py 的实现)
    # generated_tokens shape: [1, new_tokens_len]
    output_ids = generated_tokens[0].tolist()
    
    # 解码为文本
    decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return decoded_text

def main():
    if model is None:
        print("Model failed to load, exiting.")
        return

    # 测试 Prompt
    test_prompts = [
        "Please explain the theory of relativity.",
        "Solve this math problem: 24 * 12 =",
        "Write a poem about the ocean."
    ]
    
    print(f"{'='*20} Start Generation {'='*20}")
    
    for prompt in test_prompts:
        print(f"\n[Input]: {prompt}")
        # 这里设置稍微短一点的 token 以便快速测试，正式使用可以加大
        result = generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.6)
        print(f"[Output]:\n{result}")
        print(f"{'-'*60}")

if __name__ == "__main__":
    main()
