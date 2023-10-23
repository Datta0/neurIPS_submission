from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
import torch, os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import os
from copy import deepcopy
from wandb import login as wandb_login

# Set torch config to probably avoid OOM maybe
print(f'Setting PYTORCH_CUDA_ALLOC_CONF to max_split_size_mb:256 to possibly avoid fragmentation')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Login to WANDB. Needs env key WANDB_API_KEY
wandb_login()
device = torch.device('cuda:0')
model_name = 'Qwen/Qwen-14B'
output_dir = '/home/datta0/qwen-tiny-textbooks'
learning_rate = 3e-5
lr_scheduler_type = "cosine"
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
grad_acc_steps = 8
train_batch_size = 2
train_size = 100000
test_size = 1000
kwargs = {
    # 'cache_dir':'/home/datta0/.cache/huggingface/hub'
}




def get_model_and_tokenizer(lora_config):

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code = True, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, config = model_config, use_fast = False,trust_remote_code = True, **kwargs)
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config = model_config,
        torch_dtype = torch.bfloat16,
        device_map = 'auto',
        trust_remote_code = True,
        **kwargs
        # use_flash_attention_2 = True,
    )

    # Freeze Model Params  
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later

    model.gradient_checkpointing_enable()
    
    model = get_peft_model(model,lora_config)

    get_trainable_params(model)

    return tokenizer,model

def get_token_len(sample):
    sample['len_of_token'] = len(sample['input_ids'])
    return sample

def load_dataset_sorted(tokenizer):
    # Randomly shuffle and choose only 60k sample. We train on 50k and eval on 1k. Rest are buffer
    dataset = load_dataset('nampdn-ai/tiny-textbooks',split='train',cache_dir = '/home/datta0/.cache/huggingface/datasets').filter(lambda sample: sample['len'] > 200).shuffle().select(list(range(1,int(train_size*1.2))))
    dataset = dataset.map(lambda x:tokenizer(x['text']),batched = True).filter(lambda x:len(x['input_ids'])<=2048).map(get_token_len)
    cols = dataset.column_names
    cols.remove('input_ids')
    cols.remove('attention_mask')
    dataset = dataset.train_test_split(shuffle = True,train_size=train_size,test_size=test_size).sort('len_of_token').remove_columns(cols)
    train_data,eval_data = dataset['train'], dataset['test']
    assert 'input_ids' in train_data.column_names and 'attention_mask' in train_data.column_names
    
    return train_data,eval_data



def get_trainable_params(model):
    trainable_params = 0
    all_param = 0
    trainables = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainables.append((name,param.numel()))
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train_model():
    print('Training the model')
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha=lora_alpha,
        target_modules=["c_attn",'c_proj'],
        layers_to_transform=list(range(30,40)),
        lora_dropout=0.1,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    tokenizer,model = get_model_and_tokenizer(lora_config)
    train_data, eval_data = load_dataset_sorted(tokenizer)

    trainer = Trainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = eval_data,
        tokenizer = tokenizer,
        args = TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            gradient_checkpointing=True,
            warmup_steps=0.01,
            # max_steps=50, #20,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            bf16=True,
            logging_steps=0.02,
            output_dir=output_dir,
            optim=f"paged_adamw_32bit",
            save_steps = 0.02,
            save_total_limit=3,
            disable_tqdm = False,
            resume_from_checkpoint=True,
            remove_unused_columns=True,
            evaluation_strategy="steps",
            eval_steps = 0.02,
            # eval_accumulation_steps=1,
            per_device_eval_batch_size=train_batch_size,
            # load_best_model_at_end=True,
            report_to="wandb",
            run_name=output_dir
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer,pad_to_multiple_of=8, return_tensors="pt",mlm=False)
    )
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    print(model)

    with torch.cuda.amp.autocast(enabled=True, dtype = torch.bfloat16):
        trainer.train(resume_from_checkpoint = True)

    trainer.push_to_hub('qwen-text-neurips')
    return True


