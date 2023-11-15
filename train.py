from train_helper import train_model
import os

os.environ['HUGGINGFACE_TOKEN']="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"
os.environ['WANDB_API_KEY']='1663281c6220a7c530453cbf8d51869cd0e95580'
os.environ['WANDB_PROJECT']='neurips_submission'
for dataset in ['databricks/databricks-dolly-15k','cnn_dailymail','OpenAssistant/oasst_top1_2023-08-25','jeopardy','nampdn-ai/tiny-textbooks']:
    print(f'Training for dataset {dataset}')
    train_model(dataset)
    print(f'Finshed training for dataset {dataset}')