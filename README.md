# NeurIPS LLM Efficiency Challenge.

This repo contains my submission for [neurIPS LLM efficiency challenge](https://llm-efficiency-challenge.github.io/).

## Info
- Base Model: [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)
- Adapters: [imdatta0](https://huggingface.co/imdatta0) [This submission trains multiple adapters and uses them depending on the task at hand.]
- dtype: bfloat16
- GPU/Track: A100
- Datasets: [nampdn-ai/tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)-50k, [OpenAssistant](OpenAssistant/oasst_top1_2023-08-25) ~13k, [jeopardy](https://huggingface.co/datasets/jeopardy) ~50k, [dolly](databricks/databricks-dolly-15k) -15k
- Training Samples: 125,000
- Eval Samples: 1000
- Approx Training time(if run): 20h

## How to Run
Note: If you want to run training as well, please set the env variable `TRAIN_MODEL=true` in [Dockerfile](./Dockerfile) (This currently only does training on tiny-books, adapter for the rest will be added soon)

To build the Image, run
```
docker build -f Dockerfile -t neurips_inference .
```

To start the server up and make it ready for inference, run
```
docker run -v --gpus "device=0" -p 8080:80 --rm -ti neurips_inference
```
This will start the server on port 8080.
Once the server is up, you can start sending requests via [HELM](https://github.com/stanford-crfm/helm/tree/main).