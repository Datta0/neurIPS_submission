# NeurIPS LLM Efficiency Challenge.

This repo contains my submission for [neurIPS LLM efficiency challenge](https://llm-efficiency-challenge.github.io/).

## Info
- Base Model: [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)
- Adapter: [imdatta0/qwen-tiny-textbooks](https://huggingface.co/imdatta0/qwen-tiny-textbooks)
- dtype: bfloat16
- GPU/Track: A100
- Dataset: [nampdn-ai/tiny-textbooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)
- Training Samples: 100,000
- Eval Samples: 1000
- Approx Training time(if run): 9h

## How to Run
Note: If you want to run training as well, please set the env variable `TRAIN_MODEL=true` in [Dockerfile](./Dockerfile)

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