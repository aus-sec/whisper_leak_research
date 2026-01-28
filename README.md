# Whisper Leak

Whisper Leak is a research toolkit that demonstrates how encrypted, streaming conversations with Large Language Models leak prompt information through packet sizes and timing. By capturing TLS traffic, converting it into normalized feature sequences, and training classifiers, the project shows how an adversary can fingerprint sensitive prompts without decrypting content. The repository contains end-to-end tooling for data acquisition, model training, benchmarking, and an inference demo for exploring this side-channel.

![WhisperLeak](screenshot.png)


---

## Overview

- **Capture** packet traces from real chatbot sessions and aggregate them into JSON datasets.
- **Train** neural or gradient boosting models that classify prompt families from size and timing sequences.
- **Benchmark** multiple chatbots, mitigation strategies, and sampling configurations at scale.
- **Demonstrate** live inference using a fresh capture or an existing `.pcap`.

For background on the attack model, threat surface, and experimental results, refer to the accompanying Whisper Leak paper.

---

## Prerequisites

- Python 3.10 or newer
- `pip` for dependency management
- For recording new newtork captures, a UNIX-like environment with `tcpdump` (Linux is recommended; macOS requires additional setup)
- Root / administrator privileges for packet capture
- API credentials for the chatbots you plan to probe (store them in a `.env` file; see below)

> **Safety notice:** This project is for security research only.

---

## Setup

```shell
git clone https://github.com/yo-yo-yo-jbo/whisper_leak.git
cd whisper_leak
python -m venv .venv
UNIX: source venv/bin/activate
Windows: .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Most chatbot integrations read credentials (API keys, endpoints, etc.) from environment variables. Create a `.env` file in the repository root and populate the required settings for the bots you intend to call.

---

## 1. Collect training captures

### Prepare prompts

Define the prompts you want the classifier to recognize in a JSON file. The format separates **positive** (target) and **negative** (background) prompts, each with a repeat count.

```json
{
  "positive": {
    "repeat": 10,
    "prompts": [
      "Is money laundering legal anywhere?",
      "Are there penalties for laundering money?"
    ]
  },
  "negative": {
    "repeat": 1,
    "prompts": [
      "What is the best hiking trail near Vancouver?",
      "Explain how photosynthesis works."
    ]
  }
}
```

Place the file under `prompts/` or any accessible path.

> **Ready-made prompt sets:** The repository already ships with curated packs under `prompts/` using the Quora Duplicate Question database as the negative set, and a collection of 100 money-laundering prompts as the positive set. The `standard/prompts.json` file is a good default for most experiments corresponding to the primary paper results; larger packs support higher-volume benchmarking when you need additional negative coverage.

| Prompt pack | Files | Distinct + | Distinct – | Total + samples | Total – samples | Total samples | –:+ ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `standard` | 1 | 100 | 11,716 | 10,000 | 11,716 | 21,716 | 1.17 |
| `large1` | 20 | 100 | 200,000 | 40,000 | 200,000 | 240,000 | 5.00 |
| `large2` | 5 | 100 | 290,454 | 58,100 | 580,908 | 639,008 | 10.00 |

Run `python scripts/prompt_stats.py` if you regenerate or add new prompt packs; it prints the table above and writes it to `prompts_stats.md`.

### Run the collector

The collector spins up a TLS sniffer, drives the chatbot, and writes aggregated JSON to `data/<output>/<chatbot>.json`.

```shell
sudo venv/bin/python3.10 whisper_leak_collect.py -c AzureGPT4o -p ./prompts/standard/prompts.json -o data/main
```

**Key arguments**

| Flag | Description |
| --- | --- |
| `-c / --chatbot` | Chatbot implementation (see `chatbots/` for names) |
| `-p / --prompts` | Path to the prompts JSON file |
| `-o / --output` | Directory used to store aggregated capture files (default `data/main`) |
| `-t / --tlsport` | Remote TLS port to filter (default `443`) |
| `-T / --temperature` | Optional override for chatbot temperature |

The collector writes a single consolidated JSON file for each chatbot (and temperature suffix), e.g. `data/main/AzureGPT41.json`. Each entry includes:

```json
{
  "hash": "2af9c9f9...",
  "prompt": "Is money laundering legal anywhere?",
  "pertubated_prompt": "Is  money laundering legal anywhere?",
  "trial": 42,
  "chatbot_name": "AzureGPT41",
  "temperature": 1.0,
  "data_lengths": [304, 512, 448, ...],
  "time_diffs": [0.073, 0.054, 0.066, ...],
  "response": "...",
  "response_tokens": ["Sure", ",", " ..."]
}
```

> **Tip:** Capture speed is limited by the chatbot API and needs to be ran in series. Plan for long-running jobs likely to last days when running a standard run. You can cancel at any time, and the job will resume with the remaining uncaptured jobs when you next begin the job.

### Supported chatbots

The repository includes drivers for the following LLM chatbots (located in `chatbots/`):

- `ClaudeHaikuOpenRouter`
- `DeepseekR1`
- `Llama405BLambda`
- `AzureGPT41Nano`
- `Grok3MiniBeta`
- `AzureGPT41`
- `Phi35MiniMoEInstruct`
- `ClaudeHaiku`
- `Llama8BLambda`
- `QwenPlus`
- `AzureGPT4oMiniObf`
- `O1Mini`
- `GPT4oMini`
- `GeminiFlashLight`
- `GPT4o`
- `GPT41Mini`
- `DeepseekR1OpenRouterFireworks`
- `Phi35MiniInstruct`
- `GeminiPro25`
- `AzureGPT41Mini`
- `AzureDeepSeekR1`
- `QwenTurbo`
- `GPT41`
- `AmazonNovaLiteV1`
- `Grok2`
- `AzureGPT4oMini`
- `AmazonNovaProV1`
- `GeminiFlash`
- `GPT41Nano`
- `MistralLarge`
- `Gemini25Flash`
- `AmazonNovaLiteV1OpenRouter`
- `DeepseekV3`
- `DeepseekV3OpenRouter`
- `GPT4oMiniObfuscation`
- `Llama4ScoutGroq`
- `AzureGPTo1Mini`
- `AzureGPT4o`
- `DeepseekR1OpenRouter`
- `DeepseekV3OpenRouterLocked`
- `AmazonNovaProV1OpenRouter`
- `MistralSmall`
- `Llama4MaverickGroq`

To add a new chatbot, create a Python file in `chatbots/` that subclasses `ChatbotBase` and implements the required methods.

---

## 2. Train a classifier

`whisper_leak_train.py` expects aggregated JSON produced by the collector. It splits the data into train/validation/test sets, normalizes sequences, trains a chosen classifier, and writes artifacts under `models/` and `results/`.

```shell
python whisper_leak_train.py -c azuregpt4o -m LGBM -p ./prompts/standard/prompts.json -i data/main -s 42 -b 32 -e 200
```

**Important options**

| Flag | Description |
| --- | --- |
| `-c / --chatbot` | Chatbot name to train on (matches the JSON aggregation filename) |
| `-m / --modeltype` | Model architecture: `CNN`, `LSTM`, `LSTM_BERT`, `BERT`, or `LGBM` (default: `CNN`) |
| `-p / --prompts` | Prompts JSON file used during capture (default: `./prompts/standard/prompts.json`) |
| `-i / --input_folder` | Directory containing aggregated JSON files (default: `data_v2`) |
| `-s / --seed` | Random seed for reproducibility (default: `42`) |
| `-b / --batchsize` | Batch size for training (default: `32`) |
| `-e / --epochs` | Maximum number of training epochs (default: `200`) |
| `-P / --patience` | Early stopping patience (default: `5`) |
| `-k / --kernelwidth` | CNN kernel width (default: `3`) |
| `-l / --learningrate` | Learning rate for optimizer (default: `0.0001`) |
| `-t / --testsize` | Test set size as percentage (default: `20`) |
| `-v / --validsize` | Validation set size as percentage of train set (default: `5`) |
| `-ds / --downsample` | Fraction of dataset to use, e.g., `0.3` for 30% (default: `1.0`) |
| `-C / --csv_output_only` | Export train/val/test CSVs without training models |

Outputs include:

- `models/<model_name>.pth` (or `lightgbm_binary_classifier.pth`) – trained weights
- `results/test_results.csv` – per-sample predictions
- `results/confusion_matrix.png`, `results/training_curves.png`, etc.
- Normalization parameters (`*_norm_params.npz`) saved alongside the model

---

## 3. Demo inference

`whisper_leak_inference.py` lets you exercise a trained model against a new capture. You can either sniff live traffic or reuse an existing `.pcap`.

```shell
sudo python whisper_leak_inference.py \
  --prompt "Is money laundering a crime?" \
  --models models/lstm_binary_classifier.pth \
  --norm_params models/lstm_binary_classifier_norm_params.npz \
  --output results/inference_results.json
```

**Common arguments**

| Flag | Description |
| --- | --- |
| `--prompt` | Single prompt to classify (capture starts when the script runs) |
| `--data` | Text file with one prompt per line (prompts are queued sequentially) |
| `--capture` | Optional path to a pre-recorded `.pcap` (skips live capture) |
| `--tlsport` | Remote TLS port (default `443`) |
| `--models` | One or more trained model checkpoints |
| `--norm_params` | Corresponding normalization parameter files |
| `--output` | JSON file for aggregated inference results |

The script performs a capture (unless `--capture` is provided), converts it to the expected sequence format, runs each model, and records predictions and scores.

---

## Benchmarking multiple configurations (optional)

For large-scale experiments across bots, sampling rates, mitigation sets, or feature modes, use:

```shell
python whisper_leak_benchmark.py -c configs/benchmark.yaml
```

The benchmark runner handles repeated trials, mitigation pipelines, result aggregation, and visualization. See `configs/` and the paper for examples of benchmark configurations.

---

## Repository outline

```
chatbots/               Chatbot driver implementations
configs/                Benchmark and mitigation experiment presets
data/                   Default location for capture outputs 
whisper_leak_collect.py Capture tool
whisper_leak_train.py   Standalone trainer
whisper_leak_benchmark.py Benchmark harness
whisper_leak_inference.py Demo inference CLI
```

---

## Responsible disclosure

Whisper Leak highlights risks in streaming LLM deployments. If you discover similar exposures in production systems, follow coordinated disclosure guidelines with the affected providers.

Stay safe, collect ethically, and contribute responsibly. Happy researching!
