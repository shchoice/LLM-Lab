# Polyglot-12.8b 모델을 KoAlpaca 모델로 SFT 실험

## Overview
- [Polyglot-12.8b](https://huggingface.co/beomi/polyglot-ko-12.8b-safetensors) 및 [한국어로 번역한 Alpaca dataset](https://github.com/Beomi/KoAlpaca/blob/main/ko_alpaca_data.json)을 활용해서 Instruction Finetuning을 수행
- 해당 실험의 목적은 beomi/KoAlpaca-Polyglot-12.8B 모델을 재현해 보는 것을 목표로 함

## 기본개념

### Alpaca 란? [[출처]](https://docs.koalpaca.com/koalpaca/stanford-alpaca)

- InstructGPT와 같이, 언어 모델이 사용자의 Instuct (지시문)을 따르도록 하는 방법에 대한 연구
- Alpaca라고 하면 다음과 같은 두가지로 지칭할 수 있음
    - Alpaca Dataset : 52,000 개의 Instuct-(Input)-Output 쌍의 데이터셋
    - Alpaca Model : 위 데이터 셋으로 LLaMA 모델을 Finetune 한 모델

### Instruct Following 이란? [[출처]](https://docs.koalpaca.com/koalpaca/stanford-alpaca)

- 기존의 언어모델과 달리, 언어 모델이 사용자가 부여한 Instruct를 따르게 만들기 위해서는 새로운 형태의 데이터셋이 필요
- OpenAI의 InstructGPT에서는 다음과 같은 2가지 방법을 사용
    - Supervised Fine-Tuning(SFT) : 명령어 - 응답 셋으로 구성된 데이터셋으로 언어모델을 Finetune
    - RLHF(PPO) : 모델의 응답 중, 사람이 좀 더 선호하는 응답을 생성하도록 강화학습
- Alpaca는 SFT를 하기 위한 데이터셋을 제작하고 학습한 모델이며, 복잡한 RLHF/PPO 를 사용하지 않고서도 Instruct를 충분히 따를 수 있다는 것을 보여준 연구
    - [InstructGPT 대비 20%의 노력으로 80% 수준의 결과를 얻을 수 있음](https://www.aimlmag.com/unleashing-the-power-of-chatgpt-a-step-by-step-guide-in-alpaca-style-for-training-your-own-chatbot-part-1/)

## 준비사항

- Datasets
    - [Beomi/KoAlpaca](https://github.com/Beomi/KoAlpaca)
        - 한국어 Alpaca Dataset : [ko_alpaca_data.json](https://github.com/Beomi/KoAlpaca/blob/main/ko_alpaca_data.json)
        - 네이버 지식인 베스트 데이터 : [KoAlpaca_v1.1a.json](https://raw.githubusercontent.com/Beomi/KoAlpaca/main/KoAlpaca_v1.1.jsonl)
            - Beomi/KoAlpaca 에 따르면 기존 Alpaca 모델이 `대답을 짧게하는 경향` 및 `맥락을 이해하지 못하는 경향`을 개선하기 위해 제작했다고 전함
- Models:
  - Polyglot-12.8b 기반 [LoRA][PEFT]
    - A100 80GB 1대 (2대)

## Prompt
* baseline을 위한 실험으로 단순하게 구성
  ```python
    PROMPT_TEMPLATE = {
        "prompt_input": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
    아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n
    ### Instruction(명령어):\n{instruction}\n\n
    ### Input(입력):\n{input}\n\n
    ### Response(응답):\n""",
        
        "prompt_no_input": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n
    아래는 작업을 설명하는 명령어입니다. 명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n
    ### Instruction(명령어):\n{instruction}\n\n
    ### Response(응답):\n""",
        
        "response_split": "### Response(응답):"
    }
```
