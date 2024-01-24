# Meta LLaMA2-7B 모델을 KoAlpaca 모델로 SFT 실험

## Overview
- [LLaMA2 모델](https://github.com/facebookresearch/llama) 및 [한국어로 번역한 Alpaca dataset](https://github.com/Beomi/KoAlpaca/blob/main/ko_alpaca_data.json)을 활용해서 Instruction Finetuning을 수행
- 해당 실험의 목적은 LLaMA2 모델(Pretrain 시 충분하지 않은 한국어 데이터를 사용)에  한국어 능력의 성능이 개선된 KoALPACA 모델을 구현해냄을 목표로 함

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
  - Meta LLAMA 7B 기반 [LoRA][PEFT]
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
  

1. 프롬프트 선정

   ```
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

   

2. 파라미터 및 하이퍼 파라미터 선정

   * 모델 관련

     ```python
     BOS_TOKEN = "<s>"
     EOS_TOKEN = "</s>"
     UNK_TOKEN = "<unk>"
     IGNORE_INDEX = -100
     ```

   * Trainer 관련 파라미터

     ```python
     NUM_TRAIN_EPOCHS=2                                 # Training epochs
     TRAIN_BATCH_SIZE=16                                  # Training batch size
     EVAL_BATCH_SIZE=8                                   # Evaluation batch size
     EVALUATION_STRATEGY="steps"                         # Evaluation strategy
     EVAL_STEPS=100                                      # Evaluation steps
     SAVE_STEPS=100                                      # Save steps
     LOGGING_STEPS=100                                   # Logging steps
     LEARNING_RATE=5e-4                                  # Learning rate
     LR_SCHEDULER_TYPE="cosine"                          # LR scheduler type
     OPTIM="paged_adamw_8bit"                            # Optimizer type
     WARMUP_RATIO=0.1                                    # Warmup ratio
     WARMUP_STEPS=None
     WEIGHT_DECAY=0.05                                   # Weight decay
     GRADIENT_ACCUMULATION_STEPS=4                       # Gradient accumulation steps
     LOAD_BEST_MODEL_AT_END=True                         # Load best model at end
     FP16=True                                           # Use fp16
     DDP_FIND_UNUSED_PARAMETERS=False                    # DDP find unused parameters
     EARLY_STOPPING_PATIENCE=10                          # Early stopping patience
     ```

   * Quantization 관련 파라미터

     ```python
     LOAD_IN_4BIT=True                                   # Enable 4-bit quantization
     BNB_4BIT_QUANT_TYPE="nf4"                           # BNB 4-bit quantization type
     BNB_4BIT_COMPUTE_DTYPE=torch.bfloat16               # BNB 4-bit compute dtype
     BNB_4BIT_USE_DOUBLE_QUANT=True                      # BNB 4-bit use double quantization
     ```

   * LoRA 관련 파라미터

     ```python
     R=8                                                 # Lora attention dimension
     LORA_ALPHA=16                                       # Lora alpha parameter
     LORA_DROPOUT=0.05                                   # Lora dropout probability
     FAN_IN_FAN_OUT=False                                # Lora fan in fan out
     BIAS="none"                                         # Lora bias type
     TARGET_MODULES=["q_proj", "v_proj"]                 # Lora target modules
     INFERENCE_MODE=False                                # Inference mode
     TASK_TYPE="CAUSAL_LM"                               # Task type
     ```

   * Tokenizer 관련 파라미터

     ```python
     MAX_LENGTH=2048                                     # Max sequence length for tokenizer
     TRUNCATION=True                                     # Enable/disable truncation
     RETURN_OVERFLOWING_TOKENS=True                      # Return overflowing tokens info
     RETURN_LENGTH=True                                  # Return length of encoded inputs
     PADDING=True                                        # Enable padding to max sequence length
     PADDING_SIDE="right"                                # The side on which the model should have padding applied
     ```

3. 데이터셋에 토크나이징 후 데이터의 길이 기본 통계 정보

   ```
   Average Length: 244.09052631578948
   Max Length: 2876
   Min Length: 1
   Median Length: 150
   ```



4. Dataset 토크나이저 및 DataCollator 처리 코드 선정

   * Standford Alpaca 에서 처리한 방법을 차용함

     ```python
     class SupervisedDataset(Dataset):
         """Dataset for supervised fine-tuning."""
     
         def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
             super(SupervisedDataset, self).__init__()
             logging.warning("Loading data...")
             list_data_dict = jload(data_path)
     
             logging.warning("Formatting inputs...")
             prompt_input, prompt_no_input = PROMPT_TEMPLATE["prompt_input"], PROMPT_TEMPLATE["prompt_no_input"]
             sources = [
                 prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                 for example in list_data_dict
             ]
             targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
     
             logging.warning("Tokenizing inputs... This may take some time...")
             print("tokenizer", tokenizer)
             data_dict = preprocess(sources, targets, tokenizer)
     
             self.input_ids = data_dict["input_ids"]
             self.labels = data_dict["labels"]
     
         def __len__(self):
             return len(self.input_ids)
     
         def __getitem__(self, i) -> Dict[str, torch.Tensor]:
             return dict(input_ids=self.input_ids[i], labels=self.labels[i])
     ```

     ``` python
     def preprocess(
         sources: Sequence[str],
         targets: Sequence[str],
         tokenizer: transformers.PreTrainedTokenizer,
     ) -> Dict:
         """Preprocess the data by tokenizing."""
         examples = [s + t for s, t in zip(sources, targets)]
         examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
         input_ids = examples_tokenized["input_ids"]
         labels = copy.deepcopy(input_ids)
         for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
             label[:source_len] = LLAMA_IGNORE_INDEX
         return dict(input_ids=input_ids, labels=labels)
     ```

     ```python
     def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
         """Tokenize a list of strings."""
         tokenized_list = [
             tokenizer(
                 text,
                 return_tensors="pt",
                 padding=PADDING,
                 max_length=MAX_LENGTH,
                 truncation=True,
             )
             for text in strings
         ]
         input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
         input_ids_lens = labels_lens = [
             tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
         ]
         return dict(
             input_ids=input_ids,
             labels=labels,
             input_ids_lens=input_ids_lens,
             labels_lens=labels_lens,
         )
     ```

5. DataCollator 

   * 역할
     * 데이터 샘플들을 효율적인 배치 처리를 위함
       * 입력된 배치 input들에 패딩 처리 수행
         * 다양한 길이의 입력 데이터를 일관된 형태의 배치로 변환하여 학습에 적합한 형태로 변환

   * 코드 설명

     ```python
     @dataclass
     class DataCollatorForSupervisedDataset(object):
         """Collate examples for supervised fine-tuning."""
     
         tokenizer: transformers.PreTrainedTokenizer
     
         def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
             input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
             input_ids = torch.nn.utils.rnn.pad_sequence(
                 input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
             )
             labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=LLAMA_IGNORE_INDEX)
             return dict(
                 input_ids=input_ids,
                 labels=labels,
                 attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
             )
     ```

     

     * padding 처리 : 입력 시퀀스(`input_ids`)와 레이블(`labels`)를 배치의 가장 긴 시퀀스에 맞추기 위해 패딩 처리

       * `torch.nn.utils.rnn.pad_sequence` 함수를 사용하여 모든 시퀀스를 동일한 길이로 맞춥

     * Attention Mask를 생성

       cf) `attention_mask` 란 : 모델이 실제 데이터와 패딩된 부분을 구분할 수 있도록 함,  입력 시퀀스의 각 위치가 실제 데이터인지(값이 `True`) 아니면 패딩된 부분인지(값이 `False`)를 나타냄




## 평가 결과

### 일반 상식 실험

llama 가 pretrain된 내용을 바탕으로 답변을 잘해주는 것으로 판단됨

1. 샘플 01

    * 질의

        ```python
        gen("""
        파이썬 공부를 하고 싶습니다. 파이썬을 처음 공부해봅니다. 어떻게 공부를 시작하면 좋을까요?
        """)
        ```

    * 답변

        ```
        파이썬에 대한 기본적인 지식은 약간의 웹 개발 경험과 함께 쉽게 익히실 수 있습니다. 그러나 완전히 파이썬을 익힐 때는 직접 문제를 해결하는 방법을 사용하는 것이 가장 좋습니다. 즉, 문제를 해결하는 동안 자신이 무엇을 어떻게 해야하는지 생각하는 것이 중요합니다. 이러한 방법들은 여러 가지가 있습니다. 예를 들어, 문제를 해결하는 동안 코드를 작성하는 것은 좋은 방법입니다. 또한, 문제를 해결하는 동안 다른 사람들이 해결하는 방법을 보는 것도 유용합니다. 이러한 방법들은 모든 문제를 해결하는 것보다는 일부 문제를 해결하는 것을 위주합니다. 따라서 파이썬을 공부하는 동안 문제를 해결하는 방법을 찾아내는 것이 중요합니다.
        ```

2. 샘플 02

   * 질의

     ```python
     gen(
         instruction='컴퓨터 공학과의 일반적인 커리큘럼은 어떻게 되나요?',
         input='답변을 할 때, 2개의 문장으로 작성해주세요.'
     )
     ```

   * 답변

     ```
     컴퓨터 공학과에서는 기초 수업부터 전공 수업까지 진행됩니다. 기초 수업에서는 C/C++를 비롯한 프로그래밍 언어를 배우며, 전공 수업에서는 데이터베이스, 소프트웨어 시스템, 운영체제 등 여러 전공 분야를 선택할 수 있습니다.
     ```

### 단순 코드 계산 예제

llama 가 pretrain된 내용을 바탕으로 답변을 잘해주는 것으로 판단됨, 샘플 02에 대해 이번 답변은 틀렸지만, 같은 질문에서 여러 실험 결과로는 올바른 결과를 냈었음

1. 샘플 01

   * 질의

     ```python
     gen("""
     파이썬을 통해 1부터 10까지의 총합을 구하는 코드를 작성해주세요
     """)
     ```

   * 답변

     ````text
     ```python
     sum = 0
     for i in range(1, 11):
        sum += i
     print("The sum of numbers from 1 to 10 is:", sum)
     ```
     ````

2. 샘플 02

   * 질의

     ``` python
     gen("""
     자바 언어를 통해 1부터 15까지 곱셈하는 코드를 작성해주세요
     """)
     ```

   * 답변

     ````text
     ```java
     int i = 1;
     for (i = 2; i <= 15; i++) {
        System.out.println("" + i);
     }
     ```
     ````

### 요약 예제

Llama2의 pretrain 된 내용으로 인해 요약도 훌륭하게 해줌

1. 샘플 01

   * 질의

     ```python
     gen("""
     아래와 같은 기사가 있습니다. 기사의 핵심 내용을 추려서 간단하게 요약해주세요.
     """,
     """
     토트넘(잉글랜드)의 3연승을 이끈 '캡틴' 손흥민(31)이 맨 오브 더 매치(MOM)에 선정됐다. 
     손흥민은 크리스마스 이브인 24일(한국 시각) 영국 런던의 토트넘 홋스퍼 스타디움에서 열린 에버턴과 2023-2024시즌 잉글랜드 프리미어리그(EPL) 18라운드 홈 경기에서 팀의 두 번째 골을 터뜨렸다. 
     토트넘은 손흥민의 득점에 힘입어 2 대 1 승리를 거뒀다. 
     이로써 토트넘은 3연승 행진을 이어갔고, 11승 3무 4패 승점 36을 기록했다. 한 경기를 덜 치른 맨체스터 시티(승점 34)를 제치고 4위로 올라섰다.
     손흥민은 이날도 왼쪽 측면 공격수로 나섰다. 지난 16라운드 뉴캐슬전(1골 2도움)부터 왼쪽 측면에서 최고 윙어의 면모를 유감없이 발휘하고 있다. 
     17라운드 노팅엄 포레스트전에서는 공격 포인트를 올리지 못했으나, 이날 2경기 만에 다시 득점포를 가동했다.
     리그 11호 골을 터뜨린 손흥민은 무함마드 살라흐(리버풀), 재러드 보웬(웨스트햄)과 나란히 득점 공동 3위에 올랐다. 
     1위는 14골의 엘링 홀란(맨체스터 시티), 2위는 12골의 도미닉 솔란케(본머스)다.
     또 손흥민은 리그 반환점을 1경기 남겨둔 시점에서 벌써 지난 시즌 득점 기록을 넘어섰다. 
     스포츠 탈장 부상 여파로 고전했던 지난 시즌에는 10골 6도움을 기록했다. 
     도움 4개를 기록 중인 그는 지난 시즌 공격 포인트 기록 돌파도 눈앞에 두고 있다.
     EPL 통산 득점 랭킹에서는 아스널의 레전드 이안 라이트(113골)을 넘어섰다. 
     114골로 단독 23위에 오른 손흥민은 120골로 공동 21위인 라힘 스털링(첼시), 스티븐 제라드를 6골 차로 쫓고 있다.
     손흥민은 경기 후 EPL 사무국이 22947명의 팬을 상대로 진행한 투표에서 67.7%의 압도적인 지지를 받아 MOM에 오르는 영예를 안았다. 
     팀 동료인 굴리엘모 비카리오(15.1%), 페드로 포로(7.8%) 등을 크게 따돌렸다.
     풀타임을 뛴 손흥민은 1골을 포함해 슈팅 2회, 패스 성공률 71%(24/34), 기회 창출 1회, 볼 터치 56회, 드리블 성공 43%(3/7) 등을 기록했다. 
     축구 통계 매체 '풋몹'은 손흥민에게 팀 내 4번째로 높은 평점 7.8을 부여했다.
     """)
     ```

   * 답변

     ```
     손흥민이 맨 오브 더 매치에 선정된 것은 자신의 좋은 활약에 따른 것이며, 손흥민은 최근 3연승을 이끌었다. 손흥민은 왼쪽 측면 공격수로 나섰다. 손흥민은 맨체스터 시티와 본머스 등 강력한 라이벌들과 경합하며 많은 경기를 뛰고 있다.
     ```

2. 샘플 02

   * 질의

     ```python
     gen(
         instruction='아래와 같은 문장들이 있습니다. 답변을 할때 이 문장들을 2개의 문장으로 요약해주세요. 문장이 2개라는 것은 개행문자가 2개인 것을 의미합니다.',
         input=
     """
     FLAN (Fine-tuned LAnguage Net) 모델은 자연어 처리(NLP) 과제들을 해결하기 위해 'instruction tuning'이라는 기법을 사용하는 언어 모델입니다.
     이 모델의 핵심 아이디어는 다양한 NLP 과제를 자연어 지시사항 형태로 변형하여 이러한 과제들을 풀도록 fine-tuning하는 것입니다.
     이를 통해 FLAN 모델은 번역, 상식 추론, 감정 분류 등을 포함한 다양한 NLP 과제를 수행할 수 있도록 fine-tuning 됩니다​.
     
     FLAN의 연구 결과에 따르면, 이 모델은 zero-shot 시나리오에서 GPT-3보다 우수한 결과를 보였으며, 많은 task에서는 supervised model과 비슷한 성능을 달성했습니다.
     특히 자연어 추론(NLI)과 질문응답(QA) 작업에서 효과적이었습니다.
     Google Research Blog에서는 FLAN이 언어 모델을 사용하여 특정 실제 과제에 대한 지식을 어떻게 풀어내는지에 대해 설명합니다.
     전통적으로는 레이블이 붙은 데이터셋을 이용해 fine-tuning하는 방법이 많이 사용되었지만, FLAN은 다양한 종류의 지시사항에 대해 모델을 fine-tuning함으로써, 특정 과제가 아닌 일반적인 NLP 과제들을 해결할 수 있게 만듭니다.
     """)
     ```

   * 답변

     ```
     FLAN은 자연어 처리(NLP) 과제들을 해결하기 위해 'instruction tuning'이라는 기법을 사용하는 언어 모델입니다. 이를 통해 FLAN 모델은 번역, 상식 추론, 감정 분류 등을 포함한 다양한 NLP 과제를 수행할 수 있도록 fine-tuning 됩니다.
     ```

### One-shot 예제

새로운 개념에 대해서는 pretrain된 내용이 없기에 답변을 못하나 외부 지식을 주입해주면 답변을 잘함

1. 샘플 01

   * 질의

     ```  python
     gen("""
     솔트룩스 회사의 루시아 솔루션에 대해서 설명해주세요.
     """)
     ```

   * 답변

     ```
     Lucid Solutions은 미국 워싱턴 D.C. 지역에 위치한 소프트웨어 개발 회사로, 2014년 설립되었습니다. 회사는 정부 기관과 비영리 단체들에게 전자 문서 관리 시스템인 Lucid Viewer를 제공합니다. 이 시스템은 정부 기관과 비영리 단체들에게 전자 문서 관리 시스템인 Lucid Viewer를 제공합니다.
     ```

   * 수정된 질의

     ````python
     gen("""
     질문 : 솔트룩스 회사의 루시아 솔루션에 대해서 설명해주세요. 솔트룩스 및 루시아에 대한 정보를 External Generated Knowledge로 드립니다.
     
     ```
     인공지능 기업 솔트룩스(대표 이경일)는 ‘LUXIA Is All You Need–생성 AI 시대, 모든 것이 달라집니다'를 주제로 7일, 서울 코엑스 오디토리움에서 AI 컨퍼런스 'SAC 2023'을 개최했다.
     
     이날 행사에서는 솔트룩스가 올해 초 금융권 컨퍼런스를 통해 처음 공개한 초거대 언어모델(LLM) ‘루시아(LUXIA)’와 그 생태계가 본격적으로 소개됐다. 온오프라인 하이브리드 형태로 개최된 이번 행사의 사전 참가 신청자는 약 2,700명에 달했다.
     
     솔트룩스 이경일 대표는 키노트를 통해 도서 420만 권 분량을 학습한 자체 GPT 모델에 실시간 정보와 전문 지식을 활용해 환각 현상을 획기적으로 감소시킨 루시아GPT와 다양한 노코드 도구들을 선보였다. 또한 “처음 설립되었을 때 세상 모든 사람이 자유롭게 지식 소통하는 세상을 만들겠다는 미션을 가지고 있었던 솔트룩스는, 이제 오직 사람을 위한 인공지능 사람만을 위한 루시아GPT를 만들고 이를 통해 또 다른 성장을 시작하고자 한다”라고 말했다.  
     
     다양한 환경에 맞춤형으로 도입 가능한 똑똑한 루시아(LUXIA)
     
     루시아는 AI 데이터 구축 관련 정부 사업뿐 아니라 특허청, 행정안전부 등 다양한 분야의 사업을 수행하며 솔트룩스가 축적해 온 한글 데이터를 약 1TB 이상 학습했다. 이에 데이터 저작권 이슈를 최소화할 뿐 아니라 법률, 특허, 금융, 교육 등 각 전문 분야에 최적화된 맞춤형 언어모델을 빠르고 안전하게 구축할 수 있다. 
     
     챗GPT 등 생성 AI의 고질적인 문제로 지적된 환각 현상을 최소화하기 위해서 ‘지식그래프(Knowledge Graph)를 활용한 사실/지식 그라운딩(Factual Grounding)’과 ‘검색 증강 생성(RAG·Retrieval-Augmented Generation)’이라는 2가지 접근법을 연계했다.
     
     솔트룩스 김재은 랩장은 자사 지식그래프와 연계하여 자체 연구·개발한 인스트럭트 지식 학습(IKL·Instruct Knowledge Learning)을 통해 오픈AI의 ‘GPT-3.5’ 및 메타의 ‘라마(Llama)2’와 대비했을 때, 한국어 할루시네이션 자체 평가에서 대략 40% 더 우수한 성능을 확인할 수 있었다고 밝혔다. 
     
     출처 : 인공지능신문(https://www.aitimes.kr)
     ```
     """)
     ````

   * 답변

     ```
     솔트룩스는 초거대 언어모델 LLM을 개발하고 있다. 이 언어모델은 생성 AI 시대에 적합한 모델이라고 주장한다. 솔트룩스는 이 언어모델을 통해 데이터 저작권 문제를 해결하고, 법률, 특허, 금융, 교육 등 다양한 분야에 최적화된 맞춤형 언어모델을 빠르고 안전하게 구축할 수 있다고 주장한다.
     ```

2. 샘플 02

   * 질의

     ``` python
     gen("""
     슈카월드가 뭐에요
     """)
     ```

   * 답변

     ```
     슈카월드는 플랫폼 기반의 모바일 게임입니다.
     ```

   * 수정된 질의

     ```python
     gen("""
     질문 : 슈카월드가 뭐에요. 슈카월드 대한 정보를 External Generated Knowledge로 드립니다.
     슈카(본명: 전석재, 1979년~)는 대한민국의 유튜버이다. 증권 펀드매니저 출신으로, 경제 전문 1인 방송 슈카월드를 운영하고있다.
     매주 일요일 저녁, 슈카월드 유튜브 채널에서 라이브 방송을 진행한다
     """)
     ```

   * 답변

     ```
     슈카월드는 경제 전문 1인 방송으로, 생산자들과 소비자들의 상호작용을 중심으로 경제 현장을 알리고 살피기 위해 만들어진 채널입니다. 슈카월드는 2015년 4월 30일에 개설되었으며, 현재 160만 명의 구독자를 보유하고 있습니다.
     ```

     

 ### CoT 예제

수학적으로 사고하는 능력은 매우 약해보임, 모델의 성능을 늘리기 위해서는 CoT 예제 및 논리와 관련된 데이터셋을 통해 추가 학습을 해야할 것 같음

1. 샘플 01

   * 질의

     ```  python
     gen("""
     한 반에 30명의 학생이 있습니다. 그 중 3분의 2가 소녀입니다. 소년은 몇 명입니까?
     """)
     ```

   * 답변

     ```
     15명
     ```

   * 수정된 질의

     ```python
     gen(
         instruction = '한 반에는 소년과 소녀로 이루어져 있으며, 총 60명의 학생이 있습니다. 그 중 3분의 2가 소녀입니다. 소년은 몇 명입니까? input의 예제를 보고 논리적으로 분해하고 생각을 하여 올바른 답변을 해주세요',
         input=
     """
     예제1)
     먼저, 반 전체 학생 수인 24명 중에서 소녀의 비율을 계산해야 합니다. 소녀는 전체의 4분의 3에 해당합니다.
     24명을 4로 나누면 각 그룹에 몇 명이 있는지 알 수 있습니다. 24를 4로 나누면 6명입니다.
     이제 이 수를 3배하면 소녀의 수를 알 수 있습니다. 6명의 3배는 18명입니다. 그러므로 소녀는 18명입니다.
     전체 학생 수에서 소녀의 수를 빼면 소년의 수를 알 수 있습니다. 24명에서 18명을 빼면 6명이 남습니다.
     따라서 이 반에는 소년이 6명 있습니다.
     Response: 6명
     
     예제2)
     먼저, 반 전체 학생 수인 28명 중에서 소녀의 비율을 계산해야 합니다. 소녀는 전체의 3분의 2에 해당합니다.
     28명을 3으로 나누면 각 그룹에 몇 명이 있는지 알 수 있습니다. 28을 3으로 나누면 9.33, 즉 약 9명입니다.
     이제 이 수를 2배하면 소녀의 수를 알 수 있습니다. 9명의 2배는 18명입니다. 그러므로 소녀는 18명입니다.
     전체 학생 수에서 소녀의 수를 빼면 소년의 수를 알 수 있습니다. 28명에서 18명을 빼면 10명이 남습니다.
     따라서 이 반에는 소년이 10명 있습니다.
     Response: 10명
     
     예제3)
     먼저, 도서관 전체 책의 수인 30권 중에서 과학 책의 비율을 계산해야 합니다. 과학 책은 전체의 5분의 4에 해당합니다.
     30권을 5로 나누면 각 그룹에 몇 권이 있는지 알 수 있습니다. 30을 5로 나누면 6권입니다.
     이제 이 수를 4배하면 과학 책의 수를 알 수 있습니다. 6권의 4배는 24권입니다. 그러므로 과학 책은 24권입니다.
     전체 책의 수에서 과학 책의 수를 빼면 문학 책의 수를 알 수 있습니다. 30권에서 24권을 빼면 6권이 남습니다.
     따라서 도서관에는 문학 책이 6권 있습니다.
     Response : 6권
     
     그렇다면 instruction의 정답은??
     """)
     ```

   * 답변

     ```
     소년은 6명, 소녀는 18명입니다
     ```

     