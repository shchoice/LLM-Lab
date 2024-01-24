# LLM-Lab

## Overview
* LLM 모델들에 대한 실험 결과들을 기록하는 Repository 입니다.
* 쉽게 보기 위해 Jupyter Notebook 형태로 코드를 기록합니다.

## Description
* llama2_to_koalpaca : meta-llama/Llama2-7b-hf 모델에 KoALPACA 데이터를 바탕으로 Finetuning을 하여 LLaMA2 모델의 한국어 이해가 가능한 지에 대한 실험
* polyglot-ko_to_koalpaca : beomi/polyglot-ko-12.8b-safetensors 모델에 KoALPACA 데이터를 바탕으로 Finetuning을 하여 beomi/KoALPACA 를 재현해 보는 실험
* beomi_koalpaca : 별도의 Finetuning을 하지 않은 beomi/KoAlpaca-Polyglot-12.8B 모델에서 몇 가지 질문을 넣었을 때 어떻게 답변을 하는지 확인
