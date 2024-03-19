import os  # os 모듈 운영체제와 상호 작용할 수 있는 기능을 제공
import torch # PyTorch 라이브러리로, 주로 딥러닝과 머신러닝 모델을 구축, 학습, 테스트하는 데 사용
from datasets import load_dataset  # 데이터셋을 쉽게 불러오고 처리할 수 있는 기능을 제공
from transformers import (
    AutoModelForCausalLM, # 인과적 언어 추론(예: GPT)을 위한 모델을 자동으로 불러오는 클래스
    AutoTokenizer, # 입력 문장을 토큰 단위로 자동으로 잘라주는 역할
    BitsAndBytesConfig, # 모델 구성
    HfArgumentParser,  # 파라미터 파싱
    TrainingArguments,  # 훈련 설정
    pipeline,  # 파이프라인 설정
    logging,  #로깅을 위한 클래스
)
# 모델 튜닝을 위한 라이브러리
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

def training(model_name, plus_epoch_model, load_dataset_path, epochs, steps):
    # 데이터셋 로드
    dataset = load_dataset(load_dataset_path, split="train")

    # 훈련 데이터셋을 훈련 세트와 검증 세트로 분할
    train_dataset, validation_dataset = dataset.train_test_split(test_size=0.2).values()

    # 모델 및 토크나이저 로드 (QLoRA 설정 적용)
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        # 모델을 4비트로 로드할지 여부를 결정
        load_in_4bit=True,  
        # 양자화 유형을 설정
        bnb_4bit_quant_type="nf4", 
        # 계산에 사용될 데이터 타입을 설정
        bnb_4bit_compute_dtype=compute_dtype,  
        # 중첩 양자화를 사용할지 여부를 결정
        bnb_4bit_use_double_quant=False, 
    )

    # 만약 GPU가 최소한 버전 8 이상이라면 (major >= 8) bfloat16을 지원한다고 메시지를 출력.
    # bfloat16은 훈련 속도를 높일 수 있는 데이터 타입.
    if compute_dtype == torch.float16 and True:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # 기본 모델 로드 (QLoRA 적용 전)
    model = AutoModelForCausalLM.from_pretrained(
        # 로드할 모델 이름
        model_name,  
        # QLoRA 설정
        quantization_config=bnb_config,  
        # 모델을 로드할 장치 지정
        device_map={"": 0}  
    )
    # 캐싱 사용 여부 (False로 설정)
    model.config.use_cache = False
    # 캐싱은 디코딩 속도를 높일 수 있지만, 메모리 사용량을 증가시킬 수 있습니다. 일반적으로 메모리가 부족하거나 모델 속도가 중요하지 않은 경우 False로 설정합니다.

    # 사전 훈련 토큰 위치 (1로 설정)
    model.config.pretraining_tp = 1
    # 사전 훈련 모델에서 토큰 위치 정보를 사용하면 모델 성능을 향상시킬 수 있지만, 모델 크기가 증가할 수 있습니다. 일반적으로 모델 크기가 중요하지 않거나 모델 성능을 향상시키는 것이 중요한 경우 1로 설정합니다.

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 동일한 batch 내에서 입력의 크기를 동일하기 위해서 사용하는 Padding Token을 End of Sequence라고 하는 Special Token으로 사용한다.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training. Padding을 오른쪽 위치에 추가한다.

    # QLoRA 설정 로드
    peft_config = LoraConfig(
        # LoRA 알파 값 (16으로 설정)
        lora_alpha=16,  
        # LoRA 드롭아웃 비율 (0.1로 설정)
        lora_dropout=0.1,  
        # LoRA 랭크 (64로 설정)
        r=64,  
        # LoRA 바이어스 설정 ("none"으로 설정)
        bias="none",  
        # 파인튜닝할 태스크 ("CAUSAL_LM"으로 설정)
        task_type="CAUSAL_LM",  
        # task_type: 파인튜닝할 태스크를 지정하는 파라미터. "CAUSAL_LM"은 문장 생성과 같은 인과적 언어 모델링 태스크를, "MASKED_LM"은 마스크 언어 모델링 태스크를 지정합니다.
    )


    # Set training parameters
    training_arguments = TrainingArguments(
        # 훈련 결과 저장 경로
        output_dir = "./results",
        # 훈련 에포크 수
        num_train_epochs = epochs,
        # 배치 크기
        per_device_train_batch_size = 1,
        # 그래디언트 누적 횟수 (옵션, 기본값: 1)
        gradient_accumulation_steps = 1,
        # 옵티마이저 (옵션, 기본값: 'adamw')
        optim = "paged_adamw_32bit",
        # 로그 출력 주기 (옵션, 기본값: 500)
        logging_steps = steps,
        # 학습률 (옵션, 기본값: 5e-5)
        learning_rate = 2e-6,
        # 가중치 감소 (옵션, 기본값: 0.0)
        weight_decay = 0.001,
        # fp16 사용 여부 (옵션, 기본값: False)
        fp16 = False,
        # bf16 사용 여부 (옵션, 기본값: False)
        bf16 = False,
        # 최대 그래디언트 규범 (옵션, 기본값: 1.0)
        max_grad_norm = 0.3,
        # 최대 훈련 스텝 (옵션, 기본값: -1)
        max_steps = -1,
        # 워밍업 비율 (옵션, 기본값: 0.0)
        warmup_ratio = 0.03,
        # 입력 길이에 따라 배치를 그룹화 (옵션, 기본값: False)
        group_by_length = True,
        # 학습률 스케줄러 유형 (옵션, 기본값: 'cosine')
        lr_scheduler_type = "cosine",
        # 훈련 정보 출력 방식 (옵션, 기본값: 'tensorboard')
        report_to="tensorboard",
        # 검증 주기 (옵션, 기본값: 'epoch')
        evaluation_strategy="steps",
        # 검증 주기 (스텝) (옵션, evaluation_strategy가 'steps'일 때 필수)
        eval_steps=steps,
        # 모델 저장 주기 (옵션, 기본값: 'steps')
        save_strategy="steps",
        # 모델 저장 주기 (스텝) (옵션, save_strategy가 'steps'일 때 필수)
        save_steps = steps,
        # 훈련 종료 시 최적의 모델 로드 (옵션, 기본값: False)
        load_best_model_at_end=True,
        # 최적의 모델 선택 기준 (옵션, 기본값: 'loss')
        metric_for_best_model="loss",
        # 저장할 최대 체크포인트 수 (옵션, 기본값: None)
        save_total_limit=3,
        # # Early stopping 관련 설정
        # early_stopping_enabled=True,
        # # Early stopping 조건 (검증 손실이 patience 스텝 동안 개선되지 않으면 훈련 중단)
        # early_stopping_patience=5,
        # # Early stopping 기준 (검증 손실)
        # early_stopping_threshold=0.001,
        )

    #  Supervised Fine-tuning을 위한 트레이너 객체 생성
    trainer = SFTTrainer(
        # Fine-tuning할 모델
        model=model,  
        # 훈련 데이터셋
        train_dataset=train_dataset,  
        # 평가 데이터셋
        eval_dataset=validation_dataset,  
        # QLoRA 설정
        peft_config=peft_config,  
        # 데이터셋에서 텍스트 필드 명 (본 예시에서는 "Q")
        dataset_text_field="Q",  
        # 입력 시퀀스 최대 길이 제한 (설정하지 않음)
        max_seq_length=None,  
        # 토크나이저
        tokenizer=tokenizer,  
        # 훈련 설정 (training_arguments 변수)
        args=training_arguments,  
        # 입력 시퀀스 패딩 여부 (본 예시에서는 패딩하지 않음)
        packing=False  
    )


    # 모델 훈련
    trainer.train()

    # 훈련된 모델 저장
    trainer.model.save_pretrained(plus_epoch_model)


# Hugging Face 허브에서 훈련하고자 하는 모델을 가져와서 이름 지정
model_name = '42dot/42dot_LLM-SFT-1.3B'


# fine-tuning(미세 조정)을 거친 후의 모델에 부여될 새로운 이름을 지정하는 변수
plus_epoch_model = "/home/dacon_project/CheckPoints/base_multi_1288_50epoch_Qlora"

# 데이터셋 경로
load_dataset_path = '/home/dacon_project/Data/Load_data/Base_Multi_Q_A'

# 사용할 파라미터 설정
epochs = 100
steps = 200

training(model_name, plus_epoch_model, load_dataset_path, epochs, steps)