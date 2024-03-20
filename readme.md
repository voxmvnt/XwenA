![프로젝트 소개](https://github.com/voxmvnt/XwenA/blob/main/dacon_intro.jpg)

## 👨‍🏫 프로젝트 소개
- DACON 한솔데코 AI 경진대회 참여
- 도배 하자 질의 응답처리 모델 구현
<br> (자세한 내용은 발표 PDF 파일을 참고)

## ⏲️ 개발 기간 
- 2024.02.08(목) ~ 2024.03.14(목)
- 주제 선정: 2024.02.08(목)
- 발표 리허설: 2024.03.13(수)
- 발표평가: 2024.03.15(금)
  
## 🧑‍🤝‍🧑 개발자 소개 
- **최재빈** : Langchain 구조 설계, 데이터 Prompt 설계
- **서영우** : LLM 모델 파인튜닝, Web 설계
- **주용규** : RAG 체인 설계 및 튜닝, 데이터 증강 및 전처리
- **윤성민** : 데이터 도메인 마스터, 데이터 증강 및 전처리
  
## 💻 개발환경
- Python 3.10.11
- langchain-0.1.11
- peft-0.9.0
- faiss-gpu-1.7.2

## 📌 주요 기능 및 특이사항
- Llama 모델을 이용, 도배하자와 관련된 질문에 대한 답변을 생성.
- 질문/답변 데이터셋을 추가하여 기본 train 데이터 외의 질문에도 답변을 생성.
- 모델 훈련 내역 및 점수기록은 데이콘 제출로그 파일을 참고.
