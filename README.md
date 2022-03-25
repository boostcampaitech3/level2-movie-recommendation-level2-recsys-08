## Level2- movie-reccomendation-recsys-08

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec


## Installation

```
pip install -r requirements.txt
```

## How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```


## 협업 룰
- Data 경로는 '/opt/ml/input/data/train' 로 통일 (절대경로)

### Git
- 개인 branch 생성해서 진행
- pull request 이후 승낙되어 mearge되면 main으로 간다.

### 업무 분배
- 각 클래스 파일 (e.g. model, dataset)
    - 디렉토리 생성 후 개별 클래스당 단독 py 파일로 저장
    - 모델 중에 유난히 복잡하면 하위 디렉토리 생성해서 진행