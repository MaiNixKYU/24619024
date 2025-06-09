# 24619024
[DACON]갑상선암 진단 분류 해커톤 : 양성과 악성, AI로 정확히 구분하라!


# 갑상선암 진단 AI 모델 학습 및 예측 코드
# 데이터 로드 → 전처리 → SMOTE 성능 비교 → 최종 모델 학습 → 예측 → 제출 파일 생성

# 0. Google Drive 마운트 및 데이터 압축 해제 (Colab 환경)
from google.colab import drive
drive.mount('/content/drive')

!unzip --qq /content/drive/MyDrive/open.zip -d dataset

# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, DMatrix, train as xgb_train
import xgboost
print("XGBoost version:", xgboost.__version__)

# 2. 데이터 로드
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# 3. 데이터 전처리
X = train.drop(columns=['ID', 'Cancer'])
y = train['Cancer']
x_test = test.drop(columns=['ID'])

# 결측치 처리 (범주형은 최빈값, 수치형은 중앙값으로 대체)
for col in X.columns:
    if X[col].dtype == 'object':
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        x_test[col].fillna(mode_val, inplace=True)
    else:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        x_test[col].fillna(median_val, inplace=True)

# 범주형 변수 원-핫 인코딩
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
X = pd.get_dummies(X, columns=categorical_cols)
x_test = pd.get_dummies(x_test, columns=categorical_cols)

# train/test 컬럼 맞추기
X, x_test = X.align(x_test, join='left', axis=1, fill_value=0)

# 수치형 변수 스케일링
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

print("Train shape:", X.shape)
print("Test shape:", x_test.shape)

# 4. train/validation 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# 5. XGBoost native 훈련 함수 정의 (early stopping 포함)
def train_and_eval(X_tr, y_tr, X_val, y_val, label):
    dtrain = DMatrix(X_tr, label=y_tr)
    dval = DMatrix(X_val, label=y_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.03,
        'max_depth': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'scale_pos_weight': 1,  # SMOTE 적용 시 1로 고정
        'seed': 42
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb_train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    preds = (model.predict(dval) > 0.5).astype(int)
    f1 = f1_score(y_val, preds)
    print(f"[{label}] Validation F1-score: {f1:.4f}")
    return model, f1

# 6. SMOTE 미적용 모델 학습 및 평가
model_raw, f1_raw = train_and_eval(X_train, y_train, X_val, y_val, "RAW")

# 7. SMOTE 적용 모델 학습 및 평가
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model_smote, f1_smote = train_and_eval(X_train_smote, y_train_smote, X_val, y_val, "SMOTE")

# 8. 더 좋은 성능의 데이터로 전체 데이터 재샘플링
if f1_smote >= f1_raw:
    print("✅ SMOTE가 더 나은 성능, 전체 데이터에 SMOTE 적용")
    smote_full = SMOTE(random_state=42)
    X_final, y_final = smote_full.fit_resample(X, y)
else:
    print("✅ 원본 데이터가 더 나은 성능, 원본 데이터 사용")
    X_final, y_final = X, y

# 9. 최종 XGBClassifier 모델 학습
final_model = XGBClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_final == 0).sum() / (y_final == 1).sum(),
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_final, y_final)

# 10. 테스트 데이터 예측
final_pred = final_model.predict(x_test)

# 11. 제출 파일 생성
submission = pd.read_csv('dataset/sample_submission.csv')
submission['Cancer'] = final_pred
submission.to_csv('baseline_submit.csv', index=False)

print("Submission file saved as 'baseline_submit.csv'")


<img width="896" alt="image" src="https://github.com/user-attachments/assets/e5762aad-d437-45e0-afa3-d025560baa14" />
<img width="880" alt="image" src="https://github.com/user-attachments/assets/ea625048-8dfb-44c0-9513-a6d4a117c9bf" />

