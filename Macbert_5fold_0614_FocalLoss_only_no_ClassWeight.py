# -*- coding: utf-8 -*-
"""
Converted from: Macbert_5fold_0614_FocalLoss_only_no_ClassWeight(1).ipynb
This script was converted from the uploaded Jupyter Notebook.
Run in Colab or a Python environment with the required packages installed.
"""


# %% [cell 1]
# # MacBERT 5-Fold（Focal Loss Only，No Class Weight）
#
# 本 notebook 由原本 **Focal Loss + Class Weight + Weight Decay** 版本改寫為：
#
# - 保留 **Focal Loss**：讓模型較重視難分類樣本。
# - 移除 **Class Weight**：不再根據類別數量放大少數類別 loss。
# - 保留 **Weight Decay**：這是一般正則化，不屬於 class weight，不會直接放大少數類別。
# - 保留原本 5-fold 交叉驗證、資料前處理、標籤抽取、per-class 指標、Balanced Accuracy、Top-3 Accuracy 與 final model 訓練流程。
#
# ## 主要設定
# - `MODEL_NAME = "hfl/chinese-macbert-base"`
# - `MAX_LENGTH = 256`
# - `RANDOM_STATE = 42`
# - `NUM_EPOCHS = 3`
# - `LEARNING_RATE = 1e-5`
# - `TRAIN_BATCH_SIZE = 8`
# - `EVAL_BATCH_SIZE = 8`
# - `WEIGHT_DECAY = 0.01`
# - `N_SPLITS = 5`
# - `FOCAL_GAMMA = 2.0`
#
# ## 版本差異
# 本版不再建立、列印或傳入類別權重；`FocalLossTrainer` 只計算標準 Focal Loss。

# %% [cell 2]
# NOTE: notebook shell command; run manually if needed: !pip install -q transformers datasets accelerate scikit-learn

# %% [cell 3]
import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

try:
    from google.colab import files
    IN_COLAB = True
except ModuleNotFoundError:
    files = None
    IN_COLAB = False

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    top_k_accuracy_score,
    classification_report
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)

# %% [cell 4]
if IN_COLAB:
    uploaded = files.upload()
    uploaded_filename = list(uploaded.keys())[0]
else:
    csv_files = sorted(Path(".").glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("找不到 CSV 檔案，請把資料 CSV 放在這支程式同一個資料夾。")
    uploaded_filename = str(csv_files[0])

# %% [cell 5]
print("使用的資料檔案：", uploaded_filename)

# %% [cell 6]
INPUT_CSV = uploaded_filename
OUTPUT_BASE_DIR = Path("/content") if IN_COLAB else Path("outputs")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(OUTPUT_BASE_DIR / "macbert_taiwan_fm_output")
MODEL_NAME = "hfl/chinese-macbert-base"

TEXT_COL_MODE = "auto"          # "auto" / "question_text_only"
LABEL_COL = None                # 若你有正式標籤欄，例如 "label" 可改成 "label"

MAX_LENGTH = 256
RANDOM_STATE = 42
NUM_EPOCHS = 3
#LEARNING_RATE = 2e-5
LEARNING_RATE = 1e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
N_SPLITS = 5

# Focal Loss 與 Top-K Accuracy 設定
FOCAL_GAMMA = 2.0
TOP_K = 3

AUTO_LABEL_OUTPUT = str(OUTPUT_BASE_DIR / "MacBERT_Taiwan_FM_with_auto_label.csv")
SUCCESS_LABEL_OUTPUT = str(OUTPUT_BASE_DIR / "MacBERT_Taiwan_FM_classification_success.csv")
FAILED_LABEL_OUTPUT = str(OUTPUT_BASE_DIR / "MacBERT_Taiwan_FM_label_extract_failed.csv")
STATS_OUTPUT = str(OUTPUT_BASE_DIR / "MacBERT_Taiwan_FM_success_failed_stats.csv")

set_seed(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [cell 7]
# 6
#
DEPARTMENT_LIST = [
    "婦產科",
    "泌尿科",
    "肝膽腸胃科",
    "眼科",
    "皮膚科",
    "骨科",
    "外科",
    "內科",
    "精神科",
    "耳鼻喉科"
]
#
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_input_text(row):
    title = clean_text(row.get("question_title", ""))
    qtext = clean_text(row.get("question_text", ""))

    if TEXT_COL_MODE == "question_text_only":
        return qtext

    if title and qtext:
        return f"標題：{title} 內容：{qtext}"
    if qtext:
        return qtext
    return title

def normalize_answer_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r", "\n")
    text = text.replace("回覆：", "")
    text = text.replace("／", "/")
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"\n+", "\n", text).strip()
    return text

def remove_header_info(text):
    if not text:
        return ""
    text = re.sub(
        r"^.*?醫院\s*/\s*.*?科\s*/\s*.*?[,，]?\s*\n?\s*\d{4}/\d{1,2}/\d{1,2}\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL
    )
    return text.strip()

def extract_label_from_answer(answer_text):
    text = normalize_answer_text(answer_text)
    text = remove_header_info(text)

    if not text:
        return None

    compact = text.replace("\n", "")

    patterns = [
        r"請至([^，。；、,\n]+?科)門診",
        r"建議至([^，。；、,\n]+?科)",
        r"建議看([^，。；、,\n]+?科)",
        r"可至([^，。；、,\n]+?科)",
        r"可看([^，。；、,\n]+?科)",
        r"掛([^，。；、,\n]+?科)",
        r"先至([^，。；、,\n]+?科)",
        r"轉([^，。；、,\n]+?科)",
        r"到([^，。；、,\n]+?科)就診",
        r"([^，。；、,\n]+?科)門診",
    ]

    for p in patterns:
        m = re.search(p, compact)
        if m:
            label = m.group(1).strip()
            if label in DEPARTMENT_LIST:
                return label

    for dept in sorted(DEPARTMENT_LIST, key=len, reverse=True):
        if dept in compact:
            return dept

    return None

def safe_top_k_accuracy(y_true, logits, k=3):
    """
    安全計算 Top-K Accuracy。
    如果類別數少於 k，會自動改成 Top-類別數。
    """
    num_classes = logits.shape[1]
    k = min(k, num_classes)

    return top_k_accuracy_score(
        y_true,
        logits,
        k=k,
        labels=list(range(num_classes))
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    top3_acc = safe_top_k_accuracy(labels, logits, k=TOP_K)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "top3_accuracy": top3_acc,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }

# %% [cell 8]
# 7
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

required_cols = ["question_title", "question_text", "answer_text"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"CSV 缺少必要欄位：{missing_cols}")

df["text"] = df.apply(build_input_text, axis=1)

if LABEL_COL and LABEL_COL in df.columns:
    df["label_text"] = df[LABEL_COL].astype(str).str.strip()
else:
    df["label_text"] = df["answer_text"].apply(extract_label_from_answer)
#####
# ============================================================
# 症狀 + 科別別名 Mapping Dataset
# 放置位置：
# 1. df["label_text"] = df["answer_text"].apply(extract_label_from_answer) 之後
# 2. 過濾 label_text 空值之前
# ============================================================

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 0. 基本設定
# ------------------------------------------------------------

# True：只補 label_text 原本是空值的資料，較安全，建議使用
# False：會允許症狀 mapping 覆蓋原本 label_text，不建議
SYMPTOM_MAPPING_FILL_ONLY_MISSING = True

# 至少命中幾個症狀關鍵字才補標籤
MIN_SYMPTOM_HITS = 1

# 第一名分數必須比第二名多多少，才視為明確
# 設 1：若兩個科別同分，就不補，避免硬標錯
MIN_SCORE_MARGIN = 1

# 是否輸出 mapping 補標籤結果
EXPORT_MAPPING_RESULT = True

# ------------------------------------------------------------
# 1. 正式科別清單
# 如果 notebook 前面已經有 DEPARTMENT_LIST，就沿用；
# 如果沒有，就用這份預設清單。
# ------------------------------------------------------------

if "DEPARTMENT_LIST" not in globals():
    DEPARTMENT_LIST = [
      "婦產科","泌尿科","肝膽腸胃科","眼科","皮膚科",
      "骨科","外科","內科","精神科","耳鼻喉科"
    ]

VALID_DEPARTMENTS = set(DEPARTMENT_LIST)

# ------------------------------------------------------------
# 2. 科別別名 mapping
# 功能：把資料中不同寫法統一成正式標籤
# 例：胃腸科、消化內科、腸胃內科 → 肝膽腸胃科
# ------------------------------------------------------------

DEPARTMENT_ALIAS_MAPPING = {

    # 內科
    "一般內科": "內科",
    "心臟科": "內科",
    "心內科": "內科",
    "心血管內科": "內科",
    "心臟血管內科": "內科",
    "胸腔科": "內科",
    "肺臟科": "內科",
    "呼吸胸腔科": "內科",
    "呼吸內科": "內科",
    "神經科": "內科",
    "神內科": "內科",
    "腦神經內科": "內科",


    # 婦產科
    "婦科": "婦產科",
    "產科": "婦產科",
    "婦女科": "婦產科",
    "婦產": "婦產科",

    # 肝膽腸胃科
    "腸胃科": "肝膽腸胃科",
    "胃腸科": "肝膽腸胃科",
    "肝膽腸胃科": "肝膽腸胃科",
    "消化內科": "肝膽腸胃科",
    "胃腸肝膽科": "肝膽腸胃科",
    "肝膽胃腸科": "肝膽腸胃科",

    # 神經外科
    "神外科": "外科",
    "腦神經外科": "外科",

    # 身心科 / 精神科
    "身心醫學科": "精神科",
    "心理科": "精神科",
    "精神醫學科": "精神科",

}

# 正式科別也加入 mapping，確保原本正確的名稱不會漏掉
for dept in DEPARTMENT_LIST:
    DEPARTMENT_ALIAS_MAPPING[dept] = dept


# ------------------------------------------------------------
# 3. 症狀 → 科別 mapping dataset
# key：正式科別
# value：該科常見症狀關鍵字
# ------------------------------------------------------------

SYMPTOM_DEPARTMENT_MAPPING = {

    "內科": [
        "發燒", "發熱", "畏寒", "全身痠痛", "水腫", "貧血", "高血壓",
        "慢性病", "體重下降", "食慾不振"
    ],

    "耳鼻喉科": [
        "喉嚨痛", "喉嚨癢", "聲音沙啞", "吞嚥疼痛", "鼻塞", "流鼻水", "鼻涕",
        "鼻竇炎", "鼻過敏", "過敏性鼻炎", "耳鳴", "耳痛", "耳朵痛", "聽力下降",
        "中耳炎", "扁桃腺", "扁桃腺發炎", "口咽", "咽喉", "打鼾"
    ],

    "肝膽腸胃科": [
        "腹痛", "肚子痛", "胃痛", "胃脹", "腹脹", "脹氣", "腹瀉", "拉肚子",
        "便秘", "噁心", "嘔吐", "想吐", "胃酸", "胃食道逆流", "火燒心",
        "吞嚥困難", "消化不良", "食慾不振", "黃疸", "黑便", "血便",
        "大便出血", "肝功能", "B肝", "C肝", "膽結石", "胰臟炎"
    ],


    "骨科": [
        "骨折", "扭傷", "脫臼", "關節痛", "膝蓋痛", "肩膀痛", "手腕痛",
        "腳踝痛", "腰痛", "背痛", "下背痛", "頸椎", "脊椎", "骨刺",
        "退化性關節炎", "骨質疏鬆", "足底筋膜炎"
    ],

    "皮膚科": [
        "皮膚癢", "皮膚紅", "紅疹", "皮疹", "濕疹", "蕁麻疹", "青春痘",
        "痘痘", "毛囊炎", "水泡", "疣", "病毒疣", "香港腳", "灰指甲",
        "掉髮", "頭皮癢", "皮膚過敏", "乾癬", "異位性皮膚炎"
    ],

    "眼科": [
        "眼睛痛", "眼睛癢", "眼睛紅", "紅眼", "視力模糊", "視力下降",
        "飛蚊症", "白內障", "青光眼", "眼壓", "眼乾", "乾眼症",
        "結膜炎", "角膜炎", "流眼淚", "眼屎", "複視"
    ],

    "婦產科": [
        "月經", "經期", "月經不規則", "經痛", "陰道出血", "性交出血",
        "白帶", "陰道分泌物", "陰道癢", "陰部癢", "懷孕", "驗孕",
        "孕婦", "產檢", "避孕藥", "事後避孕", "子宮", "卵巢", "子宮頸",
        "抹片", "HPV", "CIN", "不孕", "多囊", "更年期"
    ],

    "泌尿科": [
        "頻尿", "尿急", "尿痛", "解尿痛", "排尿困難", "血尿", "尿道痛",
        "尿道分泌物", "尿路感染", "膀胱炎", "腎結石", "尿路結石",
        "攝護腺", "前列腺", "睪丸痛", "陰莖", "包皮", "性病", "菜花"
    ],

    "精神科": [
        "幻聽", "幻覺", "妄想", "躁鬱", "精神分裂", "思覺失調",
        "自殺念頭", "自傷", "躁症","焦慮", "憂鬱", "失眠", "睡不著",
        "恐慌", "壓力大", "自律神經","情緒低落", "沒動力", "緊張",
        "強迫症", "恐慌發作"
    ],


    "外科": [
        "傷口", "縫合", "膿瘍", "疝氣", "乳房腫塊", "乳房疼痛",
        "皮下腫塊", "粉瘤", "脂肪瘤", "刀傷", "外傷","氣胸", "肋骨骨折",
        "肺腫瘤", "縱膈腔腫瘤", "胸腔手術","痔瘡", "肛門痛", "肛門癢",
        "肛裂", "肛門出血", "肛門膿瘍",
        "廔管", "直腸", "大腸息肉", "大腸鏡"
    ],

}


# ------------------------------------------------------------
# 4. 基本工具函式
# ------------------------------------------------------------

def _safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_for_matching(text):
    text = _safe_text(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("／", "/")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_department_label(label):
    label = normalize_for_matching(label)

    # 去掉常見尾巴
    for tail in ["門診", "醫師", "就診", "檢查", "追蹤", "評估"]:
        label = label.replace(tail, "")

    return DEPARTMENT_ALIAS_MAPPING.get(label, None)


def get_answer_text_for_mapping(row):
    if "answer_text_clean" in row.index and pd.notna(row["answer_text_clean"]):
        return row["answer_text_clean"]
    if "answer_text" in row.index and pd.notna(row["answer_text"]):
        return row["answer_text"]
    return ""


def get_question_text_for_mapping(row):
    parts = []

    if "question_title" in row.index and pd.notna(row["question_title"]):
        parts.append(str(row["question_title"]))

    if "question_text_clean" in row.index and pd.notna(row["question_text_clean"]):
        parts.append(str(row["question_text_clean"]))
    elif "question_text" in row.index and pd.notna(row["question_text"]):
        parts.append(str(row["question_text"]))

    if "model_text" in row.index and pd.notna(row["model_text"]):
        parts.append(str(row["model_text"]))
    elif "text" in row.index and pd.notna(row["text"]):
        parts.append(str(row["text"]))

    return " ".join(parts)


# ------------------------------------------------------------
# 5. 從 answer_text 再補強一次科別抽取
# 目的：如果原本 extract_label_from_answer 沒抽到，
#      這裡會用科別別名 mapping 再試一次。
# ------------------------------------------------------------

def extract_department_from_answer_with_alias(answer_text):
    text = normalize_for_matching(answer_text)
    if not text:
        return None

    patterns = [
        r"請至([^，。；、,]+?科)門診",
        r"建議至([^，。；、,]+?科)",
        r"建議看([^，。；、,]+?科)",
        r"可至([^，。；、,]+?科)",
        r"可看([^，。；、,]+?科)",
        r"掛([^，。；、,]+?科)",
        r"先至([^，。；、,]+?科)",
        r"轉([^，。；、,]+?科)",
        r"到([^，。；、,]+?科)就診",
        r"([^，。；、,]+?科)門診"
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            raw_label = m.group(1)
            mapped_label = normalize_department_label(raw_label)
            if mapped_label in VALID_DEPARTMENTS:
                return mapped_label

    # 如果沒有句型，就直接找科別別名
    # 長詞優先，避免「內科」先吃掉「心臟內科」
    for alias in sorted(DEPARTMENT_ALIAS_MAPPING.keys(), key=len, reverse=True):
        if alias in text:
            mapped_label = DEPARTMENT_ALIAS_MAPPING[alias]
            if mapped_label in VALID_DEPARTMENTS:
                return mapped_label

    return None


# ------------------------------------------------------------
# 6. 用症狀 mapping 推測科別
# 注意：這只適合補標籤，不建議覆蓋醫師回答已抽出的標籤
# ------------------------------------------------------------

def infer_department_from_symptoms(question_text):
    text = normalize_for_matching(question_text)

    if not text:
        return None, "", 0

    dept_scores = {}
    dept_hits = {}

    for dept, keywords in SYMPTOM_DEPARTMENT_MAPPING.items():
        if dept not in VALID_DEPARTMENTS:
            continue

        hits = []
        score = 0

        for kw in keywords:
            kw_norm = normalize_for_matching(kw)
            if not kw_norm:
                continue

            if kw_norm in text:
                hits.append(kw)

                # 長關鍵字通常比較有辨識力，給較高分
                if len(kw_norm) >= 4:
                    score += 3
                elif len(kw_norm) >= 3:
                    score += 2
                else:
                    score += 1

        if hits:
            dept_scores[dept] = score
            dept_hits[dept] = hits

    if not dept_scores:
        return None, "", 0

    ranked = sorted(dept_scores.items(), key=lambda x: x[1], reverse=True)
    best_dept, best_score = ranked[0]

    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = best_score - second_score

    # 分數太低，不補
    if len(dept_hits[best_dept]) < MIN_SYMPTOM_HITS:
        return None, "", best_score

    # 第一名沒有明顯贏第二名，不補，避免腹痛/胸悶/發燒這種模糊症狀亂標
    if margin < MIN_SCORE_MARGIN:
        return None, "", best_score

    hit_words = "、".join(dept_hits[best_dept])
    return best_dept, hit_words, best_score


# ------------------------------------------------------------
# 7. 實際套用到 df
# ------------------------------------------------------------

if "df" not in globals():
    raise ValueError("找不到 df，請先執行讀取 CSV 的 cell。")

if "label_text" not in df.columns:
    df["label_text"] = np.nan

df["label_text_before_mapping"] = df["label_text"]
df["mapping_label_source"] = "original_label"
df["mapping_hit_keywords"] = ""
df["mapping_score"] = 0

# 7-1. 先用 answer_text + 科別別名 mapping 補 label_text 空值
for idx, row in df.iterrows():
    current_label = row.get("label_text", np.nan)

    if pd.notna(current_label) and str(current_label).strip() != "":
        continue

    answer_text = get_answer_text_for_mapping(row)
    mapped_label = extract_department_from_answer_with_alias(answer_text)

    if mapped_label:
        df.at[idx, "label_text"] = mapped_label
        df.at[idx, "mapping_label_source"] = "answer_alias_mapping"

# 7-2. 再用 question_text 症狀 mapping 補剩下空值
for idx, row in df.iterrows():
    current_label = row.get("label_text", np.nan)

    if SYMPTOM_MAPPING_FILL_ONLY_MISSING:
        if pd.notna(current_label) and str(current_label).strip() != "":
            continue

    question_text = get_question_text_for_mapping(row)
    symptom_label, hit_words, score = infer_department_from_symptoms(question_text)

    if symptom_label:
        df.at[idx, "label_text"] = symptom_label
        df.at[idx, "mapping_label_source"] = "symptom_mapping"
        df.at[idx, "mapping_hit_keywords"] = hit_words
        df.at[idx, "mapping_score"] = score

#
# ============================================================
# 將細分類合併成指定的 10 大類
# 放在 label_text mapping 完成後、success_mask 建立前
# ============================================================

ALLOWED_LABELS = [
    "婦產科",
    "泌尿科",
    "肝膽腸胃科",
    "眼科",
    "皮膚科",
    "骨科",
    "外科",
    "內科",
    "精神科",
    "耳鼻喉科"
]

MERGE_MAP = {
    # 內科相關細分類 → 內科
    "心臟內科": "內科",
    "神經內科": "內科",
    "胸腔內科": "內科",
    "腎臟科": "內科",
    "感染科": "內科",
    "新陳代謝科": "內科",
    "血液腫瘤科": "內科",
    "風濕免疫科": "內科",
    "過敏免疫風濕科": "內科",

    # 外科相關細分類 → 外科
    "神經外科": "外科",
    "胸腔外科": "外科",
    "大腸直腸外科": "外科",

    # 精神相關 → 精神科
    "身心科": "精神科",
    "精神醫學科": "精神科",
}
#

#
# 先把細分類合併成大類
df["label_text"] = df["label_text"].replace(MERGE_MAP)

# 只保留指定的 10 大類
df = df[df["label_text"].isin(ALLOWED_LABELS)].copy()

print("=== 合併並限制成 10 大類後的 label_text 分布 ===")
print(df["label_text"].value_counts())
print()
#

# ------------------------------------------------------------
# 8. 統計 mapping 前後結果
# ------------------------------------------------------------

before_success = df["label_text_before_mapping"].notna().sum()
after_success = df["label_text"].notna().sum()

print("========== Mapping Dataset 補標籤結果 ==========")
print(f"Mapping 前成功標籤數：{before_success}")
print(f"Mapping 後成功標籤數：{after_success}")
print(f"新增成功標籤數：{after_success - before_success}")
print()

print("========== label_text 來源統計 ==========")
print(df["mapping_label_source"].value_counts(dropna=False))
print()

print("========== Mapping 後 label_text 分布 ==========")
print(df["label_text"].value_counts(dropna=False))
print()

print("========== 症狀 mapping 補到的前 20 筆 ==========")
display_cols = [
    "question_title",
    "question_text",
    "label_text_before_mapping",
    "label_text",
    "mapping_label_source",
    "mapping_hit_keywords",
    "mapping_score"
]

display_cols = [c for c in display_cols if c in df.columns]

display(
    df[df["mapping_label_source"] == "symptom_mapping"][display_cols].head(20)
)


# ------------------------------------------------------------
# 9. 輸出檢查檔案
# ------------------------------------------------------------

if EXPORT_MAPPING_RESULT:
    if "OUTPUT_DIR" in globals():
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = Path(".")

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_output_path = output_dir / "mapping_dataset_label_result.csv"
    symptom_mapping_output_path = output_dir / "symptom_mapping_added_cases.csv"

    df.to_csv(mapping_output_path, index=False, encoding="utf-8-sig")

    df[df["mapping_label_source"] == "symptom_mapping"].to_csv(
        symptom_mapping_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("已輸出：")
    print(mapping_output_path)
    print(symptom_mapping_output_path)

#####
df.to_csv(AUTO_LABEL_OUTPUT, index=False, encoding="utf-8-sig")
print(f"已輸出完整自動標籤檔：{AUTO_LABEL_OUTPUT}")

success_mask = (
    df["label_text"].notna() &
    (df["label_text"].astype(str).str.strip() != "") &
    df["text"].notna() &
    (df["text"].astype(str).str.strip() != "")
)
failed_mask = ~success_mask

success_df = df[success_mask].copy()
failed_df = df[failed_mask].copy()

success_df.to_csv(SUCCESS_LABEL_OUTPUT, index=False, encoding="utf-8-sig")
failed_df.to_csv(FAILED_LABEL_OUTPUT, index=False, encoding="utf-8-sig")

print(f"已輸出分類成功資料：{SUCCESS_LABEL_OUTPUT}")
print(f"已輸出抽標籤失敗資料：{FAILED_LABEL_OUTPUT}")
print()

print("=== label_text 分布（含失敗） ===")
print(df["label_text"].value_counts(dropna=False))
print()

print("=== 分類成功資料筆數 ===")
print(len(success_df))
print()

print("=== 分類失敗資料筆數 ===")
print(len(failed_df))
print()

def extract_all_departments_for_stats(answer_text):
    text = normalize_answer_text(answer_text)
    text = remove_header_info(text)

    if not text:
        return []

    compact = text.replace("\n", "")
    matches = []
    used_spans = []

    for dept in sorted(DEPARTMENT_LIST, key=len, reverse=True):
        for m in re.finditer(re.escape(dept), compact):
            span = m.span()
            overlapped = any(not (span[1] <= s[0] or span[0] >= s[1]) for s in used_spans)
            if not overlapped:
                used_spans.append(span)
                matches.append((span[0], dept))

    matches = [dept for _, dept in sorted(matches, key=lambda x: x[0])]

    unique_matches = []
    seen = set()
    for dept in matches:
        if dept not in seen:
            unique_matches.append(dept)
            seen.add(dept)

    return unique_matches

success_counts = success_df["label_text"].value_counts()

failed_records = []
for _, row in failed_df.iterrows():
    depts = extract_all_departments_for_stats(row.get("answer_text", ""))
    if depts:
        failed_records.extend(depts)
    else:
        failed_records.append("未判定科別")

if failed_records:
    failed_counts = pd.Series(failed_records, name="科別").value_counts()
else:
    failed_counts = pd.Series(dtype="int64")

stats_df = pd.concat(
    [success_counts.rename("成功筆數"), failed_counts.rename("失敗筆數")],
    axis=1
).fillna(0).astype(int)

stats_df.index.name = "科別"
stats_df = stats_df.reset_index()
stats_df["總筆數"] = stats_df["成功筆數"] + stats_df["失敗筆數"]
stats_df["成功率"] = (stats_df["成功筆數"] / stats_df["總筆數"]).fillna(0).round(4)
stats_df["失敗率"] = (stats_df["失敗筆數"] / stats_df["總筆數"]).fillna(0).round(4)
stats_df = stats_df.sort_values(["總筆數", "成功筆數", "科別"], ascending=[False, False, True]).reset_index(drop=True)

stats_df.to_csv(STATS_OUTPUT, index=False, encoding="utf-8-sig")

print("=== 各科成功與失敗筆數統計表 ===")
print(stats_df.to_string(index=False))
print()
print(f"已輸出各科成功/失敗統計：{STATS_OUTPUT}")
print()

df = success_df.copy()

if len(df) == 0:
    raise ValueError("沒有可用訓練資料。請確認 label 欄或 answer_text 是否能抽出科別。")

label_counts = df["label_text"].value_counts()
valid_labels = label_counts[label_counts >= N_SPLITS].index.tolist()
df = df[df["label_text"].isin(valid_labels)].copy()

print("=== 標籤分布（每類至少 5 筆後，才可做五折交叉驗證） ===")
print(df["label_text"].value_counts())
print()

if df["label_text"].nunique() < 2:
    raise ValueError("有效類別數不足 2 類，無法做五折分類訓練。")

# 固定 label 順序：優先使用 ALLOWED_LABELS，避免 confusion matrix 類別順序亂跳
if "ALLOWED_LABELS" in globals():
    label_list = [label for label in ALLOWED_LABELS if label in df["label_text"].unique()]
else:
    label_list = sorted(df["label_text"].unique())

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["label_text"].map(label2id)

print("=== 類別對照 ===")
print(label2id)


# %% [cell 9]
# 8
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 下載繁體中文字型
# NOTE: notebook shell command; run manually if needed: !wget -O SourceHanSerifTW-VF.ttf https://github.com/adobe-fonts/source-han-serif/raw/release/Variable/TTF/Subset/SourceHanSerifTW-VF.ttf

# 加入字型檔
fm.fontManager.addfont('SourceHanSerifTW-VF.ttf')

# 設定字型
#
mpl.rc('font', family='Source Han Serif TW VF')

# %% [cell 10]
# 9
#######
#######
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

# ============================================================
# Focal Loss Trainer（Focal Loss Only / No Class Weight）
# 目的：
# 1. 只保留 Focal Loss，讓模型較重視難分類樣本
# 2. 移除 Class Weight，避免少數類別被額外放大
# ============================================================

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)

        if labels is None:
            labels = inputs.pop("label", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        labels = labels.to(logits.device).long()

        # log_softmax 後取出正確類別的 log probability
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        # Focal Loss 核心：(1 - pt)^gamma
        focal_factor = (1 - pt) ** self.focal_gamma

        # Focal Loss Only：不乘上 class weight / alpha
        loss = -focal_factor * log_pt
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

X = df["text"].values
y = df["label"].values

fold_results = []
fold_per_class_reports = []

all_true = []
all_pred = []
all_logits = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print("=" * 70)
    print(f"開始第 {fold} 折交叉驗證")
    print("=" * 70)

    fold_train_df = df.iloc[train_idx][["text", "label"]].reset_index(drop=True)
    fold_val_df = df.iloc[val_idx][["text", "label"]].reset_index(drop=True)

    # ========================================================
    # 類別數量檢查
    # 本版不計算、不使用 Class Weight；此處只列印每類樣本數，方便觀察資料不平衡。
    # ========================================================
    train_label_counts = fold_train_df["label"].value_counts().sort_index()

    class_counts = np.array([
        train_label_counts.get(i, 0) for i in range(len(label_list))
    ])

    print("本 fold 訓練資料類別數量（未使用 class weight）：")
    for i, label_name in id2label.items():
        print(f"{label_name}: count={class_counts[i]}")
    print()

    train_dataset = Dataset.from_pandas(fold_train_df)
    val_dataset = Dataset.from_pandas(fold_val_df)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label
    )

    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none"
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        focal_gamma=FOCAL_GAMMA,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()

    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    top3_acc = safe_top_k_accuracy(y_true, logits, k=TOP_K)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ========================================================
    # Per-class Precision / Recall / F1
    # ========================================================
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_list))),
        average=None,
        zero_division=0
    )

    fold_per_class_df = pd.DataFrame({
        "fold": fold,
        "label_id": list(range(len(label_list))),
        "label_name": label_list,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1": per_class_f1,
        "support": per_class_support
    })

    fold_per_class_reports.append(fold_per_class_df)

    print(f"\n第 {fold} 折 Per-class Precision / Recall / F1：")
    print(fold_per_class_df)

    fold_per_class_output = os.path.join(OUTPUT_DIR, f"fold_{fold}_per_class_metrics.csv")
    fold_per_class_df.to_csv(fold_per_class_output, index=False, encoding="utf-8-sig")
    print(f"已輸出第 {fold} 折 per-class 指標：{fold_per_class_output}")

    fold_results.append({
        "fold": fold,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "top3_accuracy": top3_acc,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    })

    all_true.extend(y_true.tolist())
    all_pred.extend(y_pred.tolist())
    all_logits.extend(logits.tolist())

    print(f"\n第 {fold} 折結果：")
    print(f"Accuracy             : {acc:.4f}")
    print(f"Balanced Accuracy    : {balanced_acc:.4f}")
    print(f"Top-{TOP_K} Accuracy       : {top3_acc:.4f}")
    print(f"Precision(weighted)  : {precision_weighted:.4f}")
    print(f"Recall(weighted)     : {recall_weighted:.4f}")
    print(f"F1(weighted)         : {f1_weighted:.4f}")
    print(f"Precision(macro)     : {precision_macro:.4f}")
    print(f"Recall(macro)        : {recall_macro:.4f}")
    print(f"F1(macro)            : {f1_macro:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    plt.title(f"Fold {fold} 混淆矩陣")
    plt.show()

# %% [cell 11]
# 10
results_df = pd.DataFrame(fold_results)
print("=== 各 Fold 結果 ===")
print(results_df)
print()

print("=== 5-Fold 平均結果 ===")

metric_cols = [
    "accuracy",
    "balanced_accuracy",
    "top3_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

print(results_df[metric_cols].mean())
print()

# ============================================================
# 5-Fold 全部 validation 預測合併後的整體指標
# ============================================================

all_true_np = np.array(all_true)
all_pred_np = np.array(all_pred)
all_logits_np = np.array(all_logits)

overall_accuracy = accuracy_score(all_true_np, all_pred_np)
overall_balanced_accuracy = balanced_accuracy_score(all_true_np, all_pred_np)
overall_top3_accuracy = safe_top_k_accuracy(all_true_np, all_logits_np, k=TOP_K)

print("=== 5-Fold 合併後整體指標 ===")
print(f"Overall Accuracy          : {overall_accuracy:.4f}")
print(f"Overall Balanced Accuracy : {overall_balanced_accuracy:.4f}")
print(f"Overall Top-{TOP_K} Accuracy    : {overall_top3_accuracy:.4f}")
print()

# ============================================================
# 5-Fold 合併後 Per-class Precision / Recall / F1
# ============================================================

overall_precision, overall_recall, overall_f1, overall_support = precision_recall_fscore_support(
    all_true_np,
    all_pred_np,
    labels=list(range(len(label_list))),
    average=None,
    zero_division=0
)

overall_per_class_df = pd.DataFrame({
    "label_id": list(range(len(label_list))),
    "label_name": label_list,
    "precision": overall_precision,
    "recall": overall_recall,
    "f1": overall_f1,
    "support": overall_support
})

print("=== 5-Fold 合併後 Per-class Precision / Recall / F1 ===")
print(overall_per_class_df)
print()

overall_per_class_output = os.path.join(OUTPUT_DIR, "overall_5fold_per_class_metrics.csv")
overall_per_class_df.to_csv(overall_per_class_output, index=False, encoding="utf-8-sig")
print(f"已輸出 5-Fold 合併後 per-class 指標：{overall_per_class_output}")
print()

print("=== Classification Report ===")
print(
    classification_report(
        all_true_np,
        all_pred_np,
        labels=list(range(len(label_list))),
        target_names=label_list,
        zero_division=0
    )
)

# ============================================================
# 輸出每一 fold 的 per-class 指標總表
# ============================================================

all_fold_per_class_df = pd.concat(fold_per_class_reports, axis=0).reset_index(drop=True)

all_fold_per_class_output = os.path.join(OUTPUT_DIR, "all_folds_per_class_metrics.csv")
all_fold_per_class_df.to_csv(all_fold_per_class_output, index=False, encoding="utf-8-sig")

print("=== 每一 fold 的 Per-class 指標總表 ===")
print(all_fold_per_class_df)
print()
print(f"已輸出所有 fold 的 per-class 指標：{all_fold_per_class_output}")
print()

overall_cm = confusion_matrix(all_true, all_pred, labels=list(range(len(label_list))))
disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=label_list)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    cmap="Blues"
)

# 設定座標區底色
ax.set_facecolor("#EAF4FF")

# 放大標題、座標軸標籤、刻度字體
ax.set_title("5-Fold 全部預測合併混淆矩陣", fontsize=22)
ax.set_xlabel("Predicted label", fontsize=18)
ax.set_ylabel("True label", fontsize=18)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

# 放大格子裡的數字
for text in ax.texts:
    text.set_fontsize(20)

plt.show()

# %% [cell 12]
# 11
# ============================================================
# 使用全部資料重新訓練 Final MacBERT Model
# ============================================================

final_output_dir = str(OUTPUT_BASE_DIR / "macbert_final_model")

final_df = df[["text", "label"]].reset_index(drop=True)
final_dataset = Dataset.from_pandas(final_df)
final_dataset = final_dataset.map(tokenize_function, batched=True)

final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label,
    use_safetensors=False
)

final_training_args = TrainingArguments(
    output_dir=final_output_dir,
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)
# Final model 同樣只使用 Focal Loss，不使用 Class Weight。

final_trainer = FocalLossTrainer(
    model=final_model,
    args=final_training_args,
    train_dataset=final_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    focal_gamma=FOCAL_GAMMA
)

final_trainer.train()

final_trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print("Final MacBERT model 已儲存到：", final_output_dir)

# =================================
# 壓縮並下載MacBERT與tokenizer
# =================================

zip_path = str(OUTPUT_BASE_DIR / "macbert_final_model.zip")

shutil.make_archive(
    base_name=zip_path.replace(".zip", ""),
    format="zip",
    root_dir=final_output_dir
)

# %% [cell 13]
results_csv = str(OUTPUT_BASE_DIR / "macbert_5fold_cv_results.csv")
results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

summary = {
    "label2id": label2id,
    "id2label": {str(k): v for k, v in id2label.items()},

    "avg_accuracy": float(results_df["accuracy"].mean()),
    "avg_balanced_accuracy": float(results_df["balanced_accuracy"].mean()),
    "avg_top3_accuracy": float(results_df["top3_accuracy"].mean()),

    "avg_precision_weighted": float(results_df["precision_weighted"].mean()),
    "avg_recall_weighted": float(results_df["recall_weighted"].mean()),
    "avg_f1_weighted": float(results_df["f1_weighted"].mean()),

    "avg_precision_macro": float(results_df["precision_macro"].mean()),
    "avg_recall_macro": float(results_df["recall_macro"].mean()),
    "avg_f1_macro": float(results_df["f1_macro"].mean()),

    "overall_accuracy": float(overall_accuracy),
    "overall_balanced_accuracy": float(overall_balanced_accuracy),
    "overall_top3_accuracy": float(overall_top3_accuracy),

    "focal_gamma": float(FOCAL_GAMMA),
    "use_focal_loss": True,
    "use_class_weight": False,
    "top_k": int(TOP_K)
}

summary_json = str(OUTPUT_BASE_DIR / "macbert_5fold_cv_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("已輸出：")
print(results_csv)
print(summary_json)
print(overall_per_class_output)
print(all_fold_per_class_output)

# %% [cell 14]
download_paths = [
    results_csv,
    summary_json,
    overall_per_class_output,
    all_fold_per_class_output,
    AUTO_LABEL_OUTPUT,
    SUCCESS_LABEL_OUTPUT,
    FAILED_LABEL_OUTPUT,
    zip_path,
]

if IN_COLAB:
    for download_path in download_paths:
        files.download(download_path)
else:
    print("輸出檔案已儲存在：")
    for download_path in download_paths:
        print(download_path)
