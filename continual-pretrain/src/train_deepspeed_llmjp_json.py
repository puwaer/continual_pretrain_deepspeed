import argparse
import os
import warnings
import json
from typing import Dict, List

warnings.filterwarnings("ignore")

import deepspeed
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import seed_everything

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_json_data(file_path: str) -> Dataset:
    """
    JSONファイルからテキストデータのみを読み込み、Datasetsオブジェクトに変換する

    Args:
        file_path (str): JSONファイルのパス

    Returns:
        Dataset: テキストデータのみを含むデータセット
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # テキストデータのみを抽出してDatasetオブジェクトに変換
    return Dataset.from_dict({
        'text': [item['text'] for item in data]
    })

def preprocess_function(
    examples: Dict[str, List[str]], 
    tokenizer: PreTrainedTokenizer, 
    max_length: int
) -> Dict[str, List[int]]:
    """
    テキストをトークナイズし、ラベルを追加する前処理関数

    Args:
        examples (Dict[str, List[str]]): 前処理するテキストの例
        tokenizer (PreTrainedTokenizer): トークナイザーのインスタンス
        max_length (int): 最大シーケンス長

    Returns:
        Dict[str, List[int]]: トークナイズされたテキストとラベルを含む辞書
    """
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
        return_special_tokens_mask=False,
        add_special_tokens=True
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        "-p",
        type=str,
        default="./configs/train_configs/train_test_llmjp_json.yaml",
        help="モデルパラメータのコンフィグ。yamlファイル",
    )
    parser.add_argument("--local_rank", "-l", type=int, default=0, help="GPUのランク")
    args = parser.parse_args()
    
    # コンフィグ読み込み
    config = OmegaConf.load(args.train_config)

    # distributed learning
    deepspeed.init_distributed()

    # seedの設定
    seed_everything(config.seed)

    # モデルの初期化
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=torch.float16,
        use_cache=config.model.use_cache,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )

    # JSONデータの読み込み（テキストのみ）
    try:
        dataset = load_json_data(config.dataset.path)
    except Exception as e:
        print(f"データセット読み込みエラー: {str(e)}")
        raise

    # データの前処理
    dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, config.model.max_length
        ),
        batched=True,
        remove_columns=["text"]
    )

    # train/testデータの分割
    dataset = dataset.train_test_split(test_size=0.2)

    # 学習の設定と実行
    training_args = TrainingArguments(**config.train)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    # 学習の実行
    with torch.autocast("cuda"):
        trainer.train()

if __name__ == "__main__":
    main()