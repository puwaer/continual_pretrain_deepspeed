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
    JSONファイルからテキストデータを読み込み、Datasetsオブジェクトに変換する

    Args:
        file_path (str): JSONファイルのパス

    Returns:
        Dataset: テキストデータを含むデータセット
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # テキストデータの抽出とバリデーション
        texts = []
        for item in data:
            if not isinstance(item.get('text', ''), str):
                continue
            if len(item['text'].strip()) == 0:
                continue
            texts.append(item['text'])
        
        return Dataset.from_dict({'text': texts})
    except Exception as e:
        raise RuntimeError(f"データの読み込みに失敗しました: {str(e)}")

def create_preprocess_function(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    トークナイズ関数を生成する

    Args:
        tokenizer (PreTrainedTokenizer): トークナイザーのインスタンス
        max_length (int): 最大シーケンス長

    Returns:
        function: 前処理関数
    """
    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        # 最適化されたトークナイズ処理
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors=None,
            return_special_tokens_mask=False,
            add_special_tokens=True
        )
        
        result["labels"] = result["input_ids"].copy()
        return result
    
    return preprocess_function

def setup_model_and_tokenizer(config):
    """
    モデルとトークナイザーの設定

    Args:
        config: モデル設定

    Returns:
        tuple: (model, tokenizer)
    """
    # モデルの初期化
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=torch.float16,
        use_cache=config.model.use_cache,
        trust_remote_code=True
    )
    
    # トークナイザーの初期化
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    """
    データセットの準備

    Args:
        config: データセット設定
        tokenizer: トークナイザー

    Returns:
        tuple: (training_dataset, eval_dataset)
    """
    # データセットの読み込み
    dataset = load_json_data(config.dataset.path)
    
    # 前処理関数の作成
    preprocess = create_preprocess_function(tokenizer, config.model.max_length)
    
    # データの前処理
    processed_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text"],
        desc="データの前処理中..."
    )
    
    # train/testデータの分割
    split_dataset = processed_dataset.train_test_split(
        test_size=0.2,
        seed=config.seed
    )
    
    return split_dataset["train"], split_dataset["test"]

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

    try:
        # モデルとトークナイザーの設定
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # データセットの準備
        train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
        
        # 学習の設定
        training_args = TrainingArguments(**config.train)
        
        # トレーナーの初期化
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # 学習の実行
        with torch.autocast("cuda"):
            trainer.train()
            
            # モデルの保存
            trainer.save_model(config.train.output_dir)
            
    except Exception as e:
        print(f"学習中にエラーが発生しました: {str(e)}")
        raise
    finally:
        # クリーンアップ
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()