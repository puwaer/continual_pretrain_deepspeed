import argparse
import os
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

import deepspeed
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import seed_everything
os.environ["HF_ENDPOINT"] = "https://huggingface.co"  # APIエンドポイントを明示的に指定
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 分散処理させるときにwarningが出たため


def preprocess_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int
) -> Dict[str, List[int]]:
    """
    与えられたテキストをトークナイズし、ラベルを追加する前処理関数。
    ラベルを追加する理由は、AutoModelForCausalLMの入力を'label'としなければならないため

    Args:
        examples (Dict[str, List[str]]): 前処理するテキストの例。キーはテキストのフィールド名（例えば 'text'）。
        tokenizer (PreTrainedTokenizer): 使用するトークナイザーのインスタンス。
        max_length (int): トークナイズ後の最大シーケンス長。

    Returns:
        Dict[str, List[int]]: トークナイズされたテキストと対応するラベルを含む辞書。
    """
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None
    )
    inputs["labels"] = inputs["input_ids"].copy()
    #inputs["labels"] = inputs.input_ids.copy()
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        "-p",
        type=str,
        default="./configs/train_configs/train_test_llmjp.yaml",
        help="モデルパラメータのコンフィグ。yamlファイル",
    )
    parser.add_argument("--local_rank", "-l", type=int, default=0, help="GPUのランク")
    args = parser.parse_args()
    local_rank = args.local_rank
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
        trust_remote_code=True  # Qwenモデルには必須
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer,
        trust_remote_code=True,  # Qwenモデルには必須
        pad_token='<|endoftext|>'  # パッディングトークンの明示的な設定
    )

    # parquetファイルの読み込み
    try:
        dataset = load_dataset(
            "parquet",
            data_files={
                "train": config.dataset.path  # YAMLで指定したパスを使用
            },
            split=config.dataset.split
        )
    except Exception as e:
        print(f"データセット読み込みエラー: {str(e)}")
        raise


    # データをモデルに入力できるように変換
    dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, config.model.max_length
        ),
        batched=True,
        remove_columns=["text"],  # すべて削除せず、必要なカラムだけ削除
        #remove_columns=dataset.column_names,
    )

    dataset = dataset.train_test_split(test_size=0.2)

    # 学習
    training_args = TrainingArguments(**config.train)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        # data_collator=data_collator,
    )

    # trainer.train()
    with torch.autocast("cuda"):
        trainer.train()


if __name__ == "__main__":
    main()
