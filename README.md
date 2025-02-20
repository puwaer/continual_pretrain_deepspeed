# continual_pretrain_deepspeed

このリポジトリは、DeepSpeedを活用した大規模言語モデル (LLM) の事前学習を行うための環境およびプログラムを提供します。

## 開発環境の構築

開発環境を構築する方法は、Docker環境とpyenv環境の2種類があります。

### 1. Docker環境の構築

1. Dockerコンテナをビルド
   ```sh
   cd continual-pretrain/docker
   chmod +x build.sh
   ./build.sh
   ```
2. Dockerコンテナを起動
   ```sh
   cd continual-pretrain/docker
   docker compose up
   ```
3. コンテナにアクセス
   ```sh
   docker exec -it LLM bash
   ```

### 2. Pyenv環境の構築

詳細なセットアップ手順は `ev-llm` ディレクトリ内に記載されています。

## 依存ライブラリのインストール

以下のコマンドで必要なライブラリをインストール、またはアップグレードしてください。

```sh
pip install --upgrade huggingface_hub datasets transformers
```

Qwenを使用する場合は、追加で以下をインストールします。

```sh
pip install --upgrade 'transformers>=4.36.0'
pip install --upgrade 'accelerate>=0.26.0'
pip install --upgrade wandb
```

## 学習の実行

### シングルノード学習

```sh
cd continual-pretrain
```

#### 一般的なLLMの事前学習
```sh
deepspeed src/train_deepspeed.py --train_config ./configs/train_configs/train_base.yaml
```

```sh
deepspeed src/train_deepspeed_2.py --train_config ./configs/train_configs/train_test.yaml
```

#### Qwenを使用する場合
```sh
deepspeed src/train_deepspeed_qwen.py --train_config ./configs/train_configs/train_test_qwen.yaml
```

#### llm_jpを使用する場合
```sh
deepspeed src/train_deepspeed_llmjp.py --train_config ./configs/train_configs/train_test_llmjp.yaml
```

#### JSONファイルの学習データを使用する場合
```sh
deepspeed src/train_deepspeed_llmjp_json.py --train_config ./configs/train_configs/train_test_llmjp_json.yaml
```

## ライセンス

このリポジトリはMITライセンスの下で提供されます。

---

