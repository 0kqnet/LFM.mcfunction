# LFM.mcfunction

Minecraft Java Edition 26.3 のコンテキスト値プロバイダーを使い、
`LFM2.5-1.2B-JP-202606-Q4_0.gguf` をデータパック内で推論する実験です。
重みと埋め込みは structure の NBT チャンクとして保持し、実行時に一時的な
marker entity から storage へ読み込みます。

## 配布用データパック

`v*`タグをpushするか、GitHub Actionsの`Build release datapack`を手動実行すると、
固定済みQ4_0モデルからデータパックを完全生成し、ZIPとSHA-256ファイルを
GitHub Releaseへ添付します。生成済みデータパック自体はGitで追跡しません。

## 現在できること

- GGUF の Q4_0 / Q2_K / Q3_K / Q6_K 重みを Minecraft 用 NBT へ変換
- LFM2.5 の全 16 層、short convolution、attention、FFN、RMSNorm を実行
- RoPE と複数トークンの KV cache を使った自己回帰推論
- 65,536 語彙の argmax とトークン文字列への復号
- 文章を公式チャット形式へ整形し、入力トークン列を生成
- 入力列の prefill 後、複数の応答トークンを自動生成
- 実行中ジョブの一時停止、再開、再起動後の recovery

推論は Minecraft サーバー上で行います。`chat-command` が使うローカルの
llama.cpp サーバーは入力の tokenization 専用で、応答生成には使いません。

## テスト

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -q
```

## データパックの構築

```powershell
$env:PYTHONPATH='src'
python -m llmcf.cli compile-model `
  'models\LFM2.5-1.2B-JP-202606-Q4_0.gguf' `
  'dist\LFM2.5-1.2B-JP-202606-Q4_0-datapack' `
  --lock 'model.lock.json'
```

`model.lock.json` の SHA-256 と一致しないモデルは拒否します。

## Minecraft 内だけで文字列を渡す

OP のプレイヤーがチャット欄から次を実行します。入力のトークン化、推論、復号、
応答表示までサーバーのデータパック内で進み、外部の tokenizer は不要です。

```text
/function lfm:api/chat {message:"こんにちは",max_tokens:8}
```

入力は可逆な文字単位トークンを優先し、該当トークンがない文字は GPT-2 の
UTF-8 基礎トークンへ分解します。現在の制限は入力 1～64 文字、出力 1～64
トークン、入力と出力の合計 256 トークン以内です。ASCII、かな、漢字、全角文字
など29,283文字を収録しています。引用符とバックスラッシュを含む入力は未対応です。

応答トークンは `[LFM] next token:` としてゲーム内チャットへ順次表示されます。
表示対象にするプレイヤーには一度だけ次を実行します。

```text
/tag @s add lfm.operator
```

入力状態は次で確認できます。

```text
/function lfm:api/input_status
```

## llama.cpp tokenizerを使う互換入力

tokenizer endpoint が `127.0.0.1:8081` で動いている状態で次を実行します。

```powershell
$env:PYTHONPATH='src'
python -m llmcf.cli chat-command 'Hello' --max-tokens 8
```

出力された `function lfm:api/start_prompt {...}` を Minecraft の管理者コンソール
から実行します。`--system '...'` で system message も付けられます。

## 続きの会話ターン

前の応答が EOS まで生成された後、`--append` を指定します。

```powershell
$env:PYTHONPATH='src'
python -m llmcf.cli chat-command '続けて' --max-tokens 8 --append
```

出力される `function lfm:api/append_prompt {...}` は、既存の KV cache と
short convolution state を維持したまま次の user turn を追加します。

## Minecraft 側の操作

```text
function lfm:api/status
function lfm:api/prefill_status
function lfm:api/generation_status
function lfm:api/pause
function lfm:api/resume
function lfm:api/reset
```

実行中にサーバーを再起動するとジョブは `recovered` になります。
ワールド保存後に `function lfm:api/resume` を実行すると、保存済みの層・行・
キャッシュ位置から再開します。

## 性能上の注意

これは実用速度より「Minecraft のデータパックだけで本物のモデル計算が通る」
ことを優先した実装です。内部計算は細かなステップに分かれていますが、通常は
行列の行演算だけ最大32ステップを1ゲームtickにまとめ、重いベクトル演算や重み
ロードは1ステップずつ実行します。生成時の `steps_per_tick` で行演算の上限を調整
でき、負荷の高い環境では小さくできます。1トークンにはなお大量の計算が必要です。
無人の検証サーバーでは `/tick sprint` を区切って使い、各区間のあとに状態と保存を
確認してください。

テスト成功とデータパックの構造検査は、実サーバー上の生成成功とは別の証拠です。
最終確認では `lfm:job`、`lfm:cache`、`lfm:generation` をライブで検査します。
