# 低解像度画像から直接2倍画像を出すDNN

## 構成と学習

`src/direct.rs` のモデルは、低解像度画像の3×3 RGB近傍（27値）を入力し、16ユニットのReLU隠れ層から、高解像度の2×2 RGBブロック（12値）を直接出力する。バイキュービック拡大画像を入力しない。652パラメータ、FP16の`QSD1`ファイルは1,312 byte。現行の補正型DNNは499パラメータ、FP16で1,006 byte。直接生成モデルは初期状態でnearest neighborと同じ画素を出し、学習でその重みを更新する。

学習画像、最大512×512のcrop、品質70の低解像度JPEG、学習率0.0003、seed 42/43/44は補正型DNNの比較に揃えた。各epochで20万サンプルを抽出する。ただし**直接生成の1サンプルは低解像度1画素から高解像度4画素を教師信号にする**一方、補正型の1サンプルは高解像度1画素を教師信号にする。更新回数は同じだが教師画素数は4倍であり、計算量まで揃えた比較ではない。

## 同じ検証・評価画像でのPSNR

単位dB、高いほど良い。DNNの値は各条件3 seedの平均。評価用4枚は以前の条件検討で参照済みなので、探索的な結果。

| 方法 | 学習画像 | epoch | 検証4枚 | 評価4枚 |
| --- | --- | ---: | ---: | ---: |
| nearest neighbor | — | — | 32.513 | 29.722 |
| bicubic | — | — | 34.355 | 31.151 |
| 補正型 FP16 | 元の18枚 | 8 | 34.332 | 31.308 |
| 直接生成 FP16 | 元の18枚 | 8 | 34.276 | 31.167 |
| 補正型 FP16 | 元の18枚 | 64 | 34.357 | 31.438 |
| 直接生成 FP16 | 元の18枚 | 64 | 34.378 | 31.292 |
| 補正型 FP16 | 18枚＋追加131枚 | 64 | 34.354 | 31.401 |
| 直接生成 FP16 | 18枚＋追加131枚 | 64 | 34.354 | 31.214 |

直接生成はnearest neighborを上回り、64 epochではbicubicも検証+0.023 dB、評価+0.141 dB上回る。元の18枚・64 epochでは補正型より検証で+0.021 dBだが、評価で-0.146 dB。追加画像込み64 epochでは検証がほぼ同じで、評価は-0.187 dB。元の18枚・64 epochの評価は3 seedすべてで補正型より低い。**この小規模な比較では、直接生成による画質改善は確認できなかった。**

個別seedと画像別のPSNR、モデルのSHA-256は [`direct_comparison.json`](direct_comparison.json) にある。既存のseed 42・8 epochの補正型モデルとの比較では、直接生成が検証34.335 dB、評価31.173 dB、補正型が34.447 dB、31.382 dB。構造の比較では単一seedの差だけを根拠にしない。

## 再現と制約

元の18枚・64 epoch・seed 42のモデルを `data/direct_fp16_baseline.qsd` に保存した（SHA-256 `8f751151079e275b1581ba1f250eb535f42da2e4bc04dbac0d3d9070e30f0f35`）。次のコマンドで再学習と評価ができる。

```sh
cargo run --release -- train-direct data/train /private/tmp/direct-64.qsd 64 200000 70 42 0.0003
cargo run --release -- eval-direct-report /private/tmp/direct-64.qsd data/val /private/tmp/direct-val.json 70
cargo run --release -- eval-direct-report /private/tmp/direct-64.qsd data/test /private/tmp/direct-test.json 70
cargo run --release -- upscale-direct data/direct_fp16_baseline.qsd low_res.jpg output_2x.jpg
```

追加画像込みの学習には `scripts/stage_combined_train.py` で作ったディレクトリを使う。このモデルのFP6化は未実装で、FP6量子化の耐性は未測定。直接生成と補正型の推論時間も比較していない。評価用画像4枚は盲検ではなく、最終的な優劣には未使用画像が必要。
