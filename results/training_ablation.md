# 学習量と追加画像の予備比較

## データと条件

- 既存の学習画像18枚と、ユーザーが `data/train_inbox/` に追加した画像131枚を比較した。追加分はJPEG 128枚、PNG 3枚、合計765,237,393 byte。動画1本と `.DS_Store` は学習対象外。
- 追加画像はすべてRustの学習器で読み取れた。既存26枚と追加131枚の間にSHA-256の完全一致はなく、9×8のdHashで距離5以下の組もなかった。追加画像内には近い組が3組ある。別カットや同じ撮影場所の混入まで否定する検査ではない。
- 追加画像のファイル名・容量・SHA-256はローカルの `data/train_inbox_manifest.json` に保存する。manifest SHA-256は `39fd0d7cf477b5475afc7921bd119478e71a594e4a2c3140862db0c3e9dc89a5`。生画像とこのmanifestはGit管理外。
- モデル構造は499パラメータのまま、元の18枚と計149枚を比較した。各画像から最大3つの512×512 cropを作る。品質70のJPEGから2倍超解像し、各epochで20万画素を抽出する。学習率0.0003、seed 42/43/44、epoch 8/32/64。追加画像を入れた場合も総抽出画素数は同じ。
- 各FP16モデルを同じ一律FP6 E2M3レシピで量子化した。FP16ファイルは1006 byte、FP6ファイルは431 byte。
- 元の8 epoch・seed 42モデルは既存の `data/gallery_fp16_selected.qsr` とSHA-256が完全一致した。

## RGB PSNR

同じ検証4枚・評価4枚を使った比較で、単位はdB。DNN行は3 seedの平均。補間方式は学習を行わない。画像別・seed別の値とモデルSHA-256は [`training_ablation.json`](training_ablation.json) と [`direct_comparison.json`](direct_comparison.json)、補間方式の画像別結果は [`interpolation_baselines.json`](interpolation_baselines.json) にある。直接生成DNNは低解像度画像から2×2画素を出力し、学習の1サンプルあたり4画素を教師にする。比較条件と制約は [`direct_comparison.md`](direct_comparison.md) を参照。

| 方法 | 学習画像 | epoch | 検証 PSNR | 評価 PSNR |
| --- | --- | ---: | ---: | ---: |
| nearest neighbor | — | — | 32.513 | 29.722 |
| bicubic | — | — | 34.355 | 31.151 |
| DNN FP16 | 元の18枚 | 8 | 34.332 | 31.308 |
| DNN FP6 | 元の18枚 | 8 | 34.269 | 31.297 |
| 直接生成DNN FP16 | 元の18枚 | 8 | 34.276 | 31.167 |
| DNN FP16 | 元の18枚 | 32 | 34.273 | 31.405 |
| DNN FP6 | 元の18枚 | 32 | 34.289 | 31.413 |
| DNN FP16 | 元の18枚 | 64 | 34.357 | 31.438 |
| DNN FP6 | 元の18枚 | 64 | 34.327 | 31.399 |
| 直接生成DNN FP16 | 元の18枚 | 64 | 34.378 | 31.292 |
| DNN FP16 | 18枚＋追加131枚 | 8 | 34.386 | 31.318 |
| DNN FP6 | 18枚＋追加131枚 | 8 | 34.371 | 31.302 |
| DNN FP16 | 18枚＋追加131枚 | 32 | 34.393 | 31.383 |
| DNN FP6 | 18枚＋追加131枚 | 32 | 34.393 | 31.378 |
| DNN FP16 | 18枚＋追加131枚 | 64 | 34.354 | 31.401 |
| DNN FP6 | 18枚＋追加131枚 | 64 | 34.322 | 31.382 |
| 直接生成DNN FP16 | 18枚＋追加131枚 | 64 | 34.354 | 31.214 |

### 数字の読み方

PSNRは**高いほど元画像に近い**。同じ画像分割で比較する。+0.1 dBは画素の平均二乗誤差が約2.3%減ることに相当する。表は3 seedの平均なので、差が小さい場合は下のseed別結果も見る。

| 比較する数字 | 良い方向 | 何が分かるか |
| --- | --- | --- |
| FP16 − bicubic | 正で、検証と評価の両方で増える | DNN自体が単純な拡大より役立つか |
| FP6 − **同じ学習条件・seedの**FP16 | 0に近い（大きな負値を避ける） | 6-bit量子化で失った画質。FP6が少し上回ることもある |
| 追加画像あり − 元の18枚（同じepoch・seed） | 検証と評価の両方で正 | 画像追加が汎化に役立つか |
| 長い学習 − 短い学習（同じ画像・seed） | 検証と評価の両方で正 | 学習量を増やす価値があるか |
| seed間の幅 | 小さい | 初期値と抽出画素による結果のぶれ |

例えば「18枚＋追加131枚・32 epoch」のFP16は検証34.393 dBで、bicubicの34.355 dBより+0.038 dB。FP6も検証34.393 dBなので量子化差はほぼ0。評価ではFP16 31.383 dB、FP6 31.378 dBで、量子化差は-0.004 dB。ただし同じ32 epochで元の18枚だけを使ったFP16評価31.405 dBより-0.022 dBなので、**量子化には耐えていても画像追加の改善は示していない**。

次のモデルで目指すのは、同じseed・画像分割で**FP16とFP6の検証PSNRを両方上げる**こと。そのうえで未使用の最終評価でも改善を確かめ、FP6 − FP16を0付近に保つ。ファイル容量と推論時間も併記し、大きくしたモデルの費用と見比べる。学習時のMSEだけが下がっても、未使用画像のPSNRが上がらなければ改善とは扱わない。

参考：bicubicは検証34.355 dB、評価31.151 dB。元の18枚のFP16を8→64 epochにすると、評価平均は+0.130 dB、検証平均は+0.025 dB。FP6の評価平均も+0.101 dB。一方、32 epochの検証平均は8 epochより低い。**この結果は、元モデルが評価画像に対して学習を続ければ伸びる余地があったことを示すが、学習不足という一般的な結論までは示さない。**

追加画像の効果は条件によって変わる。同じseed・epochで差を取ると、32 epochの検証FP16は3 seedすべてで+0.113〜+0.128 dBだが、評価FP16は-0.124〜+0.086 dB。64 epochの3 seed平均では、検証が-0.003 dB、評価が-0.037 dB。現時点で「画像を増やせば改善する」とは言えない。

seed差も大きい。元の18枚・8 epochの検証FP16は34.222〜34.447 dBで、既存のseed 42モデルが最良だった。FP16とFP6の差はこのばらつきより小さい条件が多い。FP6のPSNRがFP16をわずかに上回る行もあり、量子化誤差が常に画質低下として現れるわけではない。

### 隠れ層の浅さと量子化

現行DNNは3×3 RGB入力27値 → ReLUの隠れ層16ユニット → RGB補正3値という**隠れ層1層、499パラメータ**のモデル。表現力不足が画像追加の効果を妨げている可能性はある。ただし、層が浅いこと自体がFP6に弱いという証拠はまだない。

元の8 epoch・seed 42モデルのFP6 − FP16は検証-0.009 dB、評価-0.003 dB。追加画像あり・32 epochの3 seed平均でも検証はほぼ0、評価-0.004 dB。一方、元の18枚・8 epoch・seed 43の検証では-0.230 dBと大きく落ちる。**通常の差は小さいが、条件によって量子化に敏感になる**というのが今のデータから言える範囲。

浅さや幅が原因かを確かめるには、現行の16ユニットに対し、幅を32/64ユニットにしたモデルや隠れ層を増やしたモデルを、同じ分割・seed・学習量で比較する。各モデルについてFP16 PSNRとFP6 − FP16を別々に見る。現行のFP6ファイル形式は499パラメータの構造に固定されているため、この比較には形式の拡張も必要。

## 再現方法と限界

追加画像が同じ状態で `data/train_inbox/` にあることを確認し、次のように学習用ディレクトリを作る。

```sh
python3 scripts/stage_combined_train.py data/train data/train_inbox /private/tmp/qse-combined-train data/train_inbox_manifest.json
cargo run --release -- train /private/tmp/qse-combined-train /private/tmp/combined-32-seed42.qsr 32 200000 70 42 0.0003
cargo run --release -- quantize-fp6 /private/tmp/combined-32-seed42.qsr /private/tmp/combined-32-seed42.qsf e2m3
cargo run --release -- eval-report data/gallery_manifest.json data val /private/tmp/combined-val.json /private/tmp/combined-32-seed42.qsr /private/tmp/combined-32-seed42.qsf --runs 1
cargo run --release -- eval-report data/gallery_manifest.json data test /private/tmp/combined-test.json /private/tmp/combined-32-seed42.qsr /private/tmp/combined-32-seed42.qsf --runs 1
```

元の18枚だけを再学習するときは入力を `data/train` にする。seedとepochを上記の条件に替えて繰り返す。評価は各画像の中央最大512×512 cropに対する人工的な低解像度JPEGで、検証4枚・評価4枚に限られる。評価4枚は以前の条件検討で参照済みなので、ここでの数値は探索的なもの。別の未使用画像による最終評価が必要。追加画像は同じ撮影系列を含む可能性があり、画像枚数は独立したシーン数を表さない。推論時間は1回のみ測ったため比較しない。
