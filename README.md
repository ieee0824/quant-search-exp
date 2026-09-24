# quant-search-exp

[FP6の表現と量子化の考察](https://blog.ast.moe/blog/2026-09-24/)を出発点に、学習済みモデルの重みを低bitへ変えたときの品質・容量・速度を比較する実験用リポジトリ。最初の検証対象として、JPEG画像を**縦横2倍**に拡大する小さなDNNを実装した。

## できること

- 高解像度のJPEG/PNG画像から、2分の1へ縮小してJPEG圧縮した画像を生成し、元画像を教師として学習する。
- 低解像度JPEGをbicubicで2倍に拡大し、周囲3×3画素のRGB値を入力とする27→16→3のDNNで各画素の残差を補正する。
- 重み499個をFP16で保存する。学習と推論の計算はFP32で行うため、FP16演算の高速化を示すものではない。
- 未使用画像でbicubicとDNNのPSNRを比較する。
- 学習済みモデルを使ってJPEG画像を2倍に拡大し、品質95のJPEGとして保存する。
- ARM64でNEONが有効な場合は、推論時のDNNの積和計算にSIMDを使う。それ以外の環境ではスカラー処理を使う。

これは量子化比較のための小さなモデルであり、一般的な超解像モデルと同等の画質を保証するものではない。

## ギャラリー画像での学習

Rust 1.97以降とPython 3を使用する。[ギャラリー](https://blog.ast.moe/gallery/)の**サムネイルのリンク先**にあるJPEGを取得する。リンク先はサムネイルとは別の高解像度画像で、取得した26枚は合計約268 MiB。`data/gallery_manifest.json` に取得元URL、保存先、サイズ、SHA-256を記録する。画像はGit管理から除外し、このmanifestと選択済みFP16モデルだけを保存する。

ギャラリーの掲載順で6枚ごとに1枚を検証用、次の1枚を評価用に分け、学習18枚・検証4枚・評価4枚とする。同じ元画像は複数の組へ入れない。

```sh
python3 scripts/fetch_gallery.py
cargo run --release -- train data/train data/gallery_fp16_selected.qsr 8 200000 70 42 0.0003
cargo run --release -- eval data/gallery_fp16_selected.qsr data/val 70
cargo run --release -- eval data/gallery_fp16_selected.qsr data/test 70
cargo run --release -- upscale data/gallery_fp16_selected.qsr input.jpg output.jpg
cargo run --release -- bench data/gallery_fp16_selected.qsr input.jpg 5
```

学習では元画像の左上・中央・右下から最大512×512画素を切り出し、各切り出しを半分に縮小して品質70のJPEGにする。**元画像全体を512画素へ縮小しているわけではない。** 検証と評価には元画像の中央の最大512×512画素だけを使う。こうして元画像の細部を保ちながら、メモリ使用量と評価時間を抑える。

`train` の後ろの数値は順に、epoch数、各epochで抽出する画素数、学習用JPEGの品質、乱数seed、学習率。省略時はそれぞれ8、20000、70、42、0.001。画像は4×4画素以上が必要。

今回選んだFP16モデルは `data/gallery_fp16_selected.qsr`（1006 byte、SHA-256 `23fa6e800b7fb1cc76f7d790bdb61c338d81ce66f588cfe0978c60c7bff57916`）。RGB画素のPSNRは次のとおり。

| 画像 | bicubic | DNN | 差 |
| --- | ---: | ---: | ---: |
| 検証用4枚 | 34.355 dB | 34.447 dB | +0.092 dB |
| 評価用4枚 | 31.151 dB | 31.382 dB | +0.231 dB |

この数値は中央の切り出しと、人工的に作った低解像度JPEGに対する結果。初期の学習条件を調べる段階で評価用画像も一度見ているため、完全な盲検評価ではない。

## 任意の画像で実行する場合

学習用、検証用、評価用の画像をそれぞれ別のディレクトリへ置く。各ディレクトリ直下の `.jpg`、`.jpeg`、`.png` を読み込む。

```sh
cargo run --release -- train data/train model.qsr
cargo run --release -- eval model.qsr data/test 70
cargo run --release -- upscale model.qsr input.jpg output.jpg
cargo run --release -- bench model.qsr input.jpg 5
```

`eval` は評価用画像の中央切り出しを半分に縮小してJPEG圧縮し、そこから2倍に戻した画像と切り出し元を比較する。表示するPSNRはRGB全画素の二乗誤差から計算する。`upscale` で保存するJPEGの再圧縮誤差は、このPSNRには含まれない。

`model.qsr` は `QSR1` 形式のモデルファイルで、重みはlittle-endianのFP16。現状の実装はCPU上で動く。

`bench` は同じ拡大済み画像に対するDNN補正を、スカラー経路と使用環境で選ばれる経路で計測する。最後の数値は計測回数で、省略時は5回。JPEGの読み込み・bicubic拡大・JPEG保存の時間は含めず、画素値の最大差も表示する。学習の計算はSIMD化していない。

## 次の比較実験

学習したFP16モデルを固定し、同じ評価画像で次を比較する。

1. 一律FP6 E3M2と一律FP6 E2M3
2. blockごとの形式選択
3. 推論形式を固定したままのscale・丸め方の探索

各候補で、PSNR、実ファイル容量（scaleや形式タグを含む）、推論時間を測る。候補の選択に使う画像と最終評価画像は分ける。量子化とその比較処理はまだ実装していない。
