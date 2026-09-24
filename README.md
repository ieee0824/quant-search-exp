# quant-search-exp

[FP6の表現と量子化の考察](https://blog.ast.moe/blog/2026-09-24/)を出発点に、学習済みモデルの重みを低bitへ変えたときの品質・容量・速度を比較する実験用リポジトリ。最初の検証対象として、JPEG画像を**縦横2倍**に拡大する小さなDNNを実装した。

## できること

- 高解像度のJPEG/PNG画像から、2分の1へ縮小してJPEG圧縮した画像を生成し、元画像を教師として学習する。
- 低解像度JPEGをbicubicで2倍に拡大し、周囲3×3画素のRGB値を入力とする27→16→3のDNNで各画素の残差を補正する。
- 重み499個をFP16で保存する。学習と推論の計算はFP32で行うため、FP16演算の高速化を示すものではない。
- 比較器の確認用に、全重みを一つの共有scaleでINT8へ丸める小さな`QSI1`形式も使える。INT8専用演算の速度を示すものではない。
- 480個の接続重みをFP6 E3M2/E2M3へ量子化し、一律形式・blockごとの形式選択・量子化時の探索を比較する。19個のbiasはFP16のまま保存する。
- 分割した画像でbicubicとDNNのPSNRを比較する。
- 学習済みモデルを使ってJPEG画像を2倍に拡大し、品質95のJPEGとして保存する。
- ARM64でNEONが有効な場合は、推論時のDNNの積和計算にSIMDを使う。それ以外の環境ではスカラー処理を使う。

これは量子化比較のための小さなモデルであり、一般的な超解像モデルと同等の画質を保証するものではない。

## ギャラリー画像での学習

Rust 1.97以降とPython 3を使用する。[ギャラリー](https://blog.ast.moe/gallery/)の**サムネイルのリンク先**にあるJPEGを取得する。リンク先はサムネイルとは別の高解像度画像で、取得した26枚は合計約268 MiB。`data/gallery_manifest.json` に取得元URL、保存先、サイズ、SHA-256を記録する。画像はGit管理から除外し、manifestと小さなモデルファイルを保存する。

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

## 再現可能な評価レポート（Issue #1）

`gallery_manifest.json` はギャラリー掲載順、元画像URL、保存先、byte数、SHA-256、train/val/testの分割を固定する。以下のコマンドは**manifestを書き換えずに**26枚を取得し、全画像のbyte数とSHA-256、およびURLとsplit規則を検証する。画像本体は `.gitignore` によりGit管理しない。

```sh
python3 scripts/fetch_manifest.py data/gallery_manifest.json data
cargo run --release -- verify-data data/gallery_manifest.json data
```

候補の選択には`val`だけを使う。`test`の4枚は過去に学習条件の探索で見たため、既存結果の再現・探索的な比較に限る。レポートにもこの扱いを記録する。

```sh
cargo run --release -- eval-report data/gallery_manifest.json data val results/fp16_val.json data/gallery_fp16_selected.qsr --runs 3
cargo run --release -- eval-report data/gallery_manifest.json data test results/fp16_test_exploratory.json data/gallery_fp16_selected.qsr --runs 3
cargo run --release -- quantize-int8 data/gallery_fp16_selected.qsr data/gallery_int8_baseline.qi8
cargo run --release -- eval-report data/gallery_manifest.json data val results/fp16_int8_val.json data/gallery_fp16_selected.qsr data/gallery_int8_baseline.qi8 --runs 3
```

同じコマンドでFP16の`QSR1`、量子化したINT8の`QSI1`、FP6の`QSF1`を比較できる。`QSI1`は12byteのヘッダー（形式、重み数、FP32 scale）と499byteの符号付き重みで、biasも含め全パラメータを一つのscaleで量子化する。モデル読込時にFP32へ展開し、その所要時間は`load_decode_ms`へ別記する。DNN推論時間に毎回のINT8展開は含まれない。FP6は**推論を計測するたびに**6bitの展開とscale適用を行う。

JSONにはモデルとmanifestのSHA-256、全画像の検証結果、設定、画像別と全体のRGB PSNR、実ファイル容量、処理別の時間を残す。全体PSNRは画像ごとのdBの平均ではなく、全RGB画素の二乗誤差を集計して求める。時間は実行環境や負荷で変わり、モデル推論は指定回数の平均。元JPEGの読込・decode、低解像度JPEG作成とbicubic拡大、DNN推論、出力JPEGのメモリ内encodeを分けている。出力JPEGの再圧縮誤差はPSNRに含めない。

保存した[検証用レポート](results/fp16_val.json)と[探索的評価用レポート](results/fp16_test_exploratory.json)は、従来の`eval`と同じ値（四捨五入してそれぞれ34.355→34.447 dB、31.151→31.382 dB）を再現する。[FP16とINT8の共通評価レポート](results/fp16_int8_val.json)では、検証用4枚のPSNRがFP16で34.447 dB、INT8で34.444 dB、モデル容量がそれぞれ1006byteと511byteになった。レポート中の`evaluation_status`は検証用と探索的評価用を区別する。

厳密な最終評価には、**モデル学習にも候補選択にも使用していない新しい画像**を別途用意する。最初の評価前に画像の出典・byte数・SHA-256を別manifestへ固定し、各レコードを`split: "final"`、`path: "final/01_filename.jpg"`の形で記録する。manifestの構造は既存のものと同じで、`gallery`には新しい画像群の出典ページURL、`page_sha256`にはそのページのSHA-256を入れる。その画像を`data/final/`へ配置し、次を実行する。

```sh
cargo run --release -- eval-report data/final_manifest.json data final results/final.json data/gallery_fp16_selected.qsr --exclude-manifest data/gallery_manifest.json
```

このコマンドは新しい画像のハッシュを検証し、既存ギャラリーmanifestとURLまたは画像ハッシュが重なれば拒否する。**未使用だったこと自体はコードだけでは証明できない**ため、取得時期と候補選択に使っていない事実も最終結果とともに記録する。最終画像の結果を見てから候補を選び直した場合、その結果は探索的な比較として扱う。

## 任意の画像で実行する場合

学習用、検証用、評価用の画像をそれぞれ別のディレクトリへ置く。各ディレクトリ直下の `.jpg`、`.jpeg`、`.png` を読み込む。

```sh
cargo run --release -- train data/train model.qsr
cargo run --release -- eval model.qsr data/test 70
cargo run --release -- upscale model.qsr input.jpg output.jpg
cargo run --release -- bench model.qsr input.jpg 5
```

`eval` は評価用画像の中央切り出しを半分に縮小してJPEG圧縮し、そこから2倍に戻した画像と切り出し元を比較する。表示するPSNRはRGB全画素の二乗誤差から計算する。`upscale` で保存するJPEGの再圧縮誤差は、このPSNRには含まれない。

`model.qsr` は `QSR1` 形式のモデルファイルで、重みはlittle-endianのFP16。比較用INT8は`QSI1`形式、FP6は`QSF1`形式。現状の実装はCPU上で動く。

`bench` は同じ拡大済み画像に対するDNN補正を、スカラー経路と使用環境で選ばれる経路で計測する。最後の数値は計測回数で、省略時は5回。JPEGの読み込み・bicubic拡大・JPEG保存の時間は含めず、画素値の最大差も表示する。学習の計算はSIMD化していない。

## FP6の保存形式と比較実験（Issue #2〜#5）

[OCP MX v1.0仕様](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)のFP6 E3M2/E2M3（subnormalを含む）とE8M0 scaleを使用する。最近接・偶数丸め、符号付きゼロ、最小subnormalへ届かない値のゼロへの丸め、範囲外と無限大の符号を保った飽和を実装し、NaN入力は拒否する。標準のscaleは各32重みの絶対値最大から仕様6.3節のpower-of-two規則で選ぶ。全ゼロblockのscaleは1とする。候補探索時だけscale、clipping、丸め方向を変える。

`QSF1`はこのリポジトリ固有のコンテナ形式で、OCPが規定するバイト配置だとは主張しない。16byteのヘッダーにversion=1、block size=32、重み・bias・block・packed byte・タグの数を記録する。続けて15個のE8M0 scale（15byte）、15個の形式タグを詰めた2byte（E3M2=0、E2M3=1）、480個のFP6値を4個ずつ3byteへ詰めた360byte、19個のlittle-endian FP16 bias（38byte）を置く。alignment用の余白はなく、合計431byte。6bit値はbit列の下位から順に詰め、最後の不足bitはゼロで埋める。現行モデルの480重みには不足分がないが、packing関数は任意の個数を扱い、再読込時にpaddingを検証する。scaleと形式タグを含む重みstreamは377byte、**6.283 bit/量子化重み**。ヘッダーとFP16 biasも含めた実ファイルは**6.910 bit/全499パラメータ**。両方の値をレポートへ記録する。activationと積和計算はFP32。FP6値の展開とscale適用は推論時間に含むが、専用FP6演算器は使わない。

```sh
cargo run --release -- quantize-fp6 data/gallery_fp16_selected.qsr data/gallery_fp6_e3m2.qsf e3m2
cargo run --release -- quantize-fp6 data/gallery_fp16_selected.qsr data/gallery_fp6_e2m3.qsf e2m3
cargo run --release -- eval-report data/gallery_manifest.json data val results/fp6_uniform_val.json data/gallery_fp16_selected.qsr data/gallery_fp6_e3m2.qsf data/gallery_fp6_e2m3.qsf --runs 5
cargo run --release -- eval-report data/gallery_manifest.json data test results/fp6_uniform_test_exploratory.json data/gallery_fp16_selected.qsr data/gallery_fp6_e3m2.qsf data/gallery_fp6_e2m3.qsf --runs 5
```

blockごとの形式選択は、検証用4枚のPSNRを基準にE3M2/E2M3を切り替える。選択結果はE3M2が2block、E2M3が13block。形式タグは一律方式にも同じだけ確保するため、どちらも431byteで比較できる。選択記録には試行した15候補と時間を残す。

```sh
cargo run --release -- select-mixed-fp6 data/gallery_fp16_selected.qsr data/gallery_manifest.json data data/gallery_fp6_mixed.qsf results/fp6_mixed_selection.json
cargo run --release -- eval-report data/gallery_manifest.json data val results/fp6_mixed_val.json data/gallery_fp16_selected.qsr data/gallery_fp6_e3m2.qsf data/gallery_fp6_e2m3.qsf data/gallery_fp6_mixed.qsf --runs 5
cargo run --release -- eval-report data/gallery_manifest.json data test results/fp6_mixed_test_exploratory.json data/gallery_fp16_selected.qsr data/gallery_fp6_e3m2.qsf data/gallery_fp6_e2m3.qsf data/gallery_fp6_mixed.qsf --runs 5
```

量子化時の探索では**一律E2M3、32重みblock、同じ`QSF1`推論器**を固定する。各blockでscale指数の補正`-1/0/+1`、clipping比`0.75/0.875/1`、丸め方（最近接偶数・ゼロ方向・ゼロから離れる方向）を試す。標準候補を除く390試行の選択には検証用4枚だけを使い、探索時間と全候補をJSONに残す。推論時の形式も容量も変わらない。

```sh
cargo run --release -- search-fp6 data/gallery_fp16_selected.qsr data/gallery_manifest.json data data/gallery_fp6_e2m3_searched.qsf results/fp6_e2m3_search.json e2m3
cargo run --release -- eval-report data/gallery_manifest.json data val results/fp6_search_val.json data/gallery_fp16_selected.qsr data/gallery_fp6_e2m3.qsf data/gallery_fp6_e2m3_searched.qsf --runs 5
cargo run --release -- eval-report data/gallery_manifest.json data test results/fp6_search_test_exploratory.json data/gallery_fp16_selected.qsr data/gallery_fp6_e2m3.qsf data/gallery_fp6_e2m3_searched.qsf --runs 5
python3 scripts/summarize_results.py results/fp6_uniform_val.json results/fp6_mixed_test_exploratory.json results/fp6_search_test_exploratory.json
```

| モデル | ファイル | 検証用PSNR | 別画像4枚のPSNR* |
| --- | ---: | ---: | ---: |
| nearest neighbor補間 | モデルなし | 32.513 dB | 29.722 dB |
| bicubic補間 | モデルなし | 34.355 dB | 31.151 dB |
| FP16 | 1006byte | 34.447 dB | 31.382 dB |
| 直接生成FP16（8 epoch、seed 42） | 1312byte | 34.335 dB | 31.173 dB |
| 直接生成FP16（64 epoch、seed 42） | 1312byte | 34.381 dB | 31.278 dB |
| 一律E3M2 | 431byte | 34.398 dB | 31.373 dB |
| 一律E2M3 | 431byte | 34.437 dB | 31.379 dB |
| 混合E3M2/E2M3 | 431byte | 34.447 dB | 31.390 dB |
| 一律E2M3・探索後 | 431byte | 34.468 dB | 31.361 dB |

* 別画像4枚は候補選択には使っていないが、初期の学習条件を調べた際に見ているため**盲検の最終評価ではない**。探索後モデルは検証用で改善した一方、別画像では一律E2M3より低下した。探索による汎化性能の改善は確認できていない。速度は各レポートに処理別で記録しており、CPUでの小さな差をFP6の高速化とは解釈しない。

補間方式の比較は、両方式とも同じ中央cropと品質70の合成低解像度JPEGを入力に使い、RGB画素の平均二乗誤差からPSNRを計算した。bicubicはnearest neighborより検証で+1.842 dB、別画像で+1.429 dB。DNNはbicubicを基準に補正するため、DNN単体の改善を見る際はbicubicとの差を使う。画像別の値は `results/interpolation_baselines.json` に保存した。再現コマンドは `cargo run --release --example compare_interpolation data/val data/test > results/interpolation_baselines.json`。

## 低解像度画像から直接2倍生成

`train-direct` は低解像度の3×3 RGB近傍から高解像度の2×2 RGBブロックを直接出す。隠れ層16ユニット、652パラメータ、FP16モデルは1,312 byte。`data/direct_fp16_baseline.qsd` は元の学習画像18枚・64 epoch・seed 42で作成したモデル。既存の補正型DNNはバイキュービック拡大画像にRGB補正を足すが、このモデルの推論では補間済み画像を使わない。

```sh
cargo run --release -- train-direct data/train /private/tmp/direct.qsd 64 200000 70 42 0.0003
cargo run --release -- eval-direct data/direct_fp16_baseline.qsd data/val 70
cargo run --release -- eval-direct-report data/direct_fp16_baseline.qsd data/test /private/tmp/direct-test.json 70
cargo run --release -- upscale-direct data/direct_fp16_baseline.qsd low_res.jpg output_2x.jpg
```

3 seedの比較では、元の18枚・64 epochの直接生成モデルは検証34.378 dB、評価31.292 dBで、同条件の補正型FP16は34.357 dB、31.438 dB。追加131枚を含む64 epochでは直接生成34.354 dB、31.214 dB、補正型34.354 dB、31.401 dB。直接生成は検証で同等だが、評価では補正型より低い。各seed・画像別の値と比較条件は [`results/direct_comparison.md`](results/direct_comparison.md) に記録した。直接生成モデルのFP6化と推論時間の比較は未実施。

## 撮影系列を分けた比較と新しい12枚での最終評価

上の4枚評価は探索中に参照済みだったため、追加画像148枚と既存26枚を撮影月・近いカメラ連番で26組にまとめ、5分割で比較した。各モデルをseed 42/43、24 epoch、1 epochあたり高解像度800,000画素、JPEG品質70、学習率0.0003で学習した。直接生成は1サンプルで2×2画素、残差型は1サンプルで1画素を教師にするので、サンプル数とバッチ数を調整して**教師画素数と更新回数**を揃えた。隠れ層幅を変え、499対492パラメータ、654対652パラメータの2組を比較した。新しい12枚は学習と交差検証に使わず、全174枚で再学習したFP16モデルを評価した。さらに同じ保存済み重みを一律MXFP6 E2M3/E3M2へ量子化して評価した。演算はどちらもFP32。

| 方法 | パラメータ | モデル容量 | 5分割PSNR | 最終12枚PSNR |
| --- | ---: | ---: | ---: | ---: |
| nearest neighbor | — | — | 30.306 | 32.562 |
| bicubic | — | — | 31.914 | 33.405 |
| 残差型・幅16 FP16 | 499 | 1,010 byte | 32.128 | 33.446 |
| 残差型・幅16 FP6 E2M3 | 499 | 429 byte | 32.113 | 33.166 |
| 残差型・幅16 FP6 E3M2 | 499 | 429 byte | 31.989 | 33.369 |
| 直接生成・幅12 FP16 | 492 | 996 byte | 31.931 | 33.350 |
| 直接生成・幅12 FP6 E2M3 | 492 | 430 byte | 29.978 | 31.239 |
| 直接生成・幅12 FP6 E3M2 | 492 | 430 byte | 26.249 | 27.874 |
| 残差型・幅21 FP16 | 654 | 1,320 byte | 32.143 | 33.435 |
| 残差型・幅21 FP6 E2M3 | 654 | 557 byte | 32.119 | 33.408 |
| 残差型・幅21 FP6 E3M2 | 654 | 557 byte | 32.065 | 33.285 |
| 直接生成・幅16 FP16 | 652 | 1,316 byte | 31.943 | 33.358 |
| 直接生成・幅16 FP6 E2M3 | 652 | 560 byte | 29.840 | 31.660 |
| 直接生成・幅16 FP6 E3M2 | 652 | 560 byte | 25.938 | 27.037 |

表は各評価画像の中央最大512×512 cropで、RGB全画素の二乗誤差を合算したPSNR。DNNは2 seedの結果を合算した。FP16同士で同じ画像のPSNR差を先に取って平均すると、直接生成−残差型は約500パラメータ組で5分割 **−0.329 dB**、最終12枚 **−0.187 dB**、約650パラメータ組で **−0.321 dB**、**−0.149 dB**。5分割の26撮影組を単位としたbootstrap 95%区間はそれぞれ [−0.395, −0.272]、[−0.394, −0.256] dB。全10組の「分割×seed」と全26撮影組の平均で残差型が上だった。最終12枚でも直接生成が上だったのは各サイズで24件（12枚×2 seed）中1件。ただし同じ撮影系列の画像や2 seedは独立標本ではない。

PSNRは高いほど元画像に近い。最終12枚では幅16の残差型FP16がbicubicより **+0.042 dB**、幅12の直接生成型FP16が **−0.055 dB**。E2M3量子化による5分割の低下は残差型で **0.015〜0.024 dB**、直接生成型で **1.953〜2.103 dB**。最終12枚でも直接生成型は **1.699〜2.111 dB**低下した。残差型・幅21は最終12枚のE2M3低下が **0.028 dB**だが、幅16は **0.281 dB**下がった。E3M2は直接生成型でさらに大きく低下した。幅や量子化形式だけで一律に耐性を判断できず、同じ重みのFP16とFP6を対にして見る必要がある。直接2倍生成は実装できたが、この小さな1隠れ層モデルと一律FP6レシピではbicubic後の補正型を超えなかった。FP6値は読み込み時にFP32へ展開しており、FP6専用演算の速度を示さない。FP16の最終結果を見た後でFP6を評価したため、FP6について新たな盲検評価をしたとは扱わない。

比較用の`QSM6`ファイルは16byteヘッダー（version、残差型/直接生成型、隠れ層幅、FP6形式、接続重み・bias・block数、予約領域）の後に、32重みblockごとのE8M0 scale、下位bitから詰めた6bitの接続重み、FP16 biasを置く。形式はモデル全体で一律なのでblock別形式タグはない。FP16の`QSM1`とともに`eval-matched`で読み込める。量子化には`quantize-matched-fp6 <model.qsm> <model.qsm6> <e2m3|e3m2>`を使う。

画像のSHA-256と分割は [`results/grouped_cv_manifest.json`](results/grouped_cv_manifest.json) と [`results/final_manifest.json`](results/final_manifest.json) に固定した。学習画像と最終画像の完全一致はなく、9×8 dHashの距離5以下もなかった。撮影場面の独立性まで証明する検査ではない。集計値、分割・seed別値、最終画像別値と量子化結果は [`results/grouped_cv_results.json`](results/grouped_cv_results.json)、[`results/grouped_cv_analysis.json`](results/grouped_cv_analysis.json)、[`results/final_evaluation.json`](results/final_evaluation.json)、[`results/final_analysis.json`](results/final_analysis.json)、[`results/matched_fp6_results.json`](results/matched_fp6_results.json) に保存した。最終評価に使った8個のFP16モデルと16個のFP6モデルは [`models/matched_final/`](models/matched_final/) にあり、JSON中のSHA-256と一致する。再現にはGit管理外の元画像が必要。

```sh
python3 scripts/stage_combined_train.py data/train data/train_inbox /private/tmp/qse-combined-train data/train_inbox_manifest.json
python3 scripts/make_grouped_cv.py data /private/tmp/qse-grouped-cv-174 results/grouped_cv_manifest.json --inbox-manifest data/train_inbox_manifest.json
cargo build --release
python3 scripts/run_grouped_cv.py target/release/quant-search-exp results/grouped_cv_manifest.json data /private/tmp/qse-grouped-cv-174 /private/tmp/qse-matched-cv results/grouped_cv_results.json
for fold in 0 1 2 3 4; do target/release/quant-search-exp eval-direct-report data/direct_fp16_baseline.qsd /private/tmp/qse-grouped-cv-174/fold-$fold/eval /private/tmp/qse-fold$fold-baselines.json 70; done
python3 scripts/analyze_grouped_cv.py results/grouped_cv_manifest.json /private/tmp/qse-matched-cv /private/tmp results/grouped_cv_analysis.json
python3 scripts/check_final_overlap.py results/grouped_cv_manifest.json data data/final_inbox
python3 scripts/run_final_eval.py target/release/quant-search-exp results/grouped_cv_manifest.json results/final_manifest.json results/grouped_cv_results.json data /private/tmp/qse-final-eval results/final_evaluation.json
python3 scripts/analyze_final.py results/final_evaluation.json results/final_analysis.json
python3 scripts/run_matched_fp6.py target/release/quant-search-exp /private/tmp/qse-matched-cv /private/tmp/qse-grouped-cv-174 models/matched_final /private/tmp/qse-final-eval data/final_inbox /private/tmp/qse-matched-fp6 results/matched_fp6_results.json
```
