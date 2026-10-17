# BiomedicalIR-mini

PubMed 摘要搜尋引擎的最小可執行版本：位置倒排索引、布林查詢（AND / OR / NOT / 括號 / 片語）、BM25 排序、拼字建議，
A–D 四種前處理條件的 Zipf 分析，以及從零訓練的 CBOW / Skip-gram 詞向量。
檢索與 Zipf 統計只用 Python 3.10+ 標準函式庫；Zipf 畫圖（`--plot`）需要 matplotlib，詞向量（`embed`）需要 numpy，
網頁介面（`serve`）需要 flask。

## 執行

```bash
git clone https://github.com/raindrop1102/biomedical-ir-mini.git
cd biomedical-ir-mini
python -m pip install numpy matplotlib flask             # 選用：只用搜尋與 Zipf 統計時不需要
python -m bioir search                                   # 互動模式，預設載入 data/glp1_1000
python -m bioir search GLP-1 -k 5
python -m bioir search '"type 2 diabetes" AND (semaglutide OR tirzepatide) NOT mice'
python -m bioir search --corpus my.csv cancer            # .jsonl / .csv / PubMed .xml
python -m bioir zipf                                     # A–D 摘要統計、分區擬合、前 10 名、CF/DF/IDF 表
python -m bioir zipf --out results --plot               # 另存 CSV 與 zipf.png（疊圖、擬合、殘差）
python -m bioir zipf --no-numbers --terms "semaglutide,insulin,Patients"   # 排除純數字；CF/DF/IDF 表加詞
python -m bioir zipf --compound --baseline D            # 保留連字號複合詞（同原專案）；以 D 為基準
python -m bioir embed train -o cbow.npz                   # CBOW，100 維、window 5、20 輪（約 75 秒）
python -m bioir embed train --arch skipgram -o sg.npz
python -m bioir embed similar cbow.npz semaglutide insulin GLP-1
python -m bioir embed compare --out embed_results --plot # 6 個模型：架構 × window × seed，約 10 分鐘
python -m bioir embed project cbow.npz sg.npz --out proj --plot   # PCA / t-SNE 投影圖
python -m bioir serve                                    # 網頁介面 http://127.0.0.1:5000（搜尋、Zipf、詞向量）
python -m bioir fetch pmids.txt --email you@example.com -o abstracts.jsonl
python -m bioir fetch --query '"GLP-1"[tiab] AND hasabstract' -n 1000 --email you@example.com
python -m unittest discover tests
```

查詢語法：相鄰詞預設為 AND；運算子須大寫（小寫 `or`、`not` 視為一般詞）；`"..."` 為片語。
查詢上限：整串 500 字元、最多 20 個詞（片語內的詞也算，運算子與括號不算）、單詞 50 字元，超過時丟 `ValueError`。

`pmids.txt` 可用空白、逗號、分號或換行分隔，也接受 `https://pubmed.ncbi.nlm.nih.gov/12345678/` 網址；
`--query` 走 ESearch（relevance 排序）取前 `-n` 筆 PMID 再 EFetch。沒有摘要的文章會被略過並列出，
要湊滿篇數請在查詢式加 `AND hasabstract`。設定環境變數 `NCBI_API_KEY` 即會附上 API key。

### 程式內使用

```python
from bioir import Index, highlight, load

index = Index(load("abstracts.jsonl"))
results = index.search('"type 2 diabetes" GLP-1', k=5)
for score, doc in results.hits:
    spans = results.spans[doc.pmid]          # {"title": [(start, end), ...], "abstract": [...]}
    html = highlight(doc.abstract, spans["abstract"])   # 全文 html.escape，命中處包 <mark>
```

`spans` 是查詢中每個片語（NOT 以外）在原文的字元位置，以整段片語為單位：查 `GLP-1` 時
`GLP-1RAs` 的 `GLP-1` 會被標示，但單獨出現的 `1` 不會。只有回傳的前 k 篇會計算 spans。

## 資料與版權

`data/glp1_1000/` 是 1,000 篇提到 GLP-1 的英文 PubMed 摘要（`abstracts.jsonl`）、對應的 `pmids.txt`
與收集紀錄 `manifest.json`（查詢式、隨機種子、篩選條件、各檔 SHA-256）。資料以 NCBI E-utilities
（ESearch + EFetch）取得，在最新的 5,000 筆候選中以固定種子抽樣，不是 GLP-1 文獻的代表性隨機樣本。

**摘要的著作權屬於原作者與出版者**，本儲存庫只為教學與研究示範而保留，並未重新授權；每篇都可由 PMID
連回 `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`。若要自行取得同一批資料，可用
`python -m bioir fetch data/glp1_1000/pmids.txt --email 你的email -o abstracts.jsonl` 重新下載。

`results/embed_glp1/` 是 `python -m bioir embed compare --out results/embed_glp1 --plot` 的完整輸出
（六個模型、CSV、投影圖、manifest），可直接給網頁的詞向量頁使用。

## 結構

```
bioir/
  text.py  porter.py  corpus.py  pubmed.py     共用基礎（只用標準函式庫）
  search/  index.py  query.py  spelling.py     檢索
  zipf/    stats.py  plot.py                   Zipf 分析（plot.py 才需要 matplotlib）
  embed/   train.py  model.py  evaluate.py     詞向量（需要 numpy）
  web/     app.py  templates/  static/         網頁介面（需要 flask）
  __main__.py                                  命令列：search / zipf / embed / serve / fetch
```

依賴只能單向：`search`、`zipf`、`embed` 都只 import 共用基礎，彼此不互相 import；`web` 是唯一同時使用三者的
最上層。numpy 只在 `embed` 內使用、flask 只在 `web` 內使用，命令列也只在執行對應指令時才載入。三者共用同一個 tokenizer，因此 GLP-1 斷詞規則改一次就同時生效；
各自的前處理流程不同：檢索去停用詞＋詞幹化、Zipf 依 A–D、詞向量保留停用詞且不詞幹化。

| 檔案 | 職責 |
|---|---|
| `bioir/text.py` | 斷詞、正規化、停用詞、分句、純數字判斷；`analyze()`（檢索）、`tokens()`（Zipf B、詞向量）、`highlight()` |
| `bioir/porter.py` | Porter (1980) stemmer，含作者 BLI / LOGI 修正 |
| `bioir/corpus.py` | `Document` 與 JSONL / CSV / PubMed XML 讀寫、SHA-256 manifest |
| `bioir/pubmed.py` | NCBI ESearch / EFetch：限速、重試、逾時、API key |
| `bioir/search/query.py` | 遞迴下降解析器 → 語法樹；查詢長度上限 |
| `bioir/search/index.py` | 位置倒排索引、布林比對、片語、BM25、命中字元位置 |
| `bioir/search/spelling.py` | OSA 編輯距離（含相鄰字母對調）＋ 語料詞頻 |
| `bioir/zipf/stats.py` | 條件 A–D 前處理、cf / df / idf、全域與 10/80/10 分區 OLS 擬合、CF/DF/IDF 對照、CSV 輸出 |
| `bioir/zipf/plot.py` | A–D 疊圖＋k = 1 參考線、基準條件擬合圖、殘差圖 |
| `bioir/embed/train.py` | CBOW / Skip-gram 負取樣訓練（NumPy）、語料分句 |
| `bioir/embed/model.py` | `Embeddings`：近鄰、相似度、單一 `.npz` 存取 |
| `bioir/embed/evaluate.py` | 隨機詞對 cosine 健全性檢查、近鄰／詞對表、top-5 重疊、六模型比較實驗、manifest |
| `bioir/embed/projection.py` | PCA、exact t-SNE、trustworthiness（NumPy，不需 scikit-learn） |
| `bioir/embed/plot.py` | 投影圖：依預先指定的詞組上色，標籤自動避開其他標籤與點 |
| `bioir/__main__.py` | 命令列介面 |

## Zipf 分析

| 條件 | 前處理 |
|---|---|
| A | 轉小寫，以空白切分（標點仍黏在字上） |
| B | A ＋ 斷詞器 |
| C | B ＋ 移除停用詞 |
| D | C ＋ Porter 詞幹化：**與檢索索引的詞完全相同**（有測試保證） |

只分析摘要；依 cf 由大到小排名，同頻時依字母序；對全部排名做
log10(cf) = intercept + slope · log10(rank) 的最小平方法，exponent = −slope，另回報 R²、RMSE（log10 單位）；
idf = ln(N / df)。迴歸以純 Python 計算，不需要 numpy。

| 選項 | 作用 |
|---|---|
| `--baseline B` | 基準條件（預設 B）：分區擬合、CF/DF/IDF 表與圖的第 2、3 格都用它 |
| `--no-numbers` | 所有條件都排除純數字（`2`、`0.05`、`10,000`、`45%`、`(2)`）；`glp-1`、`p53` 不受影響 |
| `--terms a,b` | CF/DF/IDF 表在基準條件前 10 名之外加入這些詞；每個詞照基準條件的前處理轉換（D 下 `Patients` → `patient`），停用詞或語料中沒有的詞顯示為「—」 |
| `--compound` | 改用原專案的斷詞（保留 `glp-1ras`、`patient's`），D 會先去掉所有格 `'s` 再詞幹化 |

**分區擬合（前 10%／中間 80%／後 10% 的排名）**：另列每區「不同 cf 值的個數」。尾端通常全是 cf = 1 的平台，
只有 1 種頻率，斜率 0、R² 無法定義（顯示 n/a）；頭部最接近 k = 1（GLP-1 語料 B：k 0.92、R² 0.991），
中段明顯較陡。均分三等分會把頭部和中段混在一起，看不出這兩個現象（改自 bioscope-ir-hw2 的做法；原專案用三等分）。

**圖（`zipf.png`）**：① A–D 疊在同一張 log-log 圖，由淺到深依序代表四個累進條件，基準條件加粗，並畫出
通過基準條件第 1 名頻率的 k = 1 參考線（f(1)/rank）；② 基準條件的觀測值、全域 OLS 線與參考線；③ OLS 殘差。
② ③ 以灰線標出分區邊界。殘差呈系統性的弧形：R² 高不代表真的是 power law。

**CF/DF/IDF 表**：cf/df 是含該詞的文件中平均出現幾次。cf 高但 df 低的詞集中在少數文件反覆出現；
idf 只由 df 決定，例如 `the` 出現在 979 篇，idf 0.021，`semaglutide` 241 篇，idf 1.423。

輸出檔：`summary.csv`、`regions.csv`、`terms_A–D.csv`（含 cf_per_df）、`compare_terms.csv`，加 `--plot` 另有 `zipf.png`。

**斷詞器的影響**：預設斷詞把 `GLP-1` 切成 `glp`、`1`，所以數字（`1`、`0`、`2`）會擠進 C、D 的前幾名
（可用 `--no-numbers` 排除），B–D 的詞彙量也比保留複合詞時少（GLP-1 語料 D：預設 9,282 個詞，`--compound` 12,671 個）。
這是刻意的取捨，報告時要寫清楚。

**與原專案交叉驗證**：在原專案 Zipf 使用的語料（`analysis_data/corpus/abstracts.jsonl`）上以 `--compound` 執行，
並換成原專案的 37 個停用詞時，A–D 四個條件的 tokens、詞彙量、hapax 與斜率都和 `analysis_data/results/summary.csv`
完全相同；用本專案的停用詞表時只有 C、D 不同（停用詞表較大）。

**與 bioscope-ir-hw2 比較**：對方所有條件都排除純數字，條件 B 等於我們的 `--compound --no-numbers`。
在對方語料上，B 的 tokens 161,039 vs 161,515（差 0.3%），頭部 k 0.931 vs 0.927；詞彙量 11,152 vs 10,691，
差異來自對方把撇號切開（`patient's` → `patient`、`s`）並只認 ASCII 字母。對方的條件 A 是「斷詞＋保留獨立標點」，
與我們（及原專案）的「以空白切分」定義不同，A 的數字不可直接比較。

## 詞向量（embed）

NumPy 從零實作 word2vec 的 CBOW 與 Skip-gram（負取樣 5 個、依 cf^0.75 抽樣、高頻詞降採樣 1e-3、
每個目標隨機縮小 window、學習率線性遞減），沒有使用 gensim 或 PyTorch。

**預設值來自原專案模型塌縮的消融實驗**：原專案 `models/glp1_cbow` 的隨機詞對 cosine 平均 0.93
（93% 的向量能量在同一方向），所以介面上 0.97 的相似度沒有意義。逐一改變因素的結果：

| 改變 | 隨機詞對 cosine |
|---|---|
| 原設定（5 輪、每 128 筆更新、前處理 D） | 0.930 |
| 每 1 筆或 16 筆就更新 | 0.930（批次不是原因） |
| 前處理改為保留停用詞、不詞幹化 | 0.906 |
| 上下文詞拿完整梯度（不除以上下文長度） | 0.719 |
| 20 輪 | 0.538 |
| 20 輪＋完整梯度＋保留停用詞 | 0.176 |

原因是**有效訓練量不足**，因此本實作：CBOW 每個上下文詞拿完整梯度（同 word2vec）、預設 20 輪、
min_count 3、保留停用詞且不詞幹化、保留 `glp-1` 這類複合詞、丟掉純數字（`--split` 改用檢索的斷詞）。

**健全性檢查**：訓練完自動計算 2,000 個最常見詞之間 20,000 組隨機詞對的 cosine，平均超過 0.5 判為未通過，
記錄在模型的 metadata，`embed train` 以結束碼 2 提示。相似度要和這個基準一起看；`pairs.csv` 的 z 是
「比隨機詞對平均高幾個標準差」。

**`embed similar` 預設隱藏出現少於 10 次的近鄰**（`--min-count`）：低頻詞向量不穩定，常碰巧靠近高頻詞，
例如 `weight` 不過濾時前幾名是 `5-10`、`contouring`、`hearing`（各只出現 6–9 次）。

**GLP-1 語料的比較實驗**（`embed compare`，100 維、20 輪；詞彙 6,324 個）：

| 模型 | 隨機詞對 cosine 平均 / p95 | semaglutide 的前 5 近鄰 | 訓練秒數 |
|---|---|---|---|
| CBOW window 5 | 0.123 / 0.523 | tirzepatide, liraglutide, orforglipron, dulaglutide, retatrutide | 68 |
| Skip-gram window 5 | 0.248 / 0.463 | liraglutide, dulaglutide, tirzepatide, lixisenatide, ecnoglutide | 160 |

兩種架構都學到 `glp-1` 與 `glp1`、`glucagon-like-peptide-1` 是同一概念的不同寫法。前 5 近鄰重疊率：

| 變動 | CBOW 查詢詞 | CBOW 低頻詞 | Skip-gram 查詢詞 | Skip-gram 低頻詞 |
|---|---|---|---|---|
| window 2 → 5 | 56% | 23% | 46% | 20% |
| seed 42 → 7 | 74% | 54% | 84% | 63% |

window 對近鄰的影響比 seed 大；低頻詞的近鄰最不穩定。與 bioscope-ir-hw2（gensim、不同語料）趨勢一致：
對方 window 2 → 5 的查詢詞重疊為 CBOW 66%、Skip-gram 50%，seed 換成 7 時低頻詞為 57.1%、68.6%。
重疊率與 cosine 都不是語意準確率；這是單一主題的 1,000 篇摘要，詞向量只供教學示範。

**投影圖**（`results/embed_glp1/projections.png`）：預先指定 4 組共 27 個詞（GLP-1 藥物、受體、血糖、
體重與食慾；在詞彙中的 23 個），向量先做 L2 正規化，再分別以 PCA 和 t-SNE 投影到 2 維。
t-SNE 為 exact 版本（perplexity 依詞數取 (n−1)/3，上限 15；1,000 次迭代，前 250 次 early exaggeration 12），
trustworthiness（k = 5）與 scikit-learn 的實作在測試資料上數值相同，t-SNE 品質也相當（0.819 vs 0.818）。

| 模型 | PCA 解釋變異 | PCA trustworthiness | t-SNE KL | t-SNE trustworthiness |
|---|---|---|---|---|
| CBOW window 5 | 37% | 0.88 | 0.17 | 0.87 |
| Skip-gram window 5 | 29% | 0.90 | 0.17 | 0.88 |

兩種模型、兩種投影中，五個 GLP-1 藥物都聚在一起，`body`、`loss`、`weight` 也大致相鄰；
其他詞組較分散，例如 Skip-gram 的 t-SNE 裡 `glp-1` 靠近 `glucagon`，離 `receptor`、`agonist` 較遠。
每格各自擬合，格與格之間的座標軸不能比較；投影只是幾個詞的示意，不是模型品質的檢驗。
詞組定義在 `bioir/embed/projection.py` 的 `PROJECTION_GROUPS`；換語料時若找不到至少三個詞就略過投影。

**比較實驗的完整輸出**在 `results/embed_glp1/`：`summary.csv`、`neighbors.csv`、`pairs.csv`、`overlap.csv`、
`projections.csv`、`projection_quality.csv`、`projections.png`、`manifest.json`（語料 SHA-256、共同設定、
固定詞表、Python / NumPy 版本）與 `models/*.npz`（六個模型，可用 `embed similar` 直接查詢）。

## 網頁介面（web）

`python -m bioir serve`（`--corpus`、`--results`、`--host`、`--port`）啟動 Flask 開發伺服器，三個頁面：

| 頁面 | 內容 |
|---|---|
| 文獻搜尋 `/` | 單一查詢框（AND / OR / NOT / 括號 / 片語）、BM25 排名、命中片語高亮（摘錄＋完整摘要）、拼字建議、分頁、跳至文章側欄 |
| Zipf 分析 `/zipf` | 基準條件、`--no-numbers`、`--compound` 與加入詞的表單；A–D 總表、10/80/10 分區、前 50 名、CF/DF/IDF 對照、三格圖；CSV 下載 |
| 詞向量 `/embedding` | 模型健全性、CBOW 與 Skip-gram 近鄰並排（可調近鄰最少出現次數）、兩詞相似度與 z 值、投影圖與 trustworthiness、比較實驗摘要 |

**設計**：

- **只讀**：語料、索引、拼字字典與詞向量模型在啟動時載入一次（GLP-1 語料約 2 秒），之後請求只讀取，
  不需要鎖；Zipf 的計算與圖依選項快取（第一次約 1–2 秒）。上傳語料、下載 PubMed、訓練模型都留在命令列，
  所以不會有原專案「一位使用者換語料、所有人一起換」或網頁內背景執行緒的問題。
- **高亮**：使用搜尋結果的命中字元位置，先 `html.escape` 全文再加 `<mark>`；`GLP-1` 會整段標示，單獨的 `1` 不會。
  另外設定 `X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy` 回應標頭。
- **外觀沿用原專案**：`static/style.css`、`zipf.css`、`theme.js`、`templates/theme_init.html`、導覽列與搜尋頁的
  「跳至文章」腳本都直接複製；新增的元件樣式集中在 `static/mini.css`。支援深色模式與 375px 手機寬度。
- 詞向量頁讀取 `--results`（預設 `results/embed_glp1`）的 `cbow_w5_s42` 與 `skipgram_w5_s42`；
  資料夾不存在或沒有 numpy 時，頁面會說明要執行的指令，其他頁面照常使用。
- `app.run` 是開發用伺服器；要部署時以 `bioir.web.create_app` 交給 waitress / gunicorn 等 WSGI 伺服器。

## 與原專案相比的設計決策

1. **生醫識別詞正規化（最重要）**：連續字母、連續數字各自是一個 token，連字號與字母／數字交界都會切開，查詢詞則以片語比對。
   `GLP-1`、`GLP1`、`GLP-1RAs`、`glp-1 RA` 都會變成相鄰的 `glp · 1 · ra`。
   在 GLP-1 語料上（以標題＋摘要的 regex 作為標準答案）：

   | 查詢 | 原版召回率 | 新版召回率 | 新版精確率 |
   |---|---|---|---|
   | GLP-1 | 74.3% | 100% | 99.9% |
   | GLP1 | 1.6% | 100% | 99.9% |
   | SGLT2 / SGLT-2 | 47.8% / 13.2% | 100% | 100% |
   | HbA1c | 85.7% | 100% | 100% |

2. **標題也進索引**：標題和摘要之間留一個位置的間隔，片語不會跨欄位比對。
3. **BM25**（k1=1.2，b=0.75）取代未做長度正規化的 TF-IDF；用 `heapq.nlargest` 取前 k 名，分數相同時依 PMID 排序。
4. **位置索引**：支援片語查詢；停用詞移除後仍保留原位置，`"type 2 diabetes"` 不會比對到 `type 2 and diabetes`。
5. **布林運算用 Python `set`**：由 C 實作，比手寫的合併迴圈短且快；AND 會先從最小的集合開始交集。
6. Porter 規則表在模組載入時就預先排序（原版每次呼叫都重新排序），並保留 `lru_cache`。

## 工程面（參考 bioscope-ir-hw2 移植）

檢索演算法沒有變動；以下是周邊的穩健性。

| 項目 | 做法 |
|---|---|
| 限速 | 全程序共用一個時鐘並加鎖，任兩次請求至少間隔 0.36 秒（NCBI 無 key 上限 3 次/秒），多執行緒也成立 |
| 重試 | HTTP 429 與 5xx、連線錯誤、逾時：最多試 5 次，間隔 1、2、4、8 秒；其他 4xx 立即失敗 |
| 逾時 | `urlopen(timeout=30)`，urllib 對連線與每次讀取都套用 |
| API key | 只從 `NCBI_API_KEY` 讀取，放在 POST body 而非網址；錯誤訊息只含 HTTP 狀態碼或例外類型，並以 `from None` 切斷例外鏈，不會帶出網址或 key |
| 結構化摘要 | 保留 `Label`：`BACKGROUND: ... RESULTS: ...` |
| manifest | `fetch` 在 `abstracts.jsonl` 旁寫 `abstracts.manifest.json`（查詢式、PubMed 轉譯、時間、要求/取得篇數、未取得的 PMID、SHA-256）。`load()` 看到同名 manifest 會核對 SHA-256，不符時發出 `UserWarning`（仍會載入） |
| 高亮 | `highlight()` 先 `html.escape` 全文再插入 `<mark>`，標題或摘要裡的 `<script>` 只會變成文字 |
| 測試不連網 | `tests/test_pubmed.py` 以假的 `urlopen` 與 `sleep` 測重試、限速、key 不外洩、ESearch、分批 EFetch 與 `fetch` 指令；預設的假 `urlopen` 一被呼叫就讓測試失敗 |
| Porter 參考比對 | `tests/test_porter_reference.py` 用 NLTK 逐字比對整個語料的 ASCII 詞彙（11,712 個），差異為 0；未安裝 nltk 或找不到語料時 skip |
| CI | `.github/workflows/tests.yml`：push / pull_request 時以 Python 3.12 執行 unittest |

**Porter 參考實作的模式**：NLTK 的 `ORIGINAL_ALGORITHM` 照 1980 論文，沒有作者後來的 BLI / LOGI 修正
（本語料 63 個詞不同，如 `cardiology`）；預設的 `NLTK_EXTENSIONS` 另加 NLTK 自己的規則（120 個詞不同，如 `aged`）。
`MARTIN_EXTENSIONS` 正是 tartarus.org 公布、本專案實作的版本，因此用它當參考。

**XML 解析不用 defusedxml 的取捨**：為了維持「只用標準函式庫」，仍用 `xml.etree.ElementTree`。
它不會載入外部實體（XXE、外部 DTD 都不會發生），而 Python 3.10+ 附的 expat ≥ 2.4.1 對實體膨脹（billion laughs）
有內建上限。另外 `parse_pubmed_xml()` 遇到 `<!ENTITY` 宣告直接拒絕（EFetch 輸出從不宣告實體），
XML 內的 `<ERROR>` 與格式錯誤都轉成 `ValueError`。若要解析來路不明的大量 XML，仍建議改用 defusedxml。

## 已知限制

- 同義詞（`glucagon-like peptide-1` ↔ `GLP-1`）需要 MeSH / UMLS 對照表，純字面比對做不到。
- 同形異義無法分辨：語料中有一篇的 `GLP1` 指的是靈芝多醣，仍會被找到。
- 索引只存在記憶體，每次啟動都重建（1,000 篇約 0.3 秒）；資料量大時需要持久化與壓縮。
- 尚未建立含相關性判斷（qrels）的評估集，排序品質還沒有量化。
- 高亮以片語為單位，但不區分它屬於 OR 的哪個分支；NOT 內的詞一律不標示。
