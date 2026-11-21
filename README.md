# 图纸分块定位与缩略图 API

两个相关的能力：

1. **缩略图**（`POST /thumbnail`）：把一张二值图纸缩成尽量小的分块缩略图，
   同时保留分开的笔画、闭合轮廓和外部白底。
2. **分块定位**（`POST /align`）：把大幅图纸切块扫描得到的多张图块，按**已声明的
   重叠图对**和每对的整数平移搜索窗，求出各块在整图上的位置；只有全部图对同时达标
   且相对布局唯一时，才返回以**首图左上角为原点**的拼接坐标。

技术栈：FastAPI + NumPy（Pillow 仅用于 PNG 解码/编码）。提供 Docker Compose 与可交互的
`/docs`。

## 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

在 `/docs` 演示图纸上传，两种起法：

* **缩略图**：打开 `POST /thumbnail` → Try it out → 选择
  `scripts/make_thumbnail_demo.py` 生成的 `demo/thumbnail.png` → Execute。
* **对齐**：打开 `POST /align`，按顺序上传 `demo/tile0.png …`，把
  `demo/spec.json` 的内容粘贴到 `spec`。

生成演示数据：

```bash
python scripts/make_thumbnail_demo.py demo   # demo/thumbnail.png + thumbnail.sh
python scripts/make_demo.py demo             # demo/tile*.png + spec.json + request.sh
docker compose up --build                    # 另一个终端
bash demo/thumbnail.sh                       # 期望 status = "reduced"
bash demo/request.sh                         # 期望 status = "aligned"
```

不用 Docker 时：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/thumbnail_client.py --base-url http://localhost:8000
python scripts/demo_client.py --base-url http://localhost:8000
pytest
```

## 缩略图 `POST /thumbnail`

`multipart/form-data`，单字段 `image`：**8 位灰度 PNG**（颜色类型 0、非隔行、
每边 ≤ 256 px），像素**只允许 0（黑）或 255（白）**。

### 规则

* 缩小倍数 `s` 取 2～6 的整数。以 `s` 为边长、从左上角开始分块，
  右边和下边不足一块的**补白**（视为白色）。
* 全图共用一个正整数阈值 `t`，`1 ≤ t ≤ s*s`（每块像素数）。块内黑点数 `≥ t`
  时输出黑点，否则输出白点。
* 黑色按**八邻接**、白色按**四邻接**连通；图外视为连成一片的白底。
* 按**原像素所在的块**关联两图的同色连通区域：一个原图区域与一个缩略图区域配对，
  当且仅当该原图区域有像素落在后者所覆盖的块里且颜色一致。黑、白两种颜色都要求
  **一一对应**（原图每个区域恰好配一个缩略图区域，反之亦然），外部白底也必须彼此
  对应（缩略图外部白底不能变成被黑块包围的湖，反之亦然）。
* 返回**合格的最大缩小倍数**，同一倍数下取**最小合格阈值**；2～6 全不合格时
  `status=keep_original`，`image_base64`、`factor`、`threshold` 均为 `null`，
  `message` 提示保留原图。

### 响应示例

```jsonc
{
  "status": "reduced",
  "message": "采用最大合格缩小倍数 5，共用阈值 6；…",
  "factor": 5,
  "threshold": 6,
  "source": {
    "width": 256, "height": 256,
    "black_pixels": 16420, "white_pixels": 49116,
    "black_regions": 7, "enclosed_white_regions": 42
  },
  "thumbnail": {
    "factor": 5, "threshold": 6, "block_pixels": 25,
    "columns": 52, "rows": 52,
    "padded_columns": 4, "padded_rows": 4,
    "output_width": 52, "output_height": 52,
    "black_blocks": 510,
    "black_regions": 7, "enclosed_white_regions": 42,
    "black_one_to_one": true, "white_one_to_one": true
  },
  "evaluated_factors": [6, 5, 4, 3, 2],
  "rejected": [
    { "factor": 6, "detail": "阈值 1～36 全部未通过：…无法一一对应" }
  ],
  "image_base64": "iVBORw0KGgo…"
}
```

`image_base64` 解出的 PNG 为 8 位灰度、颜色类型 0、非隔行，像素仅 0/255；
尺寸为 `columns × rows`（即分块数）。

### 错误

`400`：不是 PNG、非 8 位、非颜色类型 0、隔行、每边超过 256 px、空文件、
含 0/255 以外的像素（字段 `detail` 指出具体原因）。缺少 `image` 字段由 FastAPI
返回 `422`。

### 判定要点

* 阈值只会在渲染真的变化处取值：`1` 与每个块黑点数 `+1`（且 ≤ s*s），
  因此扫描的阈值个数不超过不同块计数数，结果仍是“最小合格阈值”。
* 关系检查是逐块的，不要求逐像素对齐；这也是题面“按原像素所在的块关联”的字面含义。
* 黑色八邻接/白色四邻接保证数字图像里的 Jordan 性质：闭合轮廓围出的白色不会被
  连通到图外，黑色笔画的对角相接会合并成一个区域。
* 想做得更小，只能依赖输入的笔画本身在更大的块网格下仍可分离；无法分离时返回
  `keep_original`，不做“尽量近似”的猜测。

## 对齐 `POST /align`

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `images` | 3～5 张 **8 位灰度 PNG**（PNG 颜色类型 0、非隔行、每边 ≤ 1024 px）。上传顺序即图号 `0..n-1`，图 0 为基准。 |
| `spec` | JSON 字符串，结构见下。 |

`GET /health`、`GET /docs`（Swagger UI）、`GET /redoc`、`GET /openapi.json`。

### spec 结构

```jsonc
{
  "pairs": [
    { "a": 0, "b": 1,
      "window": { "dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2 } },
    { "a": 1, "b": 2, "candidates": [[-450, 450], [-451, 450]] }
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```

* `pairs`：声明哪些图对重叠。每对三选一之外的两种写法**必须且只能**给一个：
  * `window`：闭区间矩形，`(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81`；
  * `candidates`：至多 81 个显式整数平移，自动去重。
* `min_overlap_pixels`：整组共用的最少重叠像素数。
* `max_mean_abs_diff`：整组共用的平均绝对灰度差（MAD）上限，0～255。

### 坐标约定

图对 `(a, b)` 的平移 `(dx, dy)` 表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处：

```
(x_b, y_b) = (x_a + dx, y_a + dy)
```

返回的 `positions[i]` 是第 i 张图左上角在整图坐标中的位置，图 0 恒为 `(0, 0)`。

### 判定逻辑

1. **连通性**：把声明的图对当作无向边，若某图块无法从图 0 到达，立即返回
   `disconnected`，并在 `rescan_image_id` 指出建议补扫的一块（取连通度最低者，平局取小编号）。
2. **图对达标**：对每对枚举搜索窗内的每个整数平移，计算重叠矩形内的 MAD。一个平移被接受需同时满足
   *重叠像素数 ≥ `min_overlap_pixels`* 且 *MAD ≤ `max_mean_abs_diff`*。
   任一对没有任何被接受的平移 → `no_solution`，`failing_pairs` 给出原因
   （`insufficient_overlap` 或 `mean_abs_diff_too_high`）与最佳尝试值。
3. **全局布局**：每对可能有多于一个达标平移。求解器在图对图上建生成树（优先候选少的边），
   枚举树边候选组合，剪掉任何与已放置节点冲突的组合，并在叶子处校验其余冗余图对；
   不同的位置向量即不同的布局。

### 响应

HTTP 状态码只表示请求是否合法；对齐结论在 `status` 字段（均返回 200）：

| `status` | 含义 | 关键字段 |
| --- | --- | --- |
| `aligned` | 所有图对同时达标且布局唯一 | `positions`、`pairs`、`mosaic_bounds`、`layout_unique=true` |
| `disconnected` | 有图块连不到首图 | `unreachable_image_ids`、`rescan_image_id` |
| `no_solution` | 无法同时对齐 | `failing_pairs`（单对不达标）或空（各对单独达标但互相矛盾） |
| `ambiguous` | 多解，位置不唯一 | `uncertain_image_ids`、`alternative_layouts`、`search_truncated` |

`pair_candidate_counts` 给出每个声明图对的达标平移个数，便于诊断阈值是否过松。

错误返回：`400` 图像非法（格式/位深/颜色类型/尺寸/数量/`Content-Type`）或 `spec` 不是 JSON；
`422` `spec` 结构不合法、图号越界、图对重复。

## 实现要点

* **缩略图的连通性**：`app/topology.py` 用行程编码 + 并查集做连通域标记（黑色八邻接、
  白色四邻接），再把触边的白色区域统一记为外部白底。随机小图上与逐像素 BFS 的结果
  完全一致（见测试）。
* **缩略图的判定**：`app/thumbnail.py` 只对“渲染会变化”的阈值做候选，
  256×256 的随机噪声（最坏情况，2～6 全不通过）实测约 1 秒；典型图纸不到 0.1 秒。
* **对齐解码**：先读 PNG 的 `IHDR` 校验位深=8、颜色类型=0、非隔行、边长上限，
  再用 Pillow 解码并核对尺寸，避免把 16 位或调色板图当作灰度图。
* **对齐打分**：不依赖 SciPy。对每个平移取两图重叠区域的 `int16` 绝对差，
  `int64` 求和后除以像素数；窗口最多 81 个位置、最多 10 个图对。
* **对齐求解**：见上，`MAX_LAYOUTS` / `MAX_NODES_EXPANDED` 限制最坏工作量；超限时返回
  `ambiguous` 且 `search_truncated=true`，把所有非基准图列为不确定——宁可说不确定，也不谎报唯一。

## 目录

```
app/
  main.py              FastAPI 路由、spec 解析与错误映射
  models.py            Pydantic 请求/响应模型与上限常量
  imaging.py           严格 PNG 解码、PNG 编码、重叠矩形
  topology.py          行程编码 + 并查集的连通域标记（黑八邻接/白四邻接）
  thumbnail.py         分块、阈值渲染与区域一一对应判定（含倍数/阈值选择）
  thumbnail_service.py 缩略图接口的编排与响应组装
  alignment.py         单图对搜索窗打分
  solver.py            全局布局枚举与唯一性判定
  service.py           对齐编排：连通性 → 打分 → 求解
tests/                 53 个测试（缩略图端到端 + 独立参考实现 + 求解器 + HTTP）
scripts/               make_thumbnail_demo.py / thumbnail_client.py（缩略图演示）
                       make_demo.py / demo_client.py / demo_data.py（对齐演示）
Dockerfile             python:3.11-slim、非 root、HEALTHCHECK
docker-compose.yml     端口 8000、只读根文件系统、健康检查
```

## 已知限制

* 缩略图只接受严格 0/255 的输入：扫描得到的灰度图需先二值化（本服务不做自适应阈值）。
* 缩略图的合格条件是“区域一一对应”，不保证细线在一个块内被截断时的形状细节；
  但任何被保留的区域都至少有一个原像素落在其块内，且区域数量与嵌套关系一致。
* 对齐只支持整数像素平移，不处理旋转、比例差或亚像素错位。
* 对齐的“达标”只看重叠区灰度 MAD；若重叠区是空白或高度重复纹理，可能出现多解，
  此时按设计返回 `ambiguous` 而不是猜测。
