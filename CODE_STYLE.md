# 我的代码风格（seg2）

> **用途**：给 AI 生成或修改本仓库代码时的风格基准；本文件优先于模型的默认习惯。
> **样本来源**：`ingest/utils/annotation2mask.py`、`ingest/her2st/get_mask.py` 中我手写/亲手重构的核心逻辑
> （2026-09-18 版本，Python 3.11，无 pyproject/ruff 配置文件；`ruff check` 为 ruff 0.16.5，其默认规则集已含 I001
> import 排序，现有 4 个文件各报一条 I001，现状未修，见第 10 节）。
> **不适用**：QC / 可视化代码（`qc_visualize` 及图例、overlay、缩略图、spot 圆点绘制等）。这类代码我通常直接让 AI
> 生成，不代表我的风格，也不要从中反推规则（例如其中的双引号、行内推导式、可调 alpha 参数都不算我的习惯）。
> 本文件会持续更新；新增规则直接追加，与旧条文冲突时以新条文为准。

## 0. 一句话总结

过程式 + 纯函数：函数只做"取数据 → 返回数据"，IO 单独一层；类型、dtype、形状全部显式；能用 numpy 向量化就不写循环；
重构只改结构，不改数据。

## 1. 总体取向

1. **函数式，不 OOP**：一个文件可以只有函数和 `TypedDict`，不引入 class / dataclass / 继承 / 抽象基类。
2. **显式优于简短**：参数类型、返回类型、dtype、依赖关系都写全；不依赖隐式转换或全局状态。
3. **最小抽象**：不预先造通用工具、配置类、基类；同一段逻辑出现第二次再提取。扁平优于分层。
4. **数据等价高于代码简洁**（本仓库是结构重构）：产物必须与旧实现逐像素一致。非显然的顺序 / 覆盖语义
   （如"区域按面积降序绘制、嵌套小区域覆盖外层大区域、描边线最后画"）必须原样保留，不许为了短而调整；
   这类地方补一行注释说明"为什么必须这样"。
5. **fail-fast**：外部输入（文件、表格、配置）在入口处断言，不在深处兜底；不写 try/except 吞异常。但配置 / 结构校验
   只留"能挡住静默错误"的那一条——例：`load_dataset_config` 只留 `names` 必须是非空 list（写成标量会被逐字符遍历，
   下游很难定位），其余交给 Python 自带的 `KeyError` / `TypeError` 报（2026-09-18 由 4 个 `raise` 精简到 1 个 `assert`）。
6. 进度输出用 `print`（带样本名），不引入 `logging`。
7. **只写下游真的会读的产物**：不为"以后可能有用"产出汇总表 / manifest / 诊断计数（`prepare_target_mpp` 就不写
   `manifest.csv`，`process_sample` 不返回记录字典，`transform_spatial` 也不返回那几个 `n_selected_*` 计数）；
   配套的统计代码一并删掉，等真有消费者再加。
   **例外**：单个产物的自描述文件（`extract_he_feature_map` 的 `metadata.json`）保留——它不是汇总统计，是给另一个程序
   读的契约（seg 的 `evaluate/data.py::load_slides` 用它的 `model` / `input_shape_hw` / `target_mpp` / `tile_size` /
   `patch_grid_hw` / `patch_mpp_yx` 校验特征与配置是否匹配，缺了会直接报错）。这类文件**字段名不能改**，改了要同步改读取端
   （2026-09-20 定）。

## 2. 文件与模块组织

- 包按"阶段 / 职责"分层：`ingest/<dataset>/<步骤>.py`（原始数据 → target-MPP samples）、`derive/<步骤>.py`（下游派生产物；
  数据集由 `derive/dataset.yaml` 参数化，所以不带 `<dataset>` 子目录）。**只给某一阶段用**的共享算法放该阶段的 `utils/`
  （现有 `ingest/utils/`）；**跨阶段共享**（`derive/` 与将来的 `segmodel/` 都要用，如 SNIC 封装、随机游走内核、吸附/阈值）放顶层
  `tools/`（`tools/snic_features.py`、`tools/propagate_random_walker.py`、`tools/pseudo_label.py`）。顶层
  `pretrained_model/` 是病理基础模型的封装（`common.py` + 8 个 `*Extractor`）：`common.py` 与 8 个 wrapper 是自己的代码，
  `<Model>/` 里是第三方权重和 HF 的 modeling 文件，与 `SNIC/` 同级对待——vendored，不按本文件风格要求，ruff 也报它。
- 脚本名用 `get_<产物>` / `prepare_<产物>` / `extract_<产物>`，与 seg 侧同名（方便两边对照），短、够用即可：
  `get_mask.py`（标注 → mask）、`get_mpp.py`（spot 坐标 → 每张切片的 mpp）、`prepare_target_mpp.py`（对齐到 target mpp
  的 samples）、  `prepare_label_mpp.py`（samples → label mpp 网格）、`get_superpixel.py`、`get_pseudo_label.py`、
  `extract_he_feature_map.py`（samples → patch / CLS 特征图）、`get_lattice_plan.py` …。**不要**改成 `build_*` / `estimate_*` 之类的"更好英语"——`get_` 是本仓库的主流动词，
  产物名词已经消歧（2026-09-18 定稿）。
- 分辨率概念成对使用：每张切片自己的 mpp（`mpp.csv` 的 `axis_mpp_x/y`，source）↔ 全流程对齐到的 `target_mpp`；同族还有
  `label_mpp`(7) / `patch_map_mpp`(14) / `export_mpp`(1)。`target_mpp` 这个名字**不要换**（它是 config 键，下游 6 个
  待搬脚本都在用）。
- 已定稿的一次改名（2026-09-18）：`label_processing_mpp` → **`label_mpp`**，值 4.0 → **7.0**，输出目录
  `1mpp/label_processing/` → **`1mpp/label_mpp/`**，脚本 `prepare_label_processing.py` → **`prepare_label_mpp.py`**。
  搬 seg 的 `process/*`（`get_superpixel` / `get_pseudo_label` 等）时按新名改路径与配置键。
- 看图 / QC 代码单独放该阶段的 `qc_visualize.py`（`ingest/her2st/qc_visualize.py`、`derive/qc_visualize.py`），
  数据链路脚本里只留一次调用——避免大段绘制逻辑混在核心逻辑里。函数名用 `qc_visualize_<对象>`：
  `qc_visualize_mask`（标注 mask）、`qc_visualize_target_mpp`（samples 三张图）、`qc_visualize_superpixels`（SNIC 边界图）、
  `qc_visualize_pseudo_label`（伪标签三张图 + 边界图）、`qc_visualize_feature_map`（每张特征图的 PCA-RGB / L2 / K-means
  SVG+PNG）。降维/聚类只在 `qc_visualize_feature_map` 里做，PCA、K-means 的随机种子固定（`random_seed=0`），产物才可复现。
- 用绝对导入按包路径引用（`from ingest.utils.annotation2mask import ClassInfo, get_sem_mask`），
  一律从仓库根运行，不用 `sys.path.insert` 兜底。
- `__init__.py` 保持 0 字节，只用于标记包；不做 re-export、不做初始化副作用。
- 脚本文件的固定骨架（自上而下；`process()` 只在单样本工作量够大时才拆，薄胶水写进 `main()` 循环，见第 5 节）：

  ```python
  from pathlib import Path
  from typing import cast

  import cv2
  import numpy as np

  from ingest.utils.annotation2mask import ClassInfo, get_sem_mask

  def save_results(...) -> Path: ...      # 落盘：写文件、返回写出的路径

  def process(...) -> None: ...           # 单样本编排：找输入 → 读 → 调核心函数 → 落盘

  def main() -> None:                     # 只放配置 + 循环
      ...

  if __name__ == '__main__':
      main()
  ```

- `main()` 里放全部硬编码配置：项目根、数据根、输出根、样本名列表、类别表，然后是 `for name in names:` 循环。
  路径依赖用参数显式传入，不用全局变量。
- 核心算法函数不碰 IO（只收数组、返回数组）；`process()` 负责编排，`save_results()` / `_write_png()` 负责落盘。
- 模块内顺序：主流程 / 公开函数在前，仅供本文件使用的私有工具在后（`_get_lines` → `_get_regions`）；
  同一份逻辑的读、算、写三步各自成函数，长度都在一屏内。
- 路径统一 `Path` + `/` 拼接；项目根用 `Path(__file__).resolve().parents[N]` 推导。

## 3. 命名

- 函数 / 变量 `snake_case`，类型 `PascalCase`；模块内私有函数和私有类型都加 `_` 前缀（`_get_lines`、`_RegionInfo`）。
- 函数名是动词短语且语义写足：`get_sem_mask`、`save_results`、`_binary_mask`、`_write_png`；
  副作用直说动作（`save_` / `write_` / `draw_`），纯计算用 `get_` 或名词短语。
- **数组变量用后缀标明"这是什么 / 为什么小心"**：
  | 后缀 | 含义 | 例 |
  |---|---|---|
  | `_mat` | `cv2.imread` 直接返回、可能为 `None` 的原始矩阵；assert 后用 `np.asarray(..., dtype=np.uint8)` 转正 | `ann_mat`、`overlay_mat` |
  | `_ui8` | uint8 图像 / 掩膜 | `line_mask_ui8` |
  | `_bl` | 布尔数组 | `line_bl` |
  | `_mask` | 掩膜 | `semantic_mask`、`kernel` 除外 |
  | `roi` | 大数组上的局部视图 | `roi = semantic_mask[region['bbox']]` |
- 循环变量固定用法：`info` 遍历类别 / TypedDict，`region` 遍历区域，`component_id` 遍历连通域，`item` 用于排序 key，
  `name` 表示切片 / 样本名。
- 不写全大写常量；可调数值做成私有函数的带默认值参数（`kernel_size: int = 11`、`area_threshold: int = 800`），
  需要时调用点显式覆盖。真正固定的编码值（0/1/255、类别 id）直接写字面量并在注释里说明。

## 4. 类型标注

- 每个函数都写参数和返回值类型，包括 `-> None`；不引入 pydantic 或运行期校验。
- 数组一律用 `numpy.typing.NDArray`：`NDArray[np.uint8]`、`NDArray[np.bool_]`；不写裸 `np.ndarray`。
- "就是一组数据"的结构用 `TypedDict`（`ClassInfo`、`_RegionInfo`），不用 dataclass / class。
- 类型不能自明的局部变量要标注：

  ```python
  lines: dict[int, NDArray[np.uint8]] = {}
  regions: list[_RegionInfo] = []
  semantic_mask: NDArray[np.uint8] = np.zeros_like(next(iter(lines.values())), dtype=np.uint8)
  ```

- 边界处（pandas 行、yaml / json）用 `cast(...)` 一次性收窄，不写 `# type: ignore`。
- 复合类型直接拼内置泛型：`list[ClassInfo]`、`dict[int, NDArray[np.uint8]]`、`tuple[int, int, int]`；
  用 `slice` 组当 bbox 是刻意选择（可直接作索引）：`bbox: tuple[slice, slice]` → `semantic_mask[region['bbox']]`。

## 5. 数据流习惯

- 长流程拆成"返回中间数据的小纯函数 + 一个组装函数"：`_get_lines()` → `_get_regions()` → `get_sem_mask()`。
- 拆分的门槛是"有实际计算量"或"被多处复用"，不是"看起来像一个步骤"：只被调用一次、内容只是读文件 + 改列名 + 转发的
  薄包装不单独成函数，直接写进 `main()` 的循环体（`get_mpp.py` 的取数就是这种写法）。
- 形状从已有数组派生，不手工解包再拼：
  `np.zeros_like(next(iter(lines.values())), dtype=np.uint8)`、`np.full(semantic_mask.shape, 255, dtype=np.uint8)`。
- 取第一个 / 唯一元素用 `next(iter(...))`，不用 `list(...)[0]`。
- 中间结果通过返回值 / 字典传递，不挂在 self、不存全局。
- 落盘函数返回写出的 `Path`（调用方不用也无所谓）。
- 断言消息短、英文、带样本名或对象名：`assert ann_mat is not None`、
  `assert not rows[['pixel_x_he', 'pixel_y_he']].isna().any().any(), f'{name}: some spots cannot be mapped'`。

## 6. 数值 / 图像处理写法

- `cv2.imread` 后立刻 `assert xxx_mat is not None`，再 `np.asarray(xxx_mat, dtype=np.uint8)` 转成有类型的数组。
- dtype 永远显式：`np.zeros_like(..., dtype=np.uint8)`、`np.full(..., 255, dtype=np.uint8)`、`interior.astype(np.uint8)`。
- 用布尔掩膜向量化赋值，不写逐像素循环：
  `roi[region['mask']] = region['class_id']`、`training[np.isin(semantic_mask, [3, 4, 5, 6])] = 0`。
- 二值掩膜统一 `np.where(cond, 255, 0).astype(np.uint8)`（或 `np.isin` 版 `_binary_mask`）。
- 大数组先切视图再局部写入，不做整体拷贝：`roi = semantic_mask[region['bbox']]`。
- cv2 的具名枚举优先，不写魔数：`cv2.CC_STAT_AREA`、`cv2.MORPH_CLOSE`、`cv2.INTER_NEAREST`、`cv2.IMWRITE_PNG_COMPRESSION`、
  `connectivity=8` 这类参数显式写。
- 排序 / 筛选用内联 lambda 表达意图：`sorted(regions, key=lambda item: item['area'], reverse=True)`。
- 写图统一用**本文件内的** `_write_image(path, image, parameters)`：`cv2.imwrite(str(path), image, parameters)`，失败抛
  `RuntimeError`；`parameters` 必填、传 cv2 参数 list（`[cv2.IMWRITE_PNG_COMPRESSION, 9]` / `[cv2.IMWRITE_JPEG_QUALITY, 92]`）。
  目录不归它管——输出目录由拥有该布局的步骤显式 `mkdir(parents=True, exist_ok=True)`（例：`prepare_target_mpp` 里
  `spatial_dir` / `masks_dir` 一起建），写图函数不顺手造目录树（2026-09-18 定）。各文件各留一份（现有 8 份：`ingest/her2st/`
  4 + `derive/` 3 + `baselines/nnunet_v2/prepare/` 1），名字 / 形参个数 / 实现保持同一形状，**但不抽公共模块**——宁可各留一份，
  也不要 `ingest/` 与 `derive/` 之间的横向依赖。
- `_write_image` 的 `image` 形参 dtype 按**该文件实际写出的格式**标注：默认 `NDArray[np.uint8]`；确实要写 16-bit 时放宽成
  `NDArray[np.uint8] | NDArray[np.uint16]`（例：`derive/get_pseudo_label.py` 的 `cancer_probability.png`，消费端按
  `/65535` 还原）。**不要**用 `cast` 骗过类型检查器，也不要为此把 float/任意 dtype 写进签名（2026-09-19 定）。
- spot 吸附有两套**不等价**的算法，别互换：`tools/pseudo_label.py::snap_spots_to_tissue` 是"方框内 argmin 平方距离 +
  整数像素阈值 + 行主序并列"（离线语义，保字节）；动态标签更新用的是 EDT 全局最近（浮点阈值、会丢弃画外点）。
  两者在边界与并列处会改变种子投票 → 概率图 → 伪标签。

## 7. 注释

- 中文注释，标点用 ASCII（`,` 而不是 `，`）；只在逻辑不自明处写，不写函数 docstring（签名已说明一切）。
- 两种形式：
  - `# 主题: 说明` —— 给一段代码起标题：`# overlay: H&E 底图 + 按类别染色`、`# thumbnail`；
  - 行尾短注 —— `# BGR`、`# 行内垂直居中`。
- 取值 / 编码约定用一行英文注释写清：`# 0: known non-tumor, 1: tumor, 255: unlabeled/ignore.`
- 领域常量字符串用英文（`'invasive cancer'`、`'undetermined spot'`），不写中文；类别表以数据形式列在 `main()` 里。
- 注释解释"为什么"，不复述代码。

## 8. 排版格式

- 4 空格缩进，无 tab，行尾无空白，文件末尾留一个换行。
- 字符串用单引号（`'ST-imgs'`、`'area'`、`'Raw image is not exist'`）；f-string 内嵌字典键用双引号：
  `f'{info["id"]}: {info["label"]}'`。
- 行长不硬卡 88，实际控制在一百以内；不为了对齐做无意义折行。
- 多行调用：在 `(` 后换行，参数一行一个、带尾逗号，`)` 与调用语句左对齐：

  ```python
  line_mask_ui8 = cv2.morphologyEx(
      line_mask_ui8,
      cv2.MORPH_CLOSE,
      kernel,
  )
  ```

  字典字面量同理，`})` 单独一行收尾：

  ```python
  regions.append({
      'class_id': class_id,
      'area': area,
      'bbox': (slice(y, y + h), slice(x, x + w)),
      'mask': component_map[y:y + h, x:x + w] == component_id,
  })
  ```

- 顶层定义之间空 1 行（现状个别处为 2 行，未统一，见第 10 节）。
- import 分组：stdlib / 三方 / 本项目 之间空行隔开（新文件如 `get_mask.py`）；旧文件里未排序的 import 不强行重排。
- 不做 `ruff format` 式的整体重排。`ruff check`（0.16.5）默认规则里已包含 I001（import 排序）：它要求 import 块之后留 2 个
  空行、`import x` 排在 `from x import y` 之前，与上面"顶层定义之间 1 空行"的写法冲突；现状是接受 I001 告警、不按它改
  （见第 10 节）。除 I001 外的报错要清零。

## 9. 明确不属于我的风格（不要照抄）

- `qc_visualize` 及同类 QC / 可视化代码：AI 生成，风格不作为基准。
- `ingest/her2st/qc_visualize.py`（`qc_visualize_mask` / `qc_visualize_target_mpp`）与 `derive/qc_visualize.py`
  （`qc_visualize_superpixels` / `qc_visualize_pseudo_label` / `qc_visualize_feature_map`）：AI 生成/重写的看图代码，
  不算我的风格，也不要照抄。**各阶段的 QC 全部集中在同目录的 `qc_visualize.py` 里**，数据链路脚本只留一次函数调用；
  不要为 QC 拆 `_draw_*` / `_thumbnail` 之类的 helper，三张图各写一段、允许重复几行。`qc_visualize_mask` 的产物
  `spot_qc_overlay.jpg` 当前与 seg 逐字节一致，改它要重跑回归。
- `derive/qc_visualize.py` 里的 `qc_visualize_feature_map` 是这条"不拆 helper"规则的例外：它拆出的是**算法阶段**
  （`_sample_features` / `_fit_visualization_models` / `_transform_feature_map`，各自有实际计算量与独立语义），不是绘制；
  存图仍集中在 `_save_figure`。它里面的 `KMeans(n_init=100,  # type: ignore[arg-type])` 是 sklearn 存根把 `n_init` 错标成
  `str` 的规避，保留这一处、不要扩散到别处。
- 仓库 `seg` 中经 AI 重构过的文件（如 `tools/annotation2mask.py`、`process/*.py`）大量使用双引号 + 中文 docstring +
  dataclass，那是 AI 的写法，不是我的；本仓库以本文件为准。
- `ingest/utils/estimate_mpp_from_spatial.py` 目前仍是 seg 原文件的原样搬运（双引号、`np.ndarray`、docstring、横幅注释、
  `raise ValueError` 校验），我暂不改它，因此它也不是风格样本。
- 新建文件不要留 `TODO` / `# type: ignore` / 未使用的占位参数（`main.py` 目前是空文件，属于待写状态）。

## 10. 待定稿（现状不一致，需要我拍板；AI 先不要当成规则）

- [ ] 顶层定义之间：1 行空行还是 2 行空行统一？
- [ ] `main()` 里的类别表目前是双引号（其余全仓单引号），是否统一成单引号？
- [ ] 核心函数是否需要一行中文 docstring（现状：无）？
- [ ] 行长上限是否上 hard limit（88 / 100）？
- [ ] ruff 的 I001 怎么办（全仓 4 个文件都告警，且它要求 import 后留 2 空行）：
      跑 `ruff check --fix` 统一 import 块 / 在配置里 ignore I001 保持现有写法 / 继续容忍？
- [ ] 是否要新建 `pyproject.toml` 固化这些规则（目前无配置文件，ruff 只能用默认规则集）？

## 11. 生成代码自检清单

1. 类型全不全（参数、返回、局部关键变量、`NDArray[...]`、`TypedDict`）？
2. dtype 是否每个数组都显式？
3. 有没有该向量化却写成 Python 循环 / 逐像素赋值的地方？
4. 外部输入是否 fail-fast 断言，且消息带样本名？
5. 核心函数是否保持"纯计算不做 IO"？IO 是否集中在 `save_*` / `_write_*`？
6. 命名是否用了本文件的 `_mat` / `_ui8` / `_bl` / `roi` / `info` 约定？
7. 有没有为了简短而改变原有绘制 / 覆盖顺序等非显然语义？
8. 注释是不是中文、稀疏、只在需要解释"为什么"的地方？
