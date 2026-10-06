# nifti-qc-sampler

神经影像质控平台的三维 NIfTI 体数据抽查服务。按扫描仪 **RAS 世界坐标**采样：
严格解析 NIfTI-1 头（方向矩阵、字节序、强度缩放逐项校验，绝不猜测），
将 RAS 坐标经所选仿射的逆变换映射到连续体素坐标，再做三线性插值。

## 布局

```
app/            FastAPI 服务
  main.py       POST /api/nifti/sample、GET /healthz、multipart/坐标校验
  nifti.py      NIfTI-1 严格解析（头校验、sform/qform、载荷）
  sampling.py   仿射逆变换 + 三线性插值
  errors.py     稳定错误码（请求级 ApiError / 点级 PointError）
tests/          pytest 单元与 API 测试（niftibuild.py 生成测试样本）
verify/         一次性校验服务（代码测试 + 镜像校验 + API 冒烟）
Dockerfile      运行镜像（python:3.12-slim，非 root 运行）
docker-compose.yml  app 服务 + verify 一次性服务
```

## API

### `POST /api/nifti/sample`

`multipart/form-data`，两个字段：

| 字段 | 内容 |
| --- | --- |
| `file` | 单个 NIfTI-1 `.nii` 文件，≤ 16 MiB |
| `points` | JSON 数组，1–256 项：`{"id": <整数>, "ras": [x, y, z]}`；`id` 唯一，坐标有限 |

```bash
curl -X POST http://localhost:8080/api/nifti/sample \
  -F "file=@vol.nii" \
  -F 'points=[{"id": 1, "ras": [10.0, 20.0, 30.0]},
              {"id": 2, "ras": [13.0, 23.0, 35.0]}]'
```

成功响应（`200`，结果保持请求顺序；越界/非有限数据按点返回错误，不影响其他点）：

```json
{
  "transform": "sform",
  "results": [
    {"id": 1, "status": "ok", "voxel": [0.0, 0.0, 0.0], "value": 12.5},
    {"id": 2, "status": "error",
     "error": {"code": "OUT_OF_BOUNDS",
               "message": "voxel coordinate i=... lies outside the closed voxel-center domain [0, 255] (axis length n=256)"}}
  ]
}
```

每点给出：连续体素坐标 `voxel`（逆变换后的 i/j/k 浮点坐标）、插值强度
`value`（先缩放后插值）、全文件统一的 `transform`（`sform` 或 `qform`）。

请求级/文件结构错误返回 `4xx`，`field` 定位到头部字段或表单字段：

```json
{"error": {"code": "TRAILING_BYTES",
           "message": "file has 4 trailing byte(s): header implies 1092 bytes but the file has 1096 bytes",
           "field": "file",
           "details": {"expected_bytes": 1092, "actual_bytes": 1096}}}
```

### `GET /healthz`

就绪探针，服务可处理请求时返回 `200 {"status": "ok"}`。

## 文件接受规则（严格，逐项拒绝）

- 单文件 `.nii`：`magic == "n+1\0"`；`sizeof_hdr == 348` 同时确定字节序
  （大端/小端均可）。
- 恰好三维：`dim[0] == 3` 且各轴长度 ≥ 1。
- 数据类型仅 `int16`（datatype 4 / bitpix 16）或 `float32`（16 / 32），
  `bitpix` 必须与 `datatype` 一致。
- `vox_offset` 必须为 ≥ 352 的有限整数值；文件长度必须恰好等于
  `vox_offset + nvox × (bitpix/8)`——载荷截断或**尾随字节**都拒绝。
- `scl_slope`/`scl_inter` 必须有限；`scl_slope == 0` 按 NIfTI-1 规范视为
  不缩放（1/0）。插值前先对体素值做 `raw*slope + inter`。
- 变换选择：`sform_code > 0` 时用 sform（srow 必须有限），否则
  `qform_code > 0` 时用 qform（四元数、pixdim、qoffset 必须有限，
  `qfac = sign(pixdim[0])`，0 视为 +1）；两者皆无 → 拒绝。
  所选仿射必须可逆（线性部分行列式非零有限），否则拒绝。
- 逆变换后的连续体素坐标必须落在体素中心闭域 `[0, n-1]`³ 内
  （容忍 1e-6 体素的浮点噪声）；上边界处邻域坍缩到唯一端点
  （`hi = lo = n-1`，权重 0），单切片轴同理。

## 稳定错误码

请求级（4xx JSON，`field` 定位）：

| code | 含义 | field |
| --- | --- | --- |
| `UNSUPPORTED_MEDIA_TYPE` | 非 multipart 请求 (415) | — |
| `REQUEST_TOO_LARGE` / `FILE_TOO_LARGE` | 请求体/文件超限 (413) | `file` |
| `FILE_MISSING` / `MULTIPLE_FILES` | 文件字段缺失/多于一个 | `file` |
| `POINTS_MISSING` / `POINTS_INVALID_JSON` / `POINTS_INVALID` | points 字段缺失或非法 | `points` |
| `POINTS_COUNT` | 点数不在 1–256 | `points` |
| `POINT_ID_INVALID` / `POINT_ID_DUPLICATE` | 编号非整数/重复 | `points` |
| `POINT_RAS_INVALID` / `POINT_COORD_NON_FINITE` | 坐标结构非法/非有限 | `points` |
| `HEADER_TRUNCATED` / `BAD_SIZEOF_HDR` / `BAD_MAGIC` | 头过短/魔数或字节序非法 | `file`/`sizeof_hdr`/`magic` |
| `BAD_DIM` | 非三维或轴长 < 1 | `dim` |
| `UNSUPPORTED_DATATYPE` / `BITPIX_MISMATCH` | 数据类型不支持/位宽矛盾 | `datatype`/`bitpix` |
| `BAD_VOX_OFFSET` | 体素偏移非法 | `vox_offset` |
| `PAYLOAD_TRUNCATED` / `TRAILING_BYTES` | 载荷长度与头部矛盾/尾随字节 | `file` |
| `NON_FINITE_SCALING` | 缩放参数非有限 | `scl_slope`/`scl_inter` |
| `NO_VALID_TRANSFORM` | sform/qform 皆无 | `sform_code` |
| `NON_FINITE_AFFINE` | 仿射参数含非有限值 | `srow_*`/`quatern_b`/`pixdim`/`qoffset_x` |
| `NON_INVERTIBLE_AFFINE` | 仿射不可逆 | `srow_x`/`pixdim` |

点级（200 响应内该点 `status: "error"`，`id` 定位）：

| code | 含义 |
| --- | --- |
| `OUT_OF_BOUNDS` | 逆变换后坐标超出体素中心闭域 |
| `NON_FINITE_DATA` | 八个邻近体素缩放值含 NaN/Inf |

## 运行

```bash
# 启动服务（宿主机端口默认 8080，可用 NIFTI_HOST_PORT 覆盖）
NIFTI_HOST_PORT=9000 docker compose up --build app

# 一次性校验：等待 app 健康后运行，退出码汇总三个阶段
docker compose up --build --exit-code-from verify
echo $?   # 0=全部通过；位掩码 1=代码测试 2=镜像校验 4=API 冒烟
```

`verify` 阶段：

1. **代码测试** — 完整 pytest 套件（解析、插值、API 共 76 项）。
2. **镜像构建校验** — 非 root 运行、`IMAGE_VERSION` 已注入、预期文件
   布局、Python ≥ 3.10、关键依赖可导入。
3. **API 冒烟** — 对健康的 live 服务跑 9 组检查，覆盖小端 int16+sform、
   大端 float32+qform、sform 优先级、qform 旋转+负 qfac、尾随字节、
   缺失变换、逐点非有限数据、请求校验（重复编号、>256 点）。

本地开发：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest            # 单元/API 测试
.venv/bin/uvicorn app.main:app --port 8000
IMAGE_VERSION=dev .venv/bin/python -m verify.verify   # 端到端自检
```
