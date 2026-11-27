# 分片词表 Embedding 与稀疏更新归属

分片词表的 Embedding/Gather 算子，以及与优化器交接的更新记录。仅依赖 Python
标准库。

## 环境

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

## 契约

输入（见 `data/input.json` 与 `data/schema.json`）：

- `shards[].shard_id/start/values`：每个分片拥有连续区间
  `start .. start + len(values) - 1`；分片之间允许空洞。
- `ids`：微批次 token 序列，可重复，顺序即输出顺序。
- `unknown_id`：未知词哨兵，永远解析为 `unknown`。
- `disabled`：被下线的词条，解析为 `disabled`，永不接受梯度更新。

输出（前向）：按原 token 顺序排列的向量、每个位置的状态
（`ok` / `unknown` / `disabled` / `missing` / `padding`）与分片归属
（`shard_id`、`shard_start`、`local_index`）。相同 `(version, token_id)`
在一个微批次内复用首个向量，后续位置记录 `reuse_of`。

反向：按 `(version, token_id)` 聚合，聚合规则显式声明为
`sum` / `mean` / `first` / `last` / `error`，绝不按出现次数静默覆盖。
禁用词产出 `BlockedUpdate`（含原因）而不产生行更新；未知词、缺失词和 padding
产出 `SkippedToken`，可交给数据质量流程。

每条 `UpdateRecord` 携带 `version`、分片定位、`layout_fingerprint`、
`row_fingerprint` 和内容寻址的 `update_id`。

## 用法

```python
import json
from operator_embedding_shards import (
    EmbeddingShards, ParameterStore, VocabularyRegistry, vocabulary_from_payload,
)

payload = json.loads(open("data/input.json", encoding="utf-8").read())
registry = VocabularyRegistry([vocabulary_from_payload(payload)])
operator = EmbeddingShards(registry, aggregation="sum", padding_id=None)
store = ParameterStore(registry, learning_rate=0.1)

forward = operator.forward(payload["ids"], step=1, micro_batch_id="mb-1")
# forward.values         -> 原 token 顺序的向量
# forward.statuses()     -> ok / unknown / disabled / missing / padding
# forward.shard_attribution() -> 每个向量来自哪个分片

gradients = [(1.0, 0.0), (0.0, 1.0), (0.0, 2.0), None, None]
batch = operator.backward(forward, gradients)
# batch.records  -> 已按规则聚合的分片更新记录
# batch.blocked  -> 禁用词及其数据质量原因
# batch.skipped  -> 未知/缺失/padding 位置
store.apply(batch)
```

## 版本切换与恢复

- `forward(ids, versions=[...])` 支持逐位置声明词表版本。每个位置按自己的版本
  解析；后一个版本不会改动已产生的向量，`ParameterStore.apply` 也会校验
  `layout_fingerprint`，拒绝跨版本的覆盖。
- `store.replayability(batch)` 干跑（不写入）地把更新分成可重放的
  `replayable` 和已失去归属的 `lost`（布局切换、行已前进、分片消失、版本卸载）。
- `store.checkpoint()` / `ParameterStore.from_checkpoint(registry, payload)`
  保存行值与已应用的 `update_id`；恢复后 `apply` 会返回 `duplicate` /
  `stale` / `unattributed`，不会重复累加。

## 资料

`data/input.json` 是可重复使用的本地合成输入，`data/schema.json` 记录字段含义和允许的状态。
