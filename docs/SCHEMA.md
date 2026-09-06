# e3.routing_record.v1

每行只表示一个 routed layer 在一个 step 的记录。

| 字段 | 含义 |
| --- | --- |
| schema_version | 固定为 `e3.routing_record.v1` |
| run_id / timestamp_utc | 运行身份与 UTC 时间 |
| step / mode | 训练或评估位置 |
| family | `moe`、`mot`、`latent` |
| layer_name / module_type | 模块身份 |
| num_experts / top_k | 路由结构 |
| expert_usage | 长度为 num_experts、和为 1 的主可视化向量 |
| usage_semantics | MoE 为 `topk_selection_share`；MoT/Latent 为 `mean_mixture_probability` |
| mean_router_probs | 发布时记录完整专家概率，否则 null |
| mean_topk_weight | 发布时记录 top-k 内权重，否则 null |
| normalized_entropy / normalized_gini | 归一化平衡指标，范围 0–1 |
| aux_loss | `status`、`observed` 与配置系数；缺失不得填 0 |
| source | 原始证据来自最终 top-k 输出或原生 snapshot |
| metadata | usage scope、分布式全局可用性和 dispatch policy |

所有 sink 在写入前重新校验记录，并禁止 NaN/Infinity JSON。

