# 容器平台常见故障排查

## Pod OOM（内存超限被杀）

### 现象
应用实例（Pod）偶发重启，`kubectl get pod` 看到 `RESTARTS` 计数在涨；`kubectl describe pod <pod>` 里 `Last State` 为 `Terminated`、`Reason: OOMKilled`、`Exit Code: 137`。

### 排查步骤
1. 看重启原因：`kubectl describe pod <pod>`，重点看 `Last State / Reason`。`OOMKilled` 即被内核因超过 memory limit 杀掉。
2. 看重启历史与频率：`kubectl get pod <pod> -o jsonpath='{.status.containerStatuses[*].restartCount}'`，结合 `--watch` 观察是否周期性。
3. 看内存用量曲线：监控面板查该 Pod 的 `container_memory_working_set_bytes` 是否周期性逼近 limit 后跌零（被杀重启的典型锯齿）。
4. 看 limit 配置：`kubectl get pod <pod> -o jsonpath='{.spec.containers[*].resources.limits.memory}'`。

### 常见原因与处置
- 应用内存泄漏或大请求峰值导致工作集超过 `limits.memory`，被 OOMKilled。
- limit 设得过低，正常峰值就会触顶。
- 处置：先确认是泄漏还是 limit 偏低；调大 limit 属于变更，**走平台变更工单由管理员执行**，不要自行 `kubectl edit`。

> ⚠️ 关联影响：Pod 被 OOMKilled 重启期间，该实例**短暂不可用**——上游网关的健康检查会把它判为 unhealthy 并摘流，恢复后再加回。所以"容器侧偶发 OOM 重启"会在"网关侧表现为该上游实例偶发不健康 / 接口偶发 5xx"。排查接口偶发失败时，这两层要**结合起来看**：网关层定位到抖动的实例 IP，容器层确认该实例是否在对应时间点 OOM 重启。

## 就绪/存活探针失败导致摘流

`readinessProbe` 失败 → Pod 被从 Service Endpoints 摘除（不再接流）；`livenessProbe` 失败 → 容器被重启。

- 查探针配置与最近失败事件：`kubectl describe pod <pod>`，看 `Events` 里的 `Unhealthy` 记录。
- 探针超时/阈值过紧时，应用 GC 停顿或瞬时高负载也会触发误摘，表现为接口偶发失败。
