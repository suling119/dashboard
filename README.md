# dashboard

一个可直接运行的 Dashboard 初版模板（纯前端，无后端依赖）。

## 快速开始

直接在浏览器打开 `index.html` 即可预览。

如果你想用本地服务打开（推荐）：

```bash
python3 -m http.server 8080
```

然后访问：`http://localhost:8080`

## 文件说明

- `index.html`：页面结构（侧边栏、指标卡、图表、表格）
- `styles.css`：样式与响应式布局
- `app.js`：模拟数据、刷新逻辑、动态渲染

## 如何改成你的业务数据

在 `app.js` 里修改：

1. `metrics`：顶部 4 个指标的候选数据（刷新按钮会随机切换）
2. `orders`：订单列表
3. `renderMetrics()` / `renderOrders()`：渲染逻辑

如果接后端接口，可以在 `setupRefreshAction()` 里把模拟数据切换替换成 `fetch('/api/xxx')`。
