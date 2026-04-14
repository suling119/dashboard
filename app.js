const metrics = [
  { sales: 86213, users: 319, orders: 142, rate: 3.7 },
  { sales: 90342, users: 347, orders: 151, rate: 3.9 },
  { sales: 97520, users: 382, orders: 166, rate: 4.2 },
];

const orders = [
  { id: "NO.20260414-001", customer: "张三", amount: 329, status: "已支付" },
  { id: "NO.20260414-002", customer: "李四", amount: 1299, status: "待发货" },
  { id: "NO.20260414-003", customer: "王五", amount: 89, status: "已完成" },
  { id: "NO.20260414-004", customer: "赵六", amount: 569, status: "退款中" },
];

function formatCurrency(value) {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(value);
}

function renderMetrics(data) {
  document.getElementById("sales").textContent = formatCurrency(data.sales);
  document.getElementById("users").textContent = String(data.users);
  document.getElementById("orders").textContent = String(data.orders);
  document.getElementById("rate").textContent = `${data.rate}%`;
}

function renderOrders() {
  const tbody = document.getElementById("orderRows");
  tbody.innerHTML = orders
    .map(
      (order) => `
      <tr>
        <td>${order.id}</td>
        <td>${order.customer}</td>
        <td>${formatCurrency(order.amount)}</td>
        <td>${order.status}</td>
      </tr>
    `
    )
    .join("");
}

function setToday() {
  const todayText = new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "full",
  }).format(new Date());
  document.getElementById("today").textContent = todayText;
}

function setupRefreshAction() {
  const button = document.getElementById("refreshBtn");
  button.addEventListener("click", () => {
    const randomIndex = Math.floor(Math.random() * metrics.length);
    renderMetrics(metrics[randomIndex]);
  });
}

function bootstrap() {
  setToday();
  renderMetrics(metrics[0]);
  renderOrders();
  setupRefreshAction();
}

bootstrap();
