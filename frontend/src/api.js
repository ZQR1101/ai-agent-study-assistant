const TOKEN_KEY = "rulebook_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function apiBase() {
  if (import.meta.env.DEV && window.location.port === "5500") {
    return "http://127.0.0.1:8000";
  }
  return "";
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : detail?.message || "请求失败");
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, { method = "GET", body, formData } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(`${apiBase()}${path}`, {
    method,
    headers,
    body: formData ? formData : body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (response.status === 401 && !path.startsWith("/auth/login")) {
    clearToken();
    window.location.hash = "#/login";
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    if (!response.ok) throw new ApiError(response.status, "请求失败");
    return response.blob();
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(response.status, data.detail ?? data);
  return data;
}

export const api = {
  login: (username, password) =>
    request("/auth/login", { method: "POST", body: { username, password } }),
  me: () => request("/auth/me"),
  playbooks: () => request("/documents/playbooks"),
  documents: () => request("/documents"),
  document: (id) => request(`/documents/${id}`),
  upload: (file, playbookId) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("playbook_id", playbookId);
    return request("/documents/upload", { method: "POST", formData });
  },
  process: (id) => request(`/documents/${id}/process`, { method: "POST", body: {} }),
  decide: (docId, verdictId, body) =>
    request(`/documents/${docId}/verdicts/${verdictId}`, { method: "PATCH", body }),
  finalize: (docId) => request(`/documents/${docId}/finalize`, { method: "POST" }),
  export: async (docId, format) => {
    const blob = await request(`/documents/${docId}/export?format=${format}`);
    return blob;
  },
  queue: () => request("/documents/review/queue"),
  rules: (playbookId) => request(`/rules?playbook_id=${encodeURIComponent(playbookId)}`),
  createRule: (body) => request("/rules", { method: "POST", body }),
  updateRule: (id, body) => request(`/rules/${id}`, { method: "PATCH", body }),
  auditRecent: () => request("/documents/audit/recent"),
  documentAudit: (id) => request(`/documents/${id}/audit`),
};

export const STATUS_LABELS = {
  uploaded: { text: "待处理", tone: "muted" },
  parsing: { text: "解析中", tone: "muted" },
  scoring: { text: "评分中", tone: "muted" },
  awaiting_review: { text: "待签字", tone: "amber" },
  finalized: { text: "已定稿", tone: "green" },
  failed: { text: "失败", tone: "red" },
};

export const STATE_LABELS = {
  drafted: { text: "初判", tone: "muted" },
  awaiting_review: { text: "待签字", tone: "amber" },
  approved: { text: "已批准", tone: "green" },
  edited: { text: "已改判", tone: "blue" },
  rejected: { text: "已驳回", tone: "red" },
};

export const RATING_LABELS = { red: "红", amber: "黄", green: "绿" };

export function formatTime(iso, withDate = true) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  const time = `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return withDate ? time : time;
}
