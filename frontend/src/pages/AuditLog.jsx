import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { Chip, ErrorBanner, Icon, MonoId, Spinner } from "../ui.jsx";

const EVENT_TONES = {
  "document.created": "muted",
  "document.finalized": "green",
  "document.duplicate_rejected": "amber",
  "review.started": "muted",
  "review.parsed": "muted",
  "review.scored": "muted",
  "review.scorecard_ready": "green",
  "review.failed": "red",
  "verdict.approve": "green",
  "verdict.edit": "blue",
  "verdict.reject": "red",
  "document.qa": "muted",
  "user.login": "muted",
  "rule.created": "muted",
  "rule.updated": "muted",
};

function formatFullTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export default function AuditLogPage() {
  const [mode, setMode] = useState("global");
  const [entries, setEntries] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [docQuery, setDocQuery] = useState("");
  const [docTitle, setDocTitle] = useState("");
  const [error, setError] = useState("");

  const loadGlobal = useCallback(async () => {
    try {
      const data = await api.auditRecent();
      setEntries(data.audit);
      setError("");
    } catch (err) {
      setError(err.message || "加载失败");
    }
  }, []);

  useEffect(() => {
    if (mode === "global") loadGlobal();
  }, [mode, loadGlobal]);

  const loadTimeline = async () => {
    const query = docQuery.trim();
    if (!query) return;
    setTimeline(null);
    try {
      const [audit, detail] = await Promise.all([api.documentAudit(query), api.document(query)]);
      setTimeline(audit.audit.slice().reverse());
      setDocTitle(detail.document.title);
      setError("");
    } catch (err) {
      setError(err.message || "未找到该文档");
      setTimeline([]);
    }
  };

  if (!entries && mode === "global" && !error) return <Spinner />;

  return (
    <>
      <div className="flex items-center justify-between">
        <h1 className="text-[20px] font-semibold text-ink">审计日志</h1>
        <div className="flex items-center gap-2">
          <span className="text-[12px] text-ink-3">视图模式：</span>
          <div className="flex rounded-md border border-line bg-surface p-0.5">
            {[
              ["global", "全局"],
              ["document", "单文档"],
            ].map(([key, label]) => (
              <button
                key={key}
                onClick={() => setMode(key)}
                className={`rounded px-3 py-1 text-[13px] font-medium transition-colors ${
                  mode === key ? "bg-accent-tint font-semibold text-accent" : "text-ink-2 hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {mode === "global" ? (
        <div className="rounded-lg border border-line bg-surface shadow-card">
          <div className="flex items-center justify-between border-b border-line-subtle px-4 py-2.5">
            <span className="text-[14px] font-semibold text-ink">系统操作流水</span>
            <span className="text-[12px] text-ink-3">append-only · 共 {entries.length} 条即时记录</span>
          </div>
          <table className="w-full">
            <thead>
              <tr className="border-b border-line-strong bg-canvas text-left text-[12px] font-medium uppercase tracking-wider text-ink-2">
                <th className="h-9 px-4 font-medium">时间</th>
                <th className="h-9 px-4 font-medium">关联编号</th>
                <th className="h-9 px-4 font-medium">事件标识</th>
                <th className="h-9 px-4 font-medium">操作人 / 执行主体</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id} className="border-b border-line-subtle transition-colors last:border-0 hover:bg-canvas">
                  <td className="tnum whitespace-nowrap px-4 py-2.5 text-[13px] text-ink-2">
                    {formatFullTime(entry.created_at)}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    {entry.correlation_id.startsWith("auth") || entry.correlation_id === "documents" || entry.correlation_id === "rulebook" ? (
                      <Chip tone="muted">{entry.correlation_id}</Chip>
                    ) : (
                      <a href={`#/documents/${entry.correlation_id}`}>
                        <MonoId>{entry.correlation_id}</MonoId>
                      </a>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    <Chip tone={EVENT_TONES[entry.event] || "muted"}>
                      <span className="font-mono">{entry.event}</span>
                    </Chip>
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-[13px] text-ink-2">{entry.actor}</td>
                </tr>
              ))}
              {!entries.length && (
                <tr>
                  <td colSpan={4} className="py-14 text-center text-[13px] text-ink-3">
                    暂无审计记录
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="space-y-5">
          <div className="flex items-center gap-2">
            <input
              value={docQuery}
              onChange={(e) => setDocQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && loadTimeline()}
              placeholder="输入文档编号，如 CG-0007"
              className="h-9 w-[280px] rounded-md border border-line-strong bg-surface px-3 font-mono text-[13px] outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/10"
            />
            <button
              onClick={loadTimeline}
              className="flex h-9 items-center gap-1.5 whitespace-nowrap rounded-md border border-line bg-surface px-3 text-[13px] font-medium text-ink-2 transition-colors hover:bg-canvas hover:text-ink"
            >
              <Icon name="search" className="text-[16px]" />
              查询时间线
            </button>
          </div>

          {timeline && (
            <div className="rounded-lg border border-line bg-surface p-5 shadow-card">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[15px] font-semibold text-ink">单文档时间线</span>
                  <MonoId>{timeline[0]?.correlation_id || docQuery}</MonoId>
                  {docTitle && <span className="text-[13px] text-ink-2">{docTitle}</span>}
                </div>
                <Chip tone="muted">共 {timeline.length} 条事件</Chip>
              </div>

              <div className="mt-4 space-y-0">
                {timeline.map((entry, index) => {
                  const last = index === timeline.length - 1;
                  return (
                    <div key={entry.id} className="relative flex gap-3 pb-5 last:pb-0">
                      {!last && <div className="absolute left-[7px] top-4 h-full w-px bg-line" />}
                      <span
                        className={`relative mt-1 h-[15px] w-[15px] shrink-0 rounded-full border-[3px] ${
                          entry.event === "document.finalized"
                            ? "border-green-line bg-green-bg"
                            : entry.event.startsWith("verdict")
                              ? "border-accent/30 bg-accent-tint"
                              : "border-line-strong bg-surface"
                        }`}
                      />
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-[13px] font-medium text-ink">{entry.event}</span>
                          <span className="tnum text-[12px] text-ink-3">{formatFullTime(entry.created_at)}</span>
                          <Chip tone="muted">{entry.actor}</Chip>
                        </div>
                        {entry.payload && Object.keys(entry.payload).length > 0 && (
                          <pre className="mt-1.5 overflow-x-auto rounded border border-line-subtle bg-canvas px-2.5 py-1.5 font-mono text-[11px] leading-4 text-ink-2">
                            {JSON.stringify(entry.payload, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  );
                })}
                {!timeline.length && (
                  <div className="py-8 text-center text-[13px] text-ink-3">未找到该文档的审计记录</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
