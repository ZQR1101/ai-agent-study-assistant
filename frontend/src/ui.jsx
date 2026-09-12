import React from "react";
import { STATUS_LABELS, STATE_LABELS, RATING_LABELS } from "./api.js";

const TONES = {
  red: "bg-red-bg border-red-line text-red-text",
  amber: "bg-amber-bg border-amber-line text-amber-text",
  green: "bg-green-bg border-green-line text-green-text",
  muted: "bg-canvas border-line text-ink-2",
  blue: "bg-accent-tint border-accent/30 text-accent",
};

export function Chip({ tone = "muted", children, dot = false, className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-2 py-0.5 text-[12px] font-medium leading-[18px] ${TONES[tone]} ${className}`}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

export function StatusChip({ status }) {
  const meta = STATUS_LABELS[status] || { text: status, tone: "muted" };
  return (
    <Chip tone={meta.tone} dot>
      {meta.text}
    </Chip>
  );
}

export function StateTag({ state }) {
  const meta = STATE_LABELS[state] || { text: state, tone: "muted" };
  return <Chip tone={meta.tone}>{meta.text}</Chip>;
}

export function RatingBadge({ rating, size = "sm" }) {
  const label = RATING_LABELS[rating] || rating;
  const cls =
    size === "lg"
      ? "h-7 w-7 text-[13px] font-semibold"
      : "h-6 w-6 text-[12px] font-semibold";
  const tone = { red: "red", amber: "amber", green: "green" }[rating] || "muted";
  return (
    <span
      className={`inline-flex ${cls} shrink-0 items-center justify-center rounded border ${TONES[tone]} whitespace-nowrap`}
    >
      {label}
    </span>
  );
}

export function RatingCounts({ red = 0, amber = 0, green = 0, showZero = true }) {
  const cell = (label, value, tone) =>
    value === 0 && !showZero ? null : (
      <Chip tone={tone}>
        {label} <span className="tnum font-semibold">{value}</span>
      </Chip>
    );
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      {cell("红", red, "red")}
      {cell("黄", amber, "amber")}
      {cell("绿", green, "green")}
    </span>
  );
}

export function MonoId({ children }) {
  return (
    <span className="inline-block whitespace-nowrap rounded border border-line bg-line-subtle px-1.5 py-0.5 font-mono text-[12px] font-medium text-ink">
      {children}
    </span>
  );
}

const NAV_ITEMS = [
  { hash: "#/", icon: "grid_view", label: "工作台" },
  { hash: "#/queue", icon: "assignment_turned_in", label: "审批队列", badge: "queue" },
  { hash: "#/rulebook", icon: "gavel", label: "规则手册", adminOnly: true },
  { hash: "#/audit", icon: "manage_search", label: "审计日志" },
];

export function Icon({ name, className = "" }) {
  return <span className={`material-symbols-outlined ${className}`}>{name}</span>;
}

export function Sidebar({ active, user, queueCount, onLogout }) {
  return (
    <aside className="fixed left-0 top-0 z-50 flex h-full w-[200px] select-none flex-col justify-between border-r border-line bg-surface">
      <div>
        <div className="flex h-14 items-center gap-2 border-b border-line px-4">
          <div className="flex h-7 w-7 items-center justify-center rounded bg-accent text-white">
            <Icon name="menu_book" className="text-[18px]" />
          </div>
          <span className="text-[16px] font-semibold tracking-tight text-ink">Rulebook</span>
        </div>
        <nav className="flex w-full flex-col gap-1 p-2">
          {NAV_ITEMS.filter((item) => !item.adminOnly || user?.role === "admin").map((item) => {
            const isActive = active === item.hash;
            return (
              <a
                key={item.hash}
                href={item.hash}
                className={`flex items-center gap-2.5 rounded px-3 py-2 text-[13px] transition-colors ${
                  isActive
                    ? "bg-accent-tint font-semibold text-accent"
                    : "text-ink-2 hover:bg-line-subtle hover:text-ink"
                }`}
              >
                <Icon name={item.icon} className="text-[18px]" />
                <span>{item.label}</span>
                {item.badge === "queue" && queueCount > 0 && (
                  <span className="ml-auto flex h-[18px] min-w-[18px] items-center justify-center rounded bg-red-text px-1 text-[11px] font-semibold text-white">
                    {queueCount}
                  </span>
                )}
              </a>
            );
          })}
        </nav>
      </div>
      <div className="border-t border-line p-3">
        <div className="flex items-center gap-2.5 rounded p-1.5 transition-colors hover:bg-line-subtle">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-white">
            <Icon name="person" className="text-[18px]" />
          </div>
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center justify-between gap-1">
              <span className="truncate text-[13px] font-semibold text-ink">
                {user?.display_name || user?.username || "—"}
              </span>
              <span className="shrink-0 rounded bg-canvas px-1.5 py-0.5 text-[11px] font-semibold text-ink-2">
                {user?.role === "admin" ? "管理员" : "评审专家"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="truncate font-mono text-[11px] text-ink-3">
                {user?.username || ""}
              </span>
              <button
                onClick={onLogout}
                className="text-[11px] text-ink-3 transition-colors hover:text-red-text"
              >
                退出
              </button>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export function Topbar({
  notifications = [],
  unread = 0,
  panelOpen = false,
  onTogglePanel,
  onMarkAllRead,
  onItemClick,
}) {
  return (
    <header className="fixed left-[200px] right-0 top-0 z-40 flex h-14 items-center justify-between border-b border-line bg-canvas/80 px-6 backdrop-blur">
      <div className="flex items-center gap-2 text-[12px] font-medium text-ink-2">
        <Icon name="shield" className="text-[16px]" />
        <span>合规策略与规则引擎</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 rounded border border-line bg-surface px-2 py-1 font-mono text-[11px] text-ink-2">
          <span className="h-2 w-2 rounded-full bg-accent" />
          <span>引擎运行中</span>
        </div>
        <div className="relative">
          <button
            onClick={onTogglePanel}
            className="relative flex h-8 w-8 items-center justify-center rounded text-ink-2 transition-colors hover:bg-line-subtle hover:text-ink"
          >
            <Icon name={unread > 0 ? "notifications_active" : "notifications"} className="text-[20px]" />
            {unread > 0 && (
              <span className="absolute -right-0.5 -top-0.5 flex h-[16px] min-w-[16px] items-center justify-center rounded-full bg-red-text px-1 text-[10px] font-semibold text-white">
                {unread > 99 ? "99+" : unread}
              </span>
            )}
          </button>
          {panelOpen && (
            <div className="absolute right-0 top-10 w-[380px] rounded-lg border border-line bg-surface shadow-pop">
              <div className="flex items-center justify-between border-b border-line-subtle px-4 py-2.5">
                <span className="text-[13px] font-semibold text-ink">通知中心</span>
                {unread > 0 && (
                  <button
                    onClick={onMarkAllRead}
                    className="text-[12px] text-accent hover:underline"
                  >
                    全部已读
                  </button>
                )}
              </div>
              <div className="max-h-[380px] overflow-y-auto">
                {notifications.length === 0 && (
                  <div className="py-10 text-center text-[13px] text-ink-3">暂无通知</div>
                )}
                {notifications.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => onItemClick(item)}
                    className={`block w-full border-b border-line-subtle px-4 py-3 text-left transition-colors last:border-0 hover:bg-canvas ${
                      item.read ? "opacity-60" : ""
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {!item.read && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />}
                      <span className={`truncate text-[13px] ${item.read ? "text-ink-2" : "font-semibold text-ink"}`}>
                        {item.title}
                      </span>
                    </div>
                    {item.body && (
                      <div className="mt-0.5 line-clamp-2 text-[12px] leading-4 text-ink-3">{item.body}</div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <button className="flex h-8 w-8 items-center justify-center rounded text-ink-2 transition-colors hover:bg-line-subtle hover:text-ink">
          <Icon name="help_outline" className="text-[20px]" />
        </button>
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-white">
          <Icon name="person" className="text-[18px]" />
        </div>
      </div>
    </header>
  );
}

export function PrimaryButton({ children, className = "", ...props }) {
  return (
    <button
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-md bg-accent px-3 py-1.5 text-[13px] font-semibold text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-ink-3 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({ children, className = "", ...props }) {
  return (
    <button
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border border-line bg-surface px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-canvas hover:text-ink disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function TextAction({ children, tone = "accent", ...props }) {
  const toneCls = tone === "red" ? "text-red-text hover:bg-red-bg" : "text-accent hover:bg-accent-tint";
  return (
    <button
      className={`whitespace-nowrap rounded px-2 py-1 text-[13px] font-medium transition-colors ${toneCls} disabled:cursor-not-allowed disabled:text-ink-3 ${props.className || ""}`}
      {...props}
    />
  );
}

export function Modal({ title, onClose, children, width = "max-w-lg" }) {
  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-[rgba(33,37,41,0.4)] p-6"
      onClick={onClose}
    >
      <div
        className={`w-full ${width} rounded-lg bg-surface shadow-modal`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line-subtle px-5 py-4">
          <span className="text-[16px] font-semibold text-ink">{title}</span>
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded text-ink-3 hover:bg-line-subtle hover:text-ink"
          >
            <Icon name="close" className="text-[18px]" />
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export function Spinner({ text = "加载中…" }) {
  return (
    <div className="flex items-center justify-center py-20 text-[13px] text-ink-3">
      <Icon name="progress_activity" className="mr-2 animate-spin text-[18px]" />
      {text}
    </div>
  );
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="rounded border border-red-line bg-red-bg px-3 py-2 text-[13px] text-red-text">
      {message}
    </div>
  );
}
