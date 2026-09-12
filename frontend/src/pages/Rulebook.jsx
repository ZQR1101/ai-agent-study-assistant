import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import {
  Chip,
  ErrorBanner,
  Icon,
  Modal,
  PrimaryButton,
  SecondaryButton,
  Spinner,
} from "../ui.jsx";

const emptyForm = { name: "", dimension: "", guidance: "", weight: 1 };

function RuleModal({ playbooks, activePlaybook, rule, dimensions, onClose, onSaved }) {
  const [form, setForm] = useState(
    rule
      ? { name: rule.name, dimension: rule.dimension, guidance: rule.guidance, weight: rule.weight }
      : { ...emptyForm, dimension: dimensions[0] || "" },
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!form.name.trim() || !form.guidance.trim() || !form.dimension) {
      setError("规则名称、维度与判定标准均为必填项");
      return;
    }
    setBusy(true);
    setError("");
    try {
      if (rule) {
        await api.updateRule(rule.id, form);
      } else {
        await api.createRule({ playbook_id: activePlaybook, ...form });
      }
      onSaved();
    } catch (err) {
      setError(err.message || "保存失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={rule ? "编辑规则" : "新增规则"} onClose={onClose}>
      <label className="block text-[13px] font-medium text-ink">
        规则名称 <span className="text-red-text">必填项</span>
      </label>
      <input
        value={form.name}
        onChange={(e) => setForm({ ...form, name: e.target.value })}
        className="mt-1.5 h-9 w-full rounded-md border border-line-strong px-3 text-[14px] outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/10"
      />

      <label className="mt-4 block text-[13px] font-medium text-ink">所属业务维度</label>
      <select
        value={form.dimension}
        onChange={(e) => setForm({ ...form, dimension: e.target.value })}
        className="mt-1.5 h-9 w-full rounded-md border border-line-strong bg-surface px-3 text-[14px] outline-none focus:border-accent"
      >
        {dimensions.map((dimension) => (
          <option key={dimension} value={dimension}>
            {dimension}
          </option>
        ))}
      </select>

      <label className="mt-4 block text-[13px] font-medium text-ink">评估权重</label>
      <div className="mt-1.5 flex gap-2">
        {[1, 2].map((w) => (
          <button
            key={w}
            type="button"
            onClick={() => setForm({ ...form, weight: w })}
            className={`flex-1 rounded-md border px-3 py-2 text-[13px] transition-colors ${
              form.weight === w
                ? "border-accent bg-accent-tint font-semibold text-accent"
                : "border-line text-ink-2 hover:border-line-strong"
            }`}
          >
            {w === 1 ? "标准权重 (×1)" : "关键核心项 (×2)"}
          </button>
        ))}
      </div>

      <label className="mt-4 block text-[13px] font-medium text-ink">
        判定标准与分级指引 <span className="float-right text-[12px] font-normal text-ink-3">支持红黄绿三阶分级</span>
      </label>
      <textarea
        value={form.guidance}
        onChange={(e) => setForm({ ...form, guidance: e.target.value })}
        rows={4}
        className="mt-1.5 w-full rounded-md border border-line-strong px-3 py-2 text-[13px] outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/10"
      />

      {error && <div className="mt-3"><ErrorBanner message={error} /></div>}

      <div className="mt-5 flex items-center justify-between">
        <span className="text-[12px] text-ink-3">保存后下一份文档立即生效，无需改代码。</span>
        <div className="flex gap-2">
          <SecondaryButton onClick={onClose}>取消</SecondaryButton>
          <PrimaryButton onClick={submit} disabled={busy}>
            {busy ? "保存中…" : "保存规则"}
          </PrimaryButton>
        </div>
      </div>
    </Modal>
  );
}

export default function RulebookPage({ user }) {
  const [playbooks, setPlaybooks] = useState([]);
  const [activePlaybook, setActivePlaybook] = useState(null);
  const [rules, setRules] = useState(null);
  const [editing, setEditing] = useState(null); // rule | "new"
  const [error, setError] = useState("");

  const isAdmin = user?.role === "admin";

  const load = useCallback(async () => {
    if (!activePlaybook) return;
    try {
      const data = await api.rules(activePlaybook);
      setRules(data.rules);
      setError("");
    } catch (err) {
      setError(err.message || "加载失败");
    }
  }, [activePlaybook]);

  useEffect(() => {
    api.playbooks().then((data) => {
      setPlaybooks(data.playbooks);
      setActivePlaybook((current) => current || data.playbooks[0]?.id || null);
    });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (!rules && !error) return <Spinner />;
  const spec = playbooks.find((p) => p.id === activePlaybook);
  const dimensions = spec?.dimensions || [];
  const grouped = dimensions
    .map((dimension) => ({
      dimension,
      rules: (rules || []).filter((r) => r.dimension === dimension),
    }))
    .filter((group) => group.rules.length > 0);

  return (
    <>
      <div className="flex items-center justify-between">
        <h1 className="text-[20px] font-semibold text-ink">规则手册</h1>
        {isAdmin && (
          <PrimaryButton onClick={() => setEditing("new")}>
            <Icon name="add" className="text-[18px]" />
            新增规则
          </PrimaryButton>
        )}
      </div>

      <div className="flex gap-2">
        {playbooks.map((p) => (
          <button
            key={p.id}
            onClick={() => setActivePlaybook(p.id)}
            className={`whitespace-nowrap rounded border px-3 py-1.5 text-[13px] font-medium transition-colors ${
              p.id === activePlaybook
                ? "border-accent bg-accent text-white"
                : "border-line bg-surface text-ink-2 hover:border-line-strong"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1.5 text-[12px] text-ink-3">
        <Icon name="info" className="text-[15px]" />
        规则保存在数据库，保存后下一份文档立即生效。
        {!isAdmin && <span>（当前账号为评审专家，仅可查看）</span>}
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="space-y-6">
        {grouped.map(({ dimension, rules: groupRules }) => (
          <section key={dimension}>
            <div className="mb-2 flex items-center gap-2">
              <span className="h-[16px] w-[3px] rounded bg-accent" />
              <span className="text-[15px] font-semibold text-ink">{dimension}</span>
              <span className="text-[12px] text-ink-3">· {groupRules.length} 条</span>
            </div>
            <div className="divide-y divide-line-subtle rounded-lg border border-line bg-surface shadow-card">
              {groupRules.map((rule) => (
                <div key={rule.id} className={`flex items-center gap-4 px-4 py-3 ${rule.active ? "" : "opacity-55"}`}>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[14px] font-semibold text-ink">{rule.name}</span>
                      <span className="rounded border border-line bg-canvas px-1.5 py-0.5 text-[11px] font-medium text-ink-2">
                        ×{rule.weight}
                      </span>
                      <Chip tone={rule.active ? "green" : "muted"}>
                        {rule.active ? "已启用" : "已停用"}
                      </Chip>
                    </div>
                    <p className="mt-1 line-clamp-2 text-[12px] leading-5 text-ink-3">{rule.guidance}</p>
                  </div>
                  {isAdmin ? (
                    <div className="flex shrink-0 items-center gap-3">
                      <label className="flex cursor-pointer items-center">
                        <input
                          type="checkbox"
                          checked={rule.active}
                          onChange={(e) =>
                            api
                              .updateRule(rule.id, { active: e.target.checked })
                              .then(load)
                              .catch((err) => setError(err.message))
                          }
                          className="peer sr-only"
                        />
                        <span className="relative h-5 w-9 rounded-full bg-line-strong transition-colors peer-checked:bg-accent after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-surface after:transition-transform peer-checked:after:translate-x-4" />
                      </label>
                      <button
                        onClick={() => setEditing(rule)}
                        className="flex h-8 w-8 items-center justify-center rounded text-ink-2 transition-colors hover:bg-line-subtle hover:text-accent"
                      >
                        <Icon name="edit_note" className="text-[18px]" />
                      </button>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {editing && (
        <RuleModal
          playbooks={playbooks}
          activePlaybook={activePlaybook}
          rule={editing === "new" ? null : editing}
          dimensions={dimensions}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null);
            load();
          }}
        />
      )}
    </>
  );
}
