import React, { useState } from "react";
import { api, setToken } from "../api.js";
import { ErrorBanner, Icon, PrimaryButton } from "../ui.jsx";

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await api.login(username.trim(), password);
      setToken(data.token);
      onLogin(data.user);
    } catch (err) {
      setError(err.status === 401 ? "用户名或密码错误" : err.message || "登录失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-canvas p-6">
      <div className="w-full max-w-[400px] rounded-lg border border-line bg-surface p-8 shadow-card">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded bg-accent text-white">
            <Icon name="menu_book" className="text-[20px]" />
          </div>
          <span className="text-[20px] font-bold tracking-tight text-ink">Rulebook</span>
        </div>
        <p className="mt-2 text-[13px] text-ink-2">规则手册驱动的文档审查平台</p>

        {error && (
          <div className="mt-5 flex items-center gap-2 rounded border border-red-line bg-red-bg px-3 py-2.5 text-[13px] font-medium text-red-text">
            <Icon name="error" className="text-[16px]" />
            {error}
          </div>
        )}

        <form className="mt-5" onSubmit={submit}>
          <label className="block text-[13px] font-medium text-ink" htmlFor="username">
            用户名
          </label>
          <input
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            className="mt-1.5 h-9 w-full rounded-md border border-line-strong bg-surface px-3 text-[14px] text-ink outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/10"
          />
          <label className="mt-4 block text-[13px] font-medium text-ink" htmlFor="password">
            密码
          </label>
          <div className="relative mt-1.5">
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              className="h-9 w-full rounded-md border border-line-strong bg-surface px-3 pr-9 text-[14px] text-ink outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/10"
            />
            <button
              type="button"
              onClick={() => setShowPassword((v) => !v)}
              className="absolute right-2 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center text-ink-3 hover:text-ink"
            >
              <Icon name={showPassword ? "visibility_off" : "visibility"} className="text-[18px]" />
            </button>
          </div>
          <PrimaryButton type="submit" disabled={loading || !username || !password} className="mt-6 h-9 w-full justify-center">
            {loading ? "登录中…" : "登录"}
          </PrimaryButton>
        </form>

        <div className="mt-6 border-t border-line-subtle pt-4 text-[12px] leading-5 text-ink-3">
          首次启动自动创建管理员账号 <code className="whitespace-nowrap font-mono text-ink-2">admin</code>
          ，初始密码见服务器{" "}
          <code className="whitespace-nowrap font-mono text-ink-2">
            data/bootstrap_admin_password.txt
          </code>
        </div>
      </div>
    </div>
  );
}
