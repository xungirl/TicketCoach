/* =============================================
   TicketCoach — AI coach drawer
   Login (signed trainee token) → chat with the tool-calling coach agent.
   Depends on esc(), apiFetch(), openLibraryItem(), startRoleplay() from app.js.
   ============================================= */

let coachHistory = []; // [{role: "user"|"assistant", content}]
let coachBusy = false;

function coachToken() { return localStorage.getItem("tc_trainee_token"); }

function openCoach() {
  const d = document.getElementById("coach-drawer");
  d.classList.add("open");
  d.setAttribute("aria-hidden", "false");
  coachRender();
}

function closeCoach() {
  const d = document.getElementById("coach-drawer");
  d.classList.remove("open");
  d.setAttribute("aria-hidden", "true");
}

function coachRender() {
  const loggedIn = !!coachToken();
  document.getElementById("coach-login").style.display = loggedIn ? "none" : "";
  document.getElementById("coach-main").style.display = loggedIn ? "" : "none";
  if (loggedIn) {
    document.getElementById("coach-who").textContent = "👤 " + (localStorage.getItem("tc_trainee_name") || "");
    coachRefreshProfile();
    if (!coachHistory.length && !document.getElementById("coach-messages").children.length) {
      coachAppend("assistant", "你好！我可以帮你查阿里云官方文档、分析你的对练记录、推荐下一个要练的剧本。试试下面的问题 👇");
    }
    document.getElementById("coach-input").focus();
  } else {
    const saved = localStorage.getItem("tc_access_token");
    if (saved) document.getElementById("coach-code").value = saved;
    document.getElementById("coach-name").focus();
  }
}

async function coachLogin(event) {
  event.preventDefault();
  const name = document.getElementById("coach-name").value.trim();
  const code = document.getElementById("coach-code").value;
  if (!name) return;
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, access_code: code || null }),
    });
    const data = await res.json();
    if (!res.ok) { showError(data.detail || "登录失败"); return; }
    localStorage.setItem("tc_trainee_token", data.token);
    localStorage.setItem("tc_trainee_name", data.name);
    if (code) localStorage.setItem("tc_access_token", code); // same code unlocks the other APIs
    coachRender();
  } catch (e) {
    showError(`登录出错：${e.message}`);
  }
}

function coachLogout() {
  localStorage.removeItem("tc_trainee_token");
  localStorage.removeItem("tc_trainee_name");
  coachHistory = [];
  document.getElementById("coach-messages").innerHTML = "";
  coachRender();
}

async function coachForget() {
  if (!confirm("确定清除教练记住的你的所有对练记录和笔记吗？此操作不可恢复。")) return;
  const res = await apiFetch("/api/me", { method: "DELETE" });
  if (res.ok) {
    coachHistory = [];
    document.getElementById("coach-messages").innerHTML = "";
    coachAppend("assistant", "已清除。我们从头开始！");
    coachRefreshProfile();
  }
}

async function coachRefreshProfile() {
  const box = document.getElementById("coach-profile");
  if (!box || !coachToken()) return;
  try {
    const res = await apiFetch("/api/me/profile");
    if (res.status === 401) { coachLogout(); return; }
    const p = await res.json();
    if (!p.sessions_count) {
      box.innerHTML = `<span class="coach-stat">还没有对练记录 · 完成一次「结束并评分」后这里会显示你的成长数据</span>`;
      return;
    }
    const trend = p.trend_vs_earlier;
    const trendHtml = trend === null || trend === undefined ? "" :
      `<span class="coach-stat ${trend >= 0 ? "up" : "down"}">${trend >= 0 ? "▲" : "▼"} ${Math.abs(trend)}</span>`;
    const weak = (p.weakest_dimensions || []).slice(0, 2).map(d => esc(d.name)).join("、");
    box.innerHTML = `
      <span class="coach-stat"><b>${p.sessions_count}</b> 次对练</span>
      <span class="coach-stat">均分 <b>${p.avg_score}</b></span>${trendHtml}
      ${weak ? `<span class="coach-stat">待加强：${weak}</span>` : ""}`;
  } catch (_) { /* non-fatal */ }
}

function coachKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    coachSubmit(event);
  }
}

function coachSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("coach-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  coachAsk(text);
}

async function coachAsk(text) {
  if (coachBusy || !text) return;
  coachBusy = true;
  document.getElementById("coach-send").disabled = true;
  document.getElementById("coach-quick").style.display = "none";
  coachAppend("user", text);
  const thinking = coachAppend("assistant", "", { thinking: true });

  try {
    const res = await apiFetch("/api/coach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: coachHistory }),
    });
    const data = await res.json();
    thinking.remove();
    if (res.status === 401) { coachLogout(); showError("登录已过期，请重新登录"); return; }
    if (!res.ok || data.error) {
      coachAppend("assistant", "⚠️ " + (data.error || data.detail || `请求失败 (HTTP ${res.status})`));
      return;
    }
    coachHistory.push({ role: "user", content: text }, { role: "assistant", content: data.answer });
    coachHistory = coachHistory.slice(-12);
    coachAppend("assistant", data.answer, data);
    if ((data.remembered || []).length) coachRefreshProfile();
  } catch (e) {
    thinking.remove();
    coachAppend("assistant", "⚠️ 网络出错：" + e.message);
  } finally {
    coachBusy = false;
    document.getElementById("coach-send").disabled = false;
  }
}

/** Append a message; for assistant turns with agent output, render sources / scripts / trace. */
function coachAppend(role, text, extra = {}) {
  const box = document.getElementById("coach-messages");
  const wrap = document.createElement("div");
  wrap.className = `coach-msg ${role}`;
  if (extra.thinking) {
    wrap.innerHTML = `<div class="coach-bubble thinking">教练正在查资料<span>.</span><span>.</span><span>.</span></div>`;
  } else if (role === "user") {
    wrap.innerHTML = `<div class="coach-bubble">${esc(text)}</div>`;
  } else {
    wrap.innerHTML = `<div class="coach-bubble">${coachMarkdown(text, extra.sources || [])}</div>`
      + coachSourcesHtml(extra.sources) + coachScriptsHtml(extra.scripts)
      + coachTraceHtml(extra.tool_trace, extra.usage, extra.remembered);
  }
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

/** Tiny, safe markdown: escape first, then re-introduce a few constructs. */
function coachMarkdown(text, sources) {
  const byRef = {};
  for (const s of sources) byRef[s.ref] = s;
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(D\d+)\]/g, (m, ref) => byRef[ref]
      ? `<sup><a href="${esc(byRef[ref].url)}" target="_blank" rel="noopener" title="${esc(byRef[ref].title)}">${ref}</a></sup>`
      : "");
  const lines = esc(text || "").split("\n");
  let html = "", list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^\s*\d+[.)、]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^#{1,4}\s+(.*)$/))) {
      closeList(); html += `<h4>${inline(m[1])}</h4>`;
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList(); html += `<p>${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function coachSourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  return `<div class="coach-sources"><span class="coach-sources-label">📄 参考文档</span>${sources.map(s =>
    `<a href="${esc(s.url)}" target="_blank" rel="noopener">[${esc(s.ref)}] ${esc(s.title)}${s.heading ? " · " + esc(s.heading) : ""}</a>`
  ).join("")}</div>`;
}

function coachScriptsHtml(scripts) {
  if (!scripts || !scripts.length) return "";
  return `<div class="coach-scripts">${scripts.map(s =>
    `<button class="coach-script" onclick="coachPractice(${Number(s.script_id)})">
       🎭 ${esc(s.title)} <span>${esc(s.business_type || "")}${s.difficulty ? " · 难度" + esc(s.difficulty) : ""}</span>
     </button>`).join("")}</div>`;
}

function coachTraceHtml(trace, usage, remembered) {
  if (!trace || !trace.length) return "";
  const steps = trace.map(t => `<li class="${t.ok ? "" : "fail"}"><code>${esc(t.tool)}</code> ${esc(t.summary)} <span>${t.ms}ms</span></li>`).join("");
  const u = usage ? `<li class="meta">${usage.llm_calls} 次模型调用 · ${usage.prompt_tokens + usage.completion_tokens} tokens · ${(usage.latency_ms / 1000).toFixed(1)}s · ${esc((usage.models || []).join(", "))}</li>` : "";
  const mem = (remembered || []).length ? `<li class="meta">📒 已记住：${remembered.map(esc).join("；")}</li>` : "";
  return `<details class="coach-trace"><summary>🔧 教练做了什么（${trace.length} 步）</summary><ul>${steps}${mem}${u}</ul></details>`;
}

async function coachPractice(scriptId) {
  closeCoach();
  await openLibraryItem(scriptId);
  if (currentScript) startRoleplay();
}
