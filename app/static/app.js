/* =============================================
   TicketCoach Frontend Logic
   Pure JS, talks to the FastAPI backend via fetch.
   ============================================= */

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Roleplay state: the script currently loaded, and the live transcript.
let currentScript = null;
let roleplayHistory = []; // [{role: "customer"|"agent", text: string}]

/** Escape user/model-provided text before injecting into innerHTML. */
function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Map a 0-100 score to a level class suffix. */
function scoreLevel100(score) {
  if (score >= 80) return "high";
  if (score >= 60) return "mid";
  return "low";
}

/** Map a 0-10 dimension score to a level class suffix. */
function scoreLevel10(score) {
  if (score >= 8) return "high";
  if (score >= 6) return "mid";
  return "low";
}

function showError(message) {
  const banner = document.getElementById("error-banner");
  document.getElementById("error-msg").textContent = message;
  banner.style.display = "flex";
}

function hideError() {
  document.getElementById("error-banner").style.display = "none";
}

function setLoading(visible, text) {
  const overlay = document.getElementById("loading-overlay");
  if (text) document.getElementById("loading-text").textContent = text;
  overlay.style.display = visible ? "flex" : "none";
  document.getElementById("btn-generate").disabled = visible;
}

/**
 * Wrapper around fetch for the protected API endpoints.
 * Attaches the stored access token; on 401, prompts for the password,
 * stores it, and retries once. (No password needed when the server has
 * ACCESS_PASSWORD unset, e.g. local dev.)
 */
async function apiFetch(url, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  const token = localStorage.getItem("tc_access_token");
  if (token) headers["X-Access-Token"] = token;

  let res = await fetch(url, Object.assign({}, options, { headers }));
  if (res.status === 401) {
    const pw = prompt("请输入访问口令：");
    if (pw) {
      localStorage.setItem("tc_access_token", pw);
      headers["X-Access-Token"] = pw;
      res = await fetch(url, Object.assign({}, options, { headers }));
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// Load dropdown options from backend
// ---------------------------------------------------------------------------

async function loadOptions() {
  try {
    const res = await fetch("/api/options");
    if (!res.ok) return; // dropdowns just stay with "随机" only
    const data = await res.json();

    fillSelect("sel-business", data.business_types);
    fillSelect("sel-issue", data.issue_categories);
  } catch (e) {
    // Non-fatal: user can still generate with random params
    console.warn("Failed to load options:", e);
  }
}

function fillSelect(selectId, items) {
  const sel = document.getElementById(selectId);
  if (!sel || !Array.isArray(items)) return;
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = item;
    sel.appendChild(opt);
  }
}

// ---------------------------------------------------------------------------
// Single generation
// ---------------------------------------------------------------------------

async function handleGenerate() {
  hideError();
  setLoading(true, "正在生成中，请稍候…");

  const body = {
    business_type: document.getElementById("sel-business").value || null,
    emotion: document.getElementById("sel-emotion").value || null,
    issue_category: document.getElementById("sel-issue").value || null,
    difficulty: document.getElementById("sel-difficulty").value || null,
  };

  try {
    const res = await apiFetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || data.detail || `请求失败 (HTTP ${res.status})`);
      return;
    }

    currentScript = data.script;
    renderTicket(data.ticket);
    renderScript(data.script);
    renderReview(data.review);
  } catch (e) {
    showError(`网络或解析错误：${e.message}`);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Column 1: Ticket rendering
// ---------------------------------------------------------------------------

function renderTicket(ticket) {
  const el = document.getElementById("ticket-body");
  if (!ticket) {
    el.innerHTML = `<div class="placeholder-msg">无工单数据</div>`;
    return;
  }

  const meta = `
    <div class="ticket-meta">
      <span class="meta-item"><strong>业务</strong> ${esc(ticket.business_type)}</span>
      <span class="meta-item"><strong>渠道</strong> ${esc(ticket.channel)}</span>
      <span class="meta-item"><strong>编号</strong> ${esc(ticket.ticket_id)}</span>
    </div>
    <div class="ticket-meta">
      <span class="meta-item"><strong>客户</strong> ${esc(ticket.customer_profile)}</span>
    </div>
  `;

  const tags = Array.isArray(ticket.tags) && ticket.tags.length
    ? `<div class="tags-row">${ticket.tags.map(t => `<span class="tag-chip">${esc(t)}</span>`).join("")}</div>`
    : "";

  const dialogue = Array.isArray(ticket.dialogue)
    ? `<div class="dialogue-list">${ticket.dialogue.map(renderBubble).join("")}</div>`
    : "";

  const resolution = ticket.resolution
    ? `<div class="resolution-box">
         <span class="resolution-label">最终处理结果</span>
         ${esc(ticket.resolution)}
       </div>`
    : "";

  el.innerHTML = meta + tags + dialogue + resolution;
}

function renderBubble(turn) {
  const role = turn.role === "agent" ? "agent" : "customer";
  const roleLabel = role === "agent" ? "客服" : "客户";
  return `
    <div class="bubble-wrap ${role}">
      <span class="bubble-role">${roleLabel}</span>
      <div class="bubble ${role}">${esc(turn.text)}</div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Column 2: Script rendering
// ---------------------------------------------------------------------------

function renderScript(script) {
  const el = document.getElementById("script-body");
  if (!script) {
    el.innerHTML = `<div class="placeholder-msg">无剧本数据</div>`;
    return;
  }

  const cards = [];

  cards.push(card("剧本标题",
    `<div class="script-title-text">${esc(script.title)}</div>`));

  cards.push(card("客户画像", esc(script.customer_persona)));
  cards.push(card("场景背景", esc(script.scenario)));
  cards.push(card("情绪曲线", esc(script.emotion_arc)));

  if (Array.isArray(script.challenge_points)) {
    const items = script.challenge_points.map((p, i) => `
      <div class="challenge-item">
        <span class="challenge-bullet">${i + 1}</span>
        <span>${esc(p)}</span>
      </div>`).join("");
    cards.push(card("关键挑战点", `<div class="challenge-list">${items}</div>`));
  }

  cards.push(card("标准应对策略", esc(script.standard_response)));

  if (Array.isArray(script.scoring_criteria)) {
    const rows = script.scoring_criteria.map(c => `
      <tr>
        <td class="dim-name">${esc(c.dimension)}</td>
        <td>${esc(c.description)}</td>
      </tr>`).join("");
    cards.push(card("评分维度", `
      <table class="criteria-table">
        <thead><tr><th>维度</th><th>说明</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`));
  }

  if (script.actor_prompt) {
    cards.push(card("角色扮演 Prompt",
      `<div class="actor-prompt-box">${esc(script.actor_prompt)}</div>`));
  }

  const startBtn = script.actor_prompt
    ? `<button class="start-roleplay-btn" onclick="startRoleplay()">🎭 用这个剧本开始实战对练</button>`
    : "";

  el.innerHTML = startBtn + cards.join("");
}

function card(headerText, bodyHtml) {
  return `
    <div class="script-card">
      <div class="script-card-header">${esc(headerText)}</div>
      <div class="script-card-body">${bodyHtml}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Column 3: Review rendering
// ---------------------------------------------------------------------------

function renderReview(review) {
  const el = document.getElementById("review-body");
  if (!review) {
    el.innerHTML = `<div class="placeholder-msg">无评审数据</div>`;
    return;
  }

  const overall = Number(review.overall_score) || 0;
  const level = scoreLevel100(overall);

  const circle = `
    <div class="score-circle-wrap">
      <div class="score-circle score-${level}">
        <span class="score-number">${Math.round(overall)}</span>
        <span class="score-label">综合得分 / 100</span>
      </div>
    </div>`;

  let dims = "";
  if (Array.isArray(review.dimensions)) {
    dims = `<div class="dimension-list">${review.dimensions.map(renderDimension).join("")}</div>`;
  }

  const issues = renderList(
    review.issues, "发现的问题", "issue-list", "issue-item", "issue-icon", "⚠️");
  const suggestions = renderList(
    review.suggestions, "改进建议", "suggestion-list", "suggestion-item", "suggestion-icon", "💡");

  el.innerHTML = circle + dims + issues + suggestions;
}

function renderDimension(dim) {
  const score = Number(dim.score) || 0;
  const level = scoreLevel10(score);
  const pct = Math.max(0, Math.min(100, score * 10));
  return `
    <div class="dimension-item">
      <div class="dim-header">
        <span class="dim-name-label">${esc(dim.name)}</span>
        <span class="dim-score-badge ${level}">${score} / 10</span>
      </div>
      <div class="dim-bar-bg">
        <div class="dim-bar-fill ${level}" style="width:${pct}%"></div>
      </div>
      <div class="dim-comment">${esc(dim.comment)}</div>
    </div>`;
}

function renderList(items, title, listClass, itemClass, iconClass, icon) {
  if (!Array.isArray(items) || items.length === 0) return "";
  const lis = items.map(it => `
    <div class="${itemClass}">
      <span class="${iconClass}">${icon}</span>
      <span>${esc(it)}</span>
    </div>`).join("");
  return `
    <div class="review-section">
      <div class="review-section-title">${esc(title)}</div>
      <div class="${listClass}">${lis}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Batch generation
// ---------------------------------------------------------------------------

async function handleBatch() {
  hideError();
  const n = parseInt(document.getElementById("batch-n").value, 10) || 3;

  document.getElementById("batch-result").style.display = "none";
  const loadingEl = document.getElementById("batch-loading");
  loadingEl.style.display = "flex";
  document.getElementById("batch-loading-text").textContent = `批量生成 ${n} 条中…（每条约 30-60 秒）`;
  document.getElementById("btn-batch").disabled = true;

  try {
    const res = await apiFetch(`/api/batch?n=${n}`, { method: "POST" });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || data.detail || `批量请求失败 (HTTP ${res.status})`);
      return;
    }

    renderBatchStats(data.stats);
    renderBatchList(data.results || []);
    document.getElementById("batch-result").style.display = "block";
  } catch (e) {
    showError(`批量生成错误：${e.message}`);
  } finally {
    loadingEl.style.display = "none";
    document.getElementById("btn-batch").disabled = false;
  }
}

function renderBatchStats(stats) {
  const el = document.getElementById("stats-summary");
  if (!stats) {
    el.innerHTML = "";
    return;
  }

  const cards = `
    <div class="stat-card">
      <div class="stat-card-label">成功生成</div>
      <div class="stat-card-value">${stats.total_success} / ${stats.total_requested}</div>
      <div class="stat-card-sub">失败 ${stats.total_failed} 条</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">平均质检分</div>
      <div class="stat-card-value">${stats.avg_score}</div>
      <div class="stat-card-sub">满分 100</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">业务类型分布</div>
      ${distTable(stats.business_type_distribution)}
    </div>
    <div class="stat-card">
      <div class="stat-card-label">难度分布</div>
      ${distTable(stats.difficulty_distribution)}
    </div>`;

  el.innerHTML = cards;
}

function distTable(dist) {
  if (!dist || Object.keys(dist).length === 0) return `<div class="stat-card-sub">无数据</div>`;
  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(e => e[1]));
  const rows = entries.map(([key, count]) => {
    const pct = max ? (count / max) * 100 : 0;
    return `
      <tr>
        <td>${esc(key)}</td>
        <td>
          <div class="dist-bar-wrap">
            <div class="dist-bar-bg"><div class="dist-bar-fill" style="width:${pct}%"></div></div>
            <span class="dist-count">${count}</span>
          </div>
        </td>
      </tr>`;
  }).join("");
  return `<table class="dist-table"><tbody>${rows}</tbody></table>`;
}

function renderBatchList(results) {
  const el = document.getElementById("batch-list");
  if (!results.length) {
    el.innerHTML = `<div class="placeholder-msg">无结果</div>`;
    return;
  }

  el.innerHTML = results.map((r, i) => {
    const score = Number(r.review?.overall_score) || 0;
    const level = scoreLevel100(score);
    const title = r.script?.title || `工单 ${r.ticket?.ticket_id || ""}`;
    const bodyId = `batch-body-${i}`;
    return `
      <div class="batch-item">
        <div class="batch-item-header" onclick="toggleBatchItem('${bodyId}')">
          <span class="batch-item-num">#${i + 1}</span>
          <span class="batch-item-title">${esc(title)}</span>
          <span class="batch-item-score dim-score-badge ${level}">${Math.round(score)}</span>
          <span class="batch-item-toggle">展开 ▾</span>
        </div>
        <div class="batch-item-body" id="${bodyId}">
          ${batchItemDetail(r)}
        </div>
      </div>`;
  }).join("");
}

function batchItemDetail(r) {
  const params = r.params || {};
  const challenge = Array.isArray(r.script?.challenge_points)
    ? r.script.challenge_points.map(p => `<div class="challenge-item"><span>• ${esc(p)}</span></div>`).join("")
    : "";
  return `
    <div class="ticket-meta">
      <span class="meta-item"><strong>业务</strong> ${esc(params.business_type)}</span>
      <span class="meta-item"><strong>情绪</strong> ${esc(params.emotion)}</span>
      <span class="meta-item"><strong>难度</strong> ${esc(params.difficulty)}</span>
    </div>
    <div class="script-card">
      <div class="script-card-header">场景</div>
      <div class="script-card-body">${esc(r.script?.scenario)}</div>
    </div>
    <div class="script-card">
      <div class="script-card-header">挑战点</div>
      <div class="script-card-body"><div class="challenge-list">${challenge}</div></div>
    </div>`;
}

function toggleBatchItem(bodyId) {
  const body = document.getElementById(bodyId);
  if (!body) return;
  body.classList.toggle("open");
  const toggle = body.previousElementSibling.querySelector(".batch-item-toggle");
  if (toggle) toggle.textContent = body.classList.contains("open") ? "收起 ▴" : "展开 ▾";
}

// ---------------------------------------------------------------------------
// Roleplay chat engine (live sparring)
// ---------------------------------------------------------------------------

function startRoleplay() {
  if (!currentScript || !currentScript.actor_prompt) {
    showError("请先生成一个包含角色扮演 Prompt 的剧本");
    return;
  }
  roleplayHistory = [];
  document.getElementById("roleplay-title").textContent = currentScript.title || "实战对练";
  document.getElementById("roleplay-messages").innerHTML = "";
  document.getElementById("roleplay-input").value = "";
  document.getElementById("roleplay-overlay").style.display = "flex";
  document.getElementById("roleplay-input").focus();
  // The customer (AI) opens the conversation.
  fetchCustomerReply();
}

function closeRoleplay() {
  document.getElementById("roleplay-overlay").style.display = "none";
}

function handleChatKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

function sendChatMessage() {
  const input = document.getElementById("roleplay-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  roleplayHistory.push({ role: "agent", text });
  appendChatBubble("agent", text);
  fetchCustomerReply();
}

async function fetchCustomerReply() {
  setChatBusy(true);
  appendTyping();
  try {
    const res = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actor_prompt: currentScript.actor_prompt,
        history: roleplayHistory,
      }),
    });
    const data = await res.json();
    removeTyping();
    if (!res.ok || data.error) {
      showError(data.error || `对话请求失败 (HTTP ${res.status})`);
      return;
    }
    roleplayHistory.push({ role: "customer", text: data.reply });
    appendChatBubble("customer", data.reply);
  } catch (e) {
    removeTyping();
    showError(`对话出错：${e.message}`);
  } finally {
    setChatBusy(false);
  }
}

function appendChatBubble(role, text) {
  const box = document.getElementById("roleplay-messages");
  const roleLabel = role === "agent" ? "客服（你）" : "客户（AI）";
  const wrap = document.createElement("div");
  wrap.className = `bubble-wrap ${role}`;
  wrap.innerHTML = `
    <span class="bubble-role">${roleLabel}</span>
    <div class="bubble ${role}">${esc(text)}</div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function appendTyping() {
  const box = document.getElementById("roleplay-messages");
  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap customer";
  wrap.id = "typing-indicator";
  wrap.innerHTML = `<div class="bubble customer typing"><span></span><span></span><span></span></div>`;
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById("typing-indicator");
  if (t) t.remove();
}

function setChatBusy(busy) {
  document.getElementById("roleplay-send").disabled = busy;
  document.getElementById("roleplay-input").disabled = busy;
}

// ---------------------------------------------------------------------------
// Session evaluation
// ---------------------------------------------------------------------------

async function endAndEvaluate() {
  if (!roleplayHistory.some(t => t.role === "agent")) {
    showError("你还没以客服身份说过话，无法评分");
    return;
  }

  const evalBody = document.getElementById("eval-body");
  evalBody.innerHTML = `
    <div class="eval-loading">
      <div class="spinner"></div>
      <p class="loading-text">正在根据评分维度评估你的表现…</p>
    </div>`;
  document.getElementById("eval-overlay").style.display = "flex";

  try {
    const res = await apiFetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script: currentScript, transcript: roleplayHistory }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      evalBody.innerHTML = `<div class="placeholder-msg">${esc(data.error || "评分失败")}</div>`;
      return;
    }
    renderEvalResult(data);
  } catch (e) {
    evalBody.innerHTML = `<div class="placeholder-msg">评分出错：${esc(e.message)}</div>`;
  }
}

function closeEval() {
  document.getElementById("eval-overlay").style.display = "none";
}

function renderEvalResult(result) {
  const overall = Number(result.overall_score) || 0;
  const level = scoreLevel100(overall);

  const circle = `
    <div class="score-circle-wrap">
      <div class="score-circle score-${level}">
        <span class="score-number">${Math.round(overall)}</span>
        <span class="score-label">你的得分 / 100</span>
      </div>
    </div>`;

  let dims = "";
  if (Array.isArray(result.dimensions)) {
    dims = `<div class="dimension-list">${result.dimensions.map(renderDimension).join("")}</div>`;
  }

  const highlights = renderList(
    result.highlights, "做得好的地方", "suggestion-list", "suggestion-item", "suggestion-icon", "✅");
  const improvements = renderList(
    result.improvements, "可改进的地方", "issue-list", "issue-item", "issue-icon", "🔧");
  const missed = renderList(
    result.missed_challenge_points, "未接住的挑战点", "issue-list", "issue-item", "issue-icon", "⚠️");

  document.getElementById("eval-body").innerHTML =
    circle + dims + highlights + improvements + missed;
}
