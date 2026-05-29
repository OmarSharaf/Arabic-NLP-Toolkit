const SAMPLES = {
  egyptian: "ازيك عامل ايه النهارده يا صاحبي؟ المنتج ده تحفة أوي",
  gulf: "شلون حالك اليوم؟ وين رحت البارحة؟ الخدمة ممتازة",
  msa: "إن التعليم هو أساس تقدم الأمم وازدهارها في العصر الحديث",
  social: "ازيك يا صاحبي 😂 شوف #مصر https://example.com @user",
  negative: "الخدمة سيئة جداً ومش راضي عنها خالص، مهدر للوقت والفلوس",
};

const TOOL_LABELS = {
  analyze: "Full analysis",
  dialect: "Dialect detection",
  sentiment: "Sentiment",
  ner: "Named entities",
  tokenize: "Tokenization",
  normalize: "Normalization",
  keywords: "Keywords",
  profile: "Text profiling",
  pos: "POS tagging",
  stats: "Statistics",
  transliterate: "Transliteration",
};

let currentTool = "analyze";
let lastResult = null;

const $ = (sel) => document.querySelector(sel);

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    $("#status-badge").textContent = "Online";
    $("#status-badge").className = "badge badge-ok";
    $("#version-badge").textContent = `v${data.version}`;
  } catch {
    $("#status-badge").textContent = "Offline";
    $("#status-badge").className = "badge";
  }
}

function getText() {
  return $("#input-text").value.trim();
}

async function runAnalysis() {
  const text = getText();
  if (!text) {
    showError("Please enter some Arabic text.");
    return;
  }

  $("#run-btn").disabled = true;
  $("#loading").classList.remove("hidden");
  $("#error").classList.add("hidden");
  $("#cards-view").innerHTML = "";
  $("#json-view").textContent = "";

  const t0 = performance.now();
  let endpoint = `/api/${currentTool}`;
  let body = { text };

  if (currentTool === "transliterate") {
    body = {
      text,
      source: $("#translit-source").value,
      target: $("#translit-target").value,
    };
  }

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    lastResult = data;
    const ms = Math.round(performance.now() - t0);
    $("#timing").textContent = `${ms} ms`;
    $("#output-title").textContent = TOOL_LABELS[currentTool] || "Results";

    renderCards(currentTool, data);
    $("#json-view").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    showError(err.message || "Request failed");
  } finally {
    $("#loading").classList.add("hidden");
    $("#run-btn").disabled = false;
  }
}

function showError(msg) {
  $("#error").textContent = msg;
  $("#error").classList.remove("hidden");
}

function renderCards(tool, data) {
  const container = $("#cards-view");
  container.innerHTML = "";

  const renderers = {
    analyze: renderAnalyze,
    dialect: renderDialect,
    sentiment: renderSentiment,
    ner: renderNer,
    tokenize: renderTokenize,
    normalize: renderNormalize,
    keywords: renderKeywords,
    profile: renderProfile,
    pos: renderPos,
    stats: renderStats,
    transliterate: renderTransliterate,
  };

  (renderers[tool] || renderGeneric)(container, data);
}

function card(title, html) {
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="card-title">${title}</div>${html}`;
  return el;
}

function renderDialect(container, d) {
  container.appendChild(
    card("Dialect", `<div class="card-value large">${d.dialect} <span style="font-size:0.6em;color:var(--text-muted)">${(d.confidence * 100).toFixed(0)}%</span></div>`)
  );
  if (d.all_scores?.length) {
    const list = d.all_scores
      .slice(0, 6)
      .map(
        (s) =>
          `<div style="margin:0.4rem 0"><span>${s.dialect}</span> <span style="float:left">${(s.confidence * 100).toFixed(0)}%</span><div class="score-bar"><div class="score-fill" style="width:${s.confidence * 100}%"></div></div></div>`
      )
      .join("");
    container.appendChild(card("All scores", list));
  }
}

function renderSentiment(container, d) {
  const emoji = { positive: "😊", negative: "😞", neutral: "😐", mixed: "😕" };
  const cls = `tag-${d.label}`;
  container.appendChild(
    card(
      "Sentiment",
      `<div class="sentiment-emoji">${emoji[d.label] || ""}</div>
       <div class="card-value"><span class="tag ${cls}">${d.label}</span> ${(d.score * 100).toFixed(0)}%</div>
       <div style="margin-top:0.75rem;font-size:0.85rem">
         + ${(d.positive_score * 100).toFixed(0)}% &nbsp; − ${(d.negative_score * 100).toFixed(0)}% &nbsp; ○ ${(d.neutral_score * 100).toFixed(0)}%
       </div>`
    )
  );
}

function renderNer(container, d) {
  const items = (d.entities || [])
    .map((e) => {
      const tagCls =
        e.label === "PERSON" ? "tag-person" : e.label === "LOCATION" ? "tag-location" : e.label === "ORGANIZATION" ? "tag-org" : "";
      return `<li class="entity-item"><span>${e.text}</span><span class="tag ${tagCls}">${e.label}</span></li>`;
    })
    .join("");
  container.appendChild(
    card(`Entities (${d.entities?.length || 0})`, `<ul class="entity-list">${items || "<li>No entities found</li>"}</ul>`)
  );
}

function renderTokenize(container, d) {
  const items = (d.tokens || [])
    .map((t) => `<li class="token-item"><span>${t.text}</span><span class="tag">${t.token_type}</span></li>`)
    .join("");
  container.appendChild(card(`Tokens (${d.count})`, `<ul class="token-list">${items}</ul>`));
}

function renderNormalize(container, d) {
  container.appendChild(
    card("Normalized", `<div class="card-value" style="font-size:1rem">${d.normalized}</div>`)
  );
  if (d.changes?.length) {
    container.appendChild(card("Changes", d.changes.map((c) => `<span class="tag">${c}</span> `).join("")));
  }
}

function renderKeywords(container, d) {
  const items = (d.keywords || [])
    .map(
      (k) =>
        `<li class="kw-item"><span>${k.text}</span><span>${(k.score * 100).toFixed(0)}% · ×${k.frequency}</span></li>`
    )
    .join("");
  container.appendChild(card("Keywords", `<ul class="kw-list">${items || "<li>None</li>"}</ul>`));
}

function renderProfile(container, d) {
  container.appendChild(
    card(
      "Profile",
      `<div class="card-grid">
        <div><div class="card-title">Register</div><div class="card-value">${d.text_register}</div></div>
        <div><div class="card-title">Quality</div><div class="card-value">${(d.quality_score * 100).toFixed(0)}%</div></div>
        <div><div class="card-title">Dialect</div><div class="card-value">${d.dialect}</div></div>
        <div><div class="card-title">Words</div><div class="card-value">${d.word_count}</div></div>
      </div>`
    )
  );
  if (d.recommendations?.length) {
    container.appendChild(
      card("Recommendations", d.recommendations.map((r) => `<div class="recommendation">${r}</div>`).join(""))
    );
  }
}

function renderPos(container, d) {
  const items = (d.tags || [])
    .map((t) => `<li class="token-item"><span>${t.token}</span><span class="tag">${t.tag}</span></li>`)
    .join("");
  container.appendChild(card("POS tags", `<ul class="token-list">${items}</ul>`));
}

function renderStats(container, d) {
  const s = d.statistics || {};
  const r = d.readability || {};
  container.appendChild(
    card(
      "Statistics",
      `<div class="card-grid">
        <div><div class="card-title">Words</div><div class="card-value">${s.total_words}</div></div>
        <div><div class="card-title">Unique</div><div class="card-value">${s.unique_words}</div></div>
        <div><div class="card-title">TTR</div><div class="card-value">${((s.type_token_ratio || 0) * 100).toFixed(0)}%</div></div>
        <div><div class="card-title">Readability</div><div class="card-value">${r.difficulty_level}</div></div>
      </div>
      <p class="recommendation">${r.recommendation || ""}</p>`
    )
  );
}

function renderTransliterate(container, d) {
  container.appendChild(
    card("Result", `<div class="card-value" style="font-size:1.1rem;direction:ltr;text-align:left">${d.transliterated}</div>`)
  );
}

function renderAnalyze(container, d) {
  renderDialect(container, d.dialect);
  renderSentiment(container, d.sentiment);
  if (d.entities?.entities) renderNer(container, d.entities);
  if (d.keywords?.length) {
    renderKeywords(container, { keywords: d.keywords });
  }
  if (d.profile) renderProfile(container, d.profile);
  container.appendChild(
    card("Summary", `<div style="font-size:0.9rem">Tokens: ${d.token_count} · Entities: ${d.entity_count} · Pipeline: ${d.pipeline_time_ms} ms</div>`)
  );
}

function renderGeneric(container, d) {
  container.appendChild(card("Result", `<pre style="font-size:0.8rem;overflow:auto">${JSON.stringify(d, null, 2)}</pre>`));
}

// Event listeners
document.querySelectorAll(".tool-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tool-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentTool = btn.dataset.tool;
    $("#translit-options").classList.toggle("hidden", currentTool !== "transliterate");
  });
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#input-text").value = SAMPLES[chip.dataset.sample] || "";
  });
});

$("#run-btn").addEventListener("click", runAnalysis);

$("#view-cards").addEventListener("click", () => {
  $("#view-cards").classList.add("active");
  $("#view-json").classList.remove("active");
  $("#cards-view").classList.remove("hidden");
  $("#json-view").classList.add("hidden");
});

$("#view-json").addEventListener("click", () => {
  $("#view-json").classList.add("active");
  $("#view-cards").classList.remove("active");
  $("#json-view").classList.remove("hidden");
  $("#cards-view").classList.add("hidden");
});

$("#copy-btn").addEventListener("click", () => {
  const text = $("#json-view").classList.contains("hidden")
    ? JSON.stringify(lastResult, null, 2)
    : $("#json-view").textContent;
  navigator.clipboard.writeText(text);
});

$("#input-text").addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") runAnalysis();
});

// ─── View switching (Playground / Project Profile) ───
function switchView(name) {
  document.querySelectorAll(".nav-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.view === name);
  });
  $("#view-playground").classList.toggle("active", name === "playground");
  $("#view-playground").classList.toggle("hidden", name !== "playground");
  $("#view-profile").classList.toggle("active", name === "profile");
  $("#view-profile").classList.toggle("hidden", name !== "profile");
  if (name === "profile" && !window._profileLoaded) {
    loadProjectProfile();
  }
  if (location.hash !== `#${name}`) {
    history.replaceState(null, "", `#${name}`);
  }
}

document.querySelectorAll(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

$("#goto-playground")?.addEventListener("click", () => switchView("playground"));

async function loadProjectProfile() {
  try {
    const res = await fetch("/api/project");
    const p = await res.json();
    window._profileLoaded = true;

    $("#profile-version").textContent = `v${p.version}`;
    $("#profile-name").textContent = p.name;
    $("#profile-tagline").textContent = p.tagline;
    $("#profile-tagline-ar").textContent = p.tagline_ar;
    $("#profile-install").textContent = p.install;

    $("#link-github").href = p.links.github;
    $("#link-pypi").href = p.links.pypi;

    const statsEl = $("#profile-stats");
    statsEl.innerHTML = Object.entries(p.stats)
      .map(
        ([k, v]) =>
          `<div class="stat-card"><div class="stat-value">${v}</div><div class="stat-label">${k}</div></div>`
      )
      .join("");

    $("#profile-features").innerHTML = p.features
      .map(
        (f) => `
      <article class="feature-card">
        <div class="feature-icon">${f.icon}</div>
        <h4>${f.title}</h4>
        <div class="feature-ar">${f.title_ar}</div>
        <p>${f.desc}</p>
      </article>`
      )
      .join("");

    const a = p.author;
    $("#profile-author").innerHTML = `
      <h4>Author · المؤلف</h4>
      <div class="author-name">${a.name}</div>
      <p style="color:var(--text-muted);font-size:0.9rem">Built with care from Egypt 🇪🇬</p>
      <div class="author-links">
        <a href="${a.website}" target="_blank" rel="noopener">Website</a>
        <a href="${a.github}" target="_blank" rel="noopener">GitHub</a>
        <a href="mailto:${a.email}">Email</a>
      </div>
    `;
  } catch (e) {
    console.error("Failed to load project profile", e);
  }
}

function initFromHash() {
  const view = location.hash.replace("#", "") || "playground";
  if (view === "profile") {
    switchView("profile");
  } else {
    switchView("playground");
    runAnalysis();
  }
}

checkHealth();
initFromHash();
