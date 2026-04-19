<script setup>
import { ref, computed, nextTick } from 'vue'
import { marked } from 'marked'

const MODES = [
  { value: 'fast',     label: 'Fast',     desc: '3 queries · ~$0.07' },
  { value: 'research', label: 'Research', desc: '5 queries · ~$0.30'  },
]
const FORMATS = [
  { value: 'markdown', label: 'Markdown', glyph: '#'  },
  { value: 'docx',     label: 'Word',     glyph: 'W'  },
  { value: 'pdf',      label: 'PDF',      glyph: 'P'  },
  { value: 'pptx',     label: 'Slides',   glyph: 'S'  },
]
const SUGGESTED = [
  { icon: '⚡', text: 'Latest EV launches in India 2026 — price, range, features' },
  { icon: '🔬', text: 'Quantum computing breakthroughs in 2025–2026' },
  { icon: '🚗', text: 'Compare Renault Duster vs Hyundai Creta 2026 specs' },
]

// ── State ────────────────────────────────────────────────────────────────
const query         = ref('')
const mode          = ref('fast')
const format        = ref('markdown')
const isRunning     = ref(false)
const sessionId     = ref(null)
const events        = ref([])
const markdownContent = ref(null)   // always the readable text
const binaryUri     = ref(null)     // set only for docx/pdf/pptx
const error         = ref(null)
const copied        = ref(false)
const startTime     = ref(null)
const timelineEl    = ref(null)
const stats         = ref({ cost: null, tokens: null, elapsed: null })
const showLog       = ref(false)
const sessions      = ref([])       // { id, query, ts }

// ── Derived ──────────────────────────────────────────────────────────────
const hasContent     = computed(() => markdownContent.value !== null || events.value.length > 0)
const isBinary       = computed(() => binaryUri.value !== null)
const renderedResult = computed(() =>
  markdownContent.value ? marked.parse(markdownContent.value) : ''
)

// ── Helpers ──────────────────────────────────────────────────────────────
function nowTS() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}
function scrollTimeline() {
  nextTick(() => { if (timelineEl.value) timelineEl.value.scrollTop = timelineEl.value.scrollHeight })
}
function useQuery(text) { query.value = text }

// ── Research pipeline ────────────────────────────────────────────────────
async function startResearch() {
  if (!query.value.trim() || isRunning.value) return
  events.value    = []
  markdownContent.value = null
  binaryUri.value = null
  error.value     = null
  copied.value    = false
  stats.value     = { cost: null, tokens: null, elapsed: null }
  isRunning.value = true
  showLog.value   = false
  startTime.value = Date.now()

  let res
  try {
    res = await fetch('/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.value, mode: mode.value, format: format.value }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  } catch (err) {
    error.value = err.message
    isRunning.value = false
    return
  }

  const { session_id } = await res.json()
  sessionId.value = session_id
  sessions.value.unshift({ id: session_id, query: query.value, ts: nowTS() })

  const es = new EventSource(`/stream/${session_id}`)

  es.addEventListener('status', e => {
    const d = JSON.parse(e.data)
    events.value.push({ phase: d.phase || 'pipeline', message: d.message, ts: nowTS() })
    scrollTimeline()
  })

  es.addEventListener('result', e => {
    const d = JSON.parse(e.data)
    const raw = d.final_output || ''
    // Detect compound JSON from binary formats: { binary, markdown }
    try {
      const parsed = JSON.parse(raw)
      if (parsed && parsed.binary && parsed.markdown) {
        binaryUri.value = parsed.binary
        markdownContent.value = parsed.markdown
      } else {
        markdownContent.value = raw
      }
    } catch (_) {
      markdownContent.value = raw
    }
    stats.value = {
      cost:    d.cost_usd ?? 0,
      tokens:  d.tokens   ?? 0,
      elapsed: ((Date.now() - startTime.value) / 1000).toFixed(1),
    }
  })

  es.addEventListener('error', e => {
    if (e.data) {
      try { error.value = JSON.parse(e.data).message } catch (_) { error.value = e.data }
    }
  })

  es.addEventListener('done', () => {
    es.close()
    isRunning.value = false
    events.value.push({ phase: 'done', message: 'Pipeline complete', ts: nowTS() })
    if (!stats.value.elapsed) stats.value.elapsed = ((Date.now() - startTime.value) / 1000).toFixed(1)
    scrollTimeline()
  })

  es.onerror = () => { es.close(); isRunning.value = false }
}

// ── Actions ──────────────────────────────────────────────────────────────
function newResearch() {
  markdownContent.value = null
  binaryUri.value = null
  events.value = []
  error.value = null
  stats.value = { cost: null, tokens: null, elapsed: null }
  query.value = ''
  sessionId.value = null
  showLog.value = false
}

function copyResult() {
  if (!markdownContent.value) return
  navigator.clipboard.writeText(markdownContent.value)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}

function downloadMarkdown() {
  if (!markdownContent.value) return
  const blob = new Blob([markdownContent.value], { type: 'text/markdown' })
  const url  = URL.createObjectURL(blob)
  Object.assign(document.createElement('a'), { href: url, download: 'research-brief.md' }).click()
  URL.revokeObjectURL(url)
}

function downloadBinary() {
  if (!binaryUri.value) return
  const ext = { docx: 'docx', pdf: 'pdf', pptx: 'pptx' }[format.value] ?? format.value
  Object.assign(document.createElement('a'), { href: binaryUri.value, download: `research-brief.${ext}` }).click()
}
</script>

<template>
  <div class="shell">

    <!-- ── Sidebar ────────────────────────────────────────────────────── -->
    <aside class="sidebar">

      <!-- Icon rail -->
      <div class="rail">
        <div class="rail-logo">
          <svg width="20" height="20" viewBox="0 0 22 22" fill="none">
            <path d="M11 1.5L20.5 6.75V17.25L11 22.5L1.5 17.25V6.75L11 1.5Z"
                  stroke="#D4A843" stroke-width="1.1" fill="rgba(212,168,67,0.08)"/>
          </svg>
        </div>
        <div class="rail-icons">
          <button class="rail-icon active" title="Research" @click="newResearch">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
          </button>
          <button class="rail-icon" title="History">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <path d="M12 8v4l3 3"/><circle cx="12" cy="12" r="10"/>
            </svg>
          </button>
          <button class="rail-icon" title="Settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- Session panel -->
      <div class="panel">
        <button class="new-btn" @click="newResearch">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 5v14M5 12h14"/></svg>
          New Research
        </button>

        <div class="panel-section" v-if="sessions.length">
          <span class="panel-label">RECENT</span>
          <div
            v-for="s in sessions.slice(0, 12)"
            :key="s.id"
            class="session-item"
          >
            <span class="session-q">{{ s.query.slice(0, 38) }}{{ s.query.length > 38 ? '…' : '' }}</span>
            <span class="session-ts">{{ s.ts }}</span>
          </div>
        </div>

        <div class="panel-section" v-else>
          <span class="panel-label">SAVED</span>
          <p class="panel-empty">No sessions yet.<br>Start researching below.</p>
        </div>
      </div>

    </aside>

    <!-- ── Main ───────────────────────────────────────────────────────── -->
    <div class="main">

      <!-- Topbar -->
      <header class="topbar" :class="{ running: isRunning }">
        <div class="topbar-left">
          <span class="topbar-title">Deep Research Agent</span>
        </div>
        <div class="topbar-right">
          <span class="version-chip">v3.0</span>
          <div class="status-badge" :class="{ active: isRunning }">
            <span class="status-pip" />
            {{ isRunning ? 'ACTIVE' : 'READY' }}
          </div>
        </div>
      </header>

      <!-- Content -->
      <div class="content">

        <!-- Error -->
        <Transition name="slide-down">
          <div class="error-bar" v-if="error">
            <span>⚠ {{ error }}</span>
            <button class="error-close" @click="error = null">×</button>
          </div>
        </Transition>

        <!-- Welcome state -->
        <Transition name="fade">
          <div class="welcome" v-if="!hasContent && !isRunning">
            <div class="welcome-orb" />
            <h1 class="welcome-title">Hi, there.</h1>
            <p class="welcome-sub">What would you like to research today?</p>
            <div class="suggestions">
              <button
                v-for="s in SUGGESTED"
                :key="s.text"
                class="suggestion-card"
                @click="useQuery(s.text)"
              >
                <span class="sug-icon">{{ s.icon }}</span>
                <span class="sug-text">{{ s.text }}</span>
              </button>
            </div>
          </div>
        </Transition>

        <!-- Live progress -->
        <Transition name="rise">
          <div class="results" v-if="hasContent || isRunning">

            <!-- Stats strip -->
            <div class="stats-strip" v-if="stats.cost !== null">
              <div class="stat">
                <span class="stat-val">${{ Number(stats.cost).toFixed(4) }}</span>
                <span class="stat-key">COST</span>
              </div>
              <div class="stat-div" />
              <div class="stat">
                <span class="stat-val">{{ Number(stats.tokens).toLocaleString() }}</span>
                <span class="stat-key">TOKENS</span>
              </div>
              <div class="stat-div" />
              <div class="stat">
                <span class="stat-val">{{ stats.elapsed }}s</span>
                <span class="stat-key">ELAPSED</span>
              </div>
            </div>

            <!-- Research brief card -->
            <div class="brief-card" v-if="markdownContent">
              <div class="brief-header">
                <span class="brief-label">RESEARCH BRIEF</span>
                <div class="brief-actions">
                  <span v-if="isBinary" class="binary-badge">{{ format.toUpperCase() }} ready</span>
                  <button class="act-btn" :class="{ done: copied }" @click="copyResult">
                    {{ copied ? '✓ Copied' : 'Copy' }}
                  </button>
                  <button class="act-btn" v-if="!isBinary" @click="downloadMarkdown">↓ .MD</button>
                  <button class="act-btn accent" v-if="isBinary" @click="downloadBinary">
                    ↓ Save .{{ format.toUpperCase() }}
                  </button>
                </div>
              </div>
              <div class="prose" v-html="renderedResult" />
            </div>

            <!-- Live "processing" card -->
            <div class="brief-card processing" v-else-if="isRunning">
              <div class="processing-inner">
                <span class="proc-dot" />
                <span class="proc-dot" style="animation-delay:.22s"/>
                <span class="proc-dot" style="animation-delay:.44s"/>
                <span class="proc-label">Researching…</span>
              </div>
            </div>

            <!-- Pipeline log (collapsible) -->
            <div class="log-card" v-if="events.length">
              <button class="log-toggle" @click="showLog = !showLog">
                <span class="panel-label">PIPELINE LOG</span>
                <span class="log-badge">{{ events.length }}</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
                     :style="{ transform: showLog ? 'rotate(180deg)' : '', transition: 'transform .2s' }">
                  <path d="m6 9 6 6 6-6"/>
                </svg>
              </button>
              <div class="log-body" v-if="showLog" ref="timelineEl">
                <TransitionGroup name="event" tag="div" class="tl-list">
                  <div v-for="(ev, i) in events" :key="i" class="tl-row">
                    <div class="tl-node" :class="'node-' + ev.phase" />
                    <div class="tl-body">
                      <span class="tl-badge" :class="'badge-' + ev.phase">{{ ev.phase }}</span>
                      <span class="tl-ts">{{ ev.ts }}</span>
                      <p class="tl-msg">{{ ev.message }}</p>
                    </div>
                  </div>
                </TransitionGroup>
              </div>
            </div>

          </div>
        </Transition>

      </div>

      <!-- ── Input bar (sticky bottom) ─────────────────────────────── -->
      <div class="input-bar">
        <div class="input-wrap">
          <textarea
            v-model="query"
            class="query-input"
            rows="2"
            placeholder="Ask me anything… (Ctrl+↵ to send)"
            :disabled="isRunning"
            @keydown.ctrl.enter="startResearch"
          />
          <div class="input-controls">
            <!-- Mode -->
            <div class="mode-strip">
              <button
                v-for="m in MODES" :key="m.value"
                class="mode-btn" :class="{ active: mode === m.value }"
                :disabled="isRunning"
                @click="mode = m.value"
              >
                {{ m.label }}
                <span class="mode-cost">{{ m.desc }}</span>
              </button>
            </div>

            <!-- Format pills -->
            <div class="fmt-row">
              <button
                v-for="f in FORMATS" :key="f.value"
                class="fmt-btn" :class="{ active: format === f.value }"
                :disabled="isRunning"
                @click="format = f.value"
              >
                <span class="fmt-glyph">{{ f.glyph }}</span>
                {{ f.label }}
              </button>
            </div>

            <!-- Send -->
            <button
              class="send-btn"
              :class="{ running: isRunning }"
              :disabled="isRunning || !query.trim()"
              @click="startResearch"
            >
              <span class="ring ring-1" />
              <span class="ring ring-2" />
              <span class="send-text">
                <template v-if="!isRunning">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M2 21 23 12 2 3v7l15 2-15 2z"/></svg>
                  Send
                </template>
                <template v-else>
                  <span class="dots"><span>.</span><span>.</span><span>.</span></span>
                </template>
              </span>
            </button>
          </div>

          <div class="session-tag" v-if="sessionId">SID · {{ sessionId.slice(0, 8).toUpperCase() }}</div>
        </div>
      </div>

    </div>
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ──────────────────────────────────────────────────── */
.shell {
  --bg:       #0B0906;
  --surf:     #131110;
  --surf-hi:  #1C1A16;
  --bdr:      #282520;
  --bdr-hi:   #38352C;
  --acc:      #D4A843;
  --acc-dim:  #B8902F;
  --acc-glow: rgba(212,168,67,0.18);
  --text:     #EDE9DF;
  --muted:    #8A8578;
  --muted-d:  #55524C;
  --ok:       #5BAD78;
  --err:      #C25E5E;
  --r:        8px;
  --r-lg:     12px;
  --ui:       'Inter', system-ui, sans-serif;
  --mono:     'JetBrains Mono', monospace;
  --prose:    'Inter', Georgia, serif;
  --rail-w:   56px;
  --panel-w:  210px;
  --ease:     cubic-bezier(0.25, 0.46, 0.45, 0.94);

  display: flex;
  height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: var(--ui);
  overflow: hidden;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
.sidebar {
  display: flex;
  flex-shrink: 0;
  height: 100vh;
  border-right: 1px solid var(--bdr);
}

.rail {
  width: var(--rail-w);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 14px 0;
  border-right: 1px solid var(--bdr);
  background: rgba(11,9,6,0.9);
  gap: 4px;
}

.rail-logo {
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.rail-icons { display: flex; flex-direction: column; gap: 4px; }

.rail-icon {
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid transparent;
  border-radius: var(--r);
  color: var(--muted-d);
  cursor: pointer;
  transition: color .2s var(--ease), background .2s var(--ease),
              border-color .2s var(--ease), transform .15s var(--ease),
              box-shadow .2s var(--ease);
  position: relative;
}
.rail-icon:hover {
  color: var(--text);
  background: var(--surf-hi);
  border-color: var(--bdr);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,.3);
}
.rail-icon.active {
  color: var(--acc);
  background: rgba(212,168,67,.10);
  border-color: rgba(212,168,67,.30);
  box-shadow: 0 0 12px rgba(212,168,67,.12);
}

/* Tooltip */
.rail-icon[title]::after {
  content: attr(title);
  position: absolute;
  left: calc(100% + 10px);
  top: 50%;
  transform: translateY(-50%) translateX(-4px);
  background: var(--surf-hi);
  border: 1px solid var(--bdr-hi);
  color: var(--text);
  font-family: var(--ui);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .02em;
  padding: 5px 10px;
  border-radius: 6px;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity .18s var(--ease), transform .18s var(--ease);
  box-shadow: 0 4px 16px rgba(0,0,0,.4);
  z-index: 100;
}
.rail-icon[title]:hover::after {
  opacity: 1;
  transform: translateY(-50%) translateX(0);
}

.panel {
  width: var(--panel-w);
  display: flex;
  flex-direction: column;
  padding: 14px 10px;
  background: var(--surf);
  overflow-y: auto;
  gap: 12px;
}

.new-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 8px 12px;
  background: rgba(212,168,67,.07);
  border: 1px solid rgba(212,168,67,.25);
  border-radius: var(--r);
  color: var(--acc);
  font-family: var(--ui);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background .2s var(--ease), border-color .2s var(--ease),
              box-shadow .2s var(--ease), transform .15s var(--ease);
}
.new-btn:hover {
  background: rgba(212,168,67,.14);
  border-color: rgba(212,168,67,.5);
  box-shadow: 0 0 14px rgba(212,168,67,.15), 0 2px 8px rgba(0,0,0,.25);
  transform: translateY(-1px);
}
.new-btn:active { transform: translateY(0); }

.panel-section { display: flex; flex-direction: column; gap: 4px; }

.panel-label {
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: .14em;
  color: var(--muted-d);
  padding: 0 4px;
  margin-bottom: 4px;
}

.panel-empty { font-size: 11px; color: var(--muted-d); padding: 0 4px; line-height: 1.5; }

.session-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  border-radius: var(--r);
  cursor: pointer;
  transition: background .18s var(--ease), transform .18s var(--ease),
              border-color .18s var(--ease);
  border: 1px solid transparent;
}
.session-item:hover {
  background: var(--surf-hi);
  border-color: var(--bdr);
  transform: translateX(2px);
}

.session-q { font-size: 11px; color: var(--muted); line-height: 1.35; }
.session-ts { font-family: var(--mono); font-size: 9px; color: var(--muted-d); }

/* ── Main ───────────────────────────────────────────────────────── */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}

/* Topbar */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
  padding: 0 24px;
  border-bottom: 1px solid var(--bdr);
  background: rgba(11,9,6,.88);
  backdrop-filter: blur(14px);
  flex-shrink: 0;
  position: relative;
}
.topbar.running::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, var(--acc), transparent);
  animation: scan 2.8s ease-in-out infinite;
}
@keyframes scan {
  0%   { clip-path: inset(0 100% 0 0); }
  45%  { clip-path: inset(0 0 0 0); }
  55%  { clip-path: inset(0 0 0 0); }
  100% { clip-path: inset(0 0 0 100%); }
}

.topbar-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: .04em;
  color: var(--text);
}

.topbar-right { display: flex; align-items: center; gap: 10px; }

.version-chip {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 2px 7px;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  transition: color .3s;
}
.status-badge.active { color: var(--acc); }

.status-pip {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--muted-d);
  transition: all .3s;
}
.status-badge.active .status-pip {
  background: var(--acc);
  box-shadow: 0 0 7px var(--acc);
  animation: pip 1.4s ease-in-out infinite;
}
@keyframes pip { 0%,100%{opacity:1}50%{opacity:.2} }

/* Content scroll area */
.content {
  flex: 1;
  overflow-y: auto;
  padding: 28px 32px 20px;
  display: flex;
  flex-direction: column;
}

/* Error bar */
.error-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: rgba(194,94,94,.09);
  border: 1px solid rgba(194,94,94,.28);
  border-radius: var(--r);
  color: #E88080;
  font-size: 13px;
  margin-bottom: 16px;
  flex-shrink: 0;
}
.error-close { background:none; border:none; color:#E88080; cursor:pointer; font-size:18px; line-height:1; }

/* Welcome */
.welcome {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding-bottom: 60px;
}

.welcome-orb {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: radial-gradient(circle at 38% 35%, #EDE9DF 0%, #D4A843 40%, #8B6914 100%);
  box-shadow: 0 0 40px rgba(212,168,67,.25), 0 0 0 0 rgba(212,168,67,.15);
  margin-bottom: 24px;
  animation: orb-float 4s ease-in-out infinite;
}
@keyframes orb-float {
  0%,100% { transform: translateY(0);    box-shadow: 0 0 40px rgba(212,168,67,.25), 0 8px 28px rgba(0,0,0,.35); }
  50%      { transform: translateY(-9px); box-shadow: 0 0 56px rgba(212,168,67,.38), 0 18px 36px rgba(0,0,0,.28); }
}

.welcome-title {
  font-family: var(--prose);
  font-size: 2.2rem;
  font-weight: 600;
  letter-spacing: -.03em;
  color: var(--text);
  margin-bottom: 8px;
}

.welcome-sub {
  font-size: 15px;
  color: var(--muted);
  margin-bottom: 32px;
}

.suggestions {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
  max-width: 600px;
}

.suggestion-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 20px;
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  color: var(--muted);
  font-size: 13.5px;
  font-weight: 450;
  text-align: left;
  cursor: pointer;
  transition: all .22s var(--ease);
  position: relative;
  overflow: hidden;
}
.suggestion-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(212,168,67,.04) 0%, transparent 60%);
  opacity: 0;
  transition: opacity .22s var(--ease);
}
.suggestion-card:hover {
  border-color: rgba(212,168,67,.45);
  color: var(--text);
  background: var(--surf-hi);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,.35), 0 0 0 1px rgba(212,168,67,.12);
}
.suggestion-card:hover::before { opacity: 1; }
.suggestion-card:active { transform: translateY(0); }
.sug-icon { font-size: 20px; flex-shrink: 0; transition: transform .2s var(--ease); }
.suggestion-card:hover .sug-icon { transform: scale(1.15); }
.sug-text { line-height: 1.45; }

/* Results area */
.results {
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex: 1;
}

/* Stats strip */
.stats-strip {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  gap: 0;
}
.stat { display:flex; flex-direction:column; gap:3px; }
.stat-val { font-family:var(--mono); font-size:17px; font-weight:500; color:var(--acc); line-height:1; }
.stat-key { font-family:var(--mono); font-size:8.5px; letter-spacing:.14em; color:var(--muted-d); }
.stat-div { width:1px; height:26px; background:var(--bdr); margin:0 20px; flex-shrink:0; }

/* Research brief card */
.brief-card {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  overflow: hidden;
  transition: border-color .25s var(--ease), box-shadow .25s var(--ease);
}
.brief-card:not(.processing):hover {
  border-color: var(--bdr-hi);
  box-shadow: 0 4px 24px rgba(0,0,0,.3);
}

.brief-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  border-bottom: 1px solid var(--bdr);
}

.brief-label { font-family:var(--mono); font-size:9.5px; font-weight:500; letter-spacing:.14em; color:var(--muted); }

.brief-actions { display:flex; align-items:center; gap:8px; }

.binary-badge {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .08em;
  color: var(--acc);
  border: 1px solid rgba(212,168,67,.35);
  border-radius: 3px;
  padding: 2px 7px;
  background: rgba(212,168,67,.06);
}

.act-btn {
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: .1em;
  padding: 4px 10px;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  color: var(--muted);
  cursor: pointer;
  transition: all .2s var(--ease);
}
.act-btn:hover {
  border-color: var(--bdr-hi);
  color: var(--text);
  transform: translateY(-1px);
  box-shadow: 0 3px 10px rgba(0,0,0,.25);
}
.act-btn:active { transform: translateY(0); box-shadow: none; }
.act-btn.done { border-color:rgba(91,173,120,.5); color:var(--ok); }
.act-btn.accent { background:rgba(212,168,67,.08); border-color:rgba(212,168,67,.4); color:var(--acc); }
.act-btn.accent:hover {
  background: rgba(212,168,67,.18);
  box-shadow: 0 3px 12px rgba(212,168,67,.18);
}

/* Processing card */
.brief-card.processing { padding: 32px; display:flex; justify-content:center; }
.processing-inner { display:flex; align-items:center; gap:8px; }
.proc-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--acc);
  animation: proc 1.4s var(--ease) infinite;
  box-shadow: 0 0 6px rgba(212,168,67,0);
}
@keyframes proc {
  0%,100% { opacity:.15; transform:scale(.7); box-shadow:0 0 4px rgba(212,168,67,0); }
  50%     { opacity:1;   transform:scale(1);  box-shadow:0 0 10px rgba(212,168,67,.5); }
}
.proc-label { font-family:var(--mono); font-size:11px; color:var(--muted); letter-spacing:.08em; margin-left:4px; }

/* Pipeline log */
.log-card {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  overflow: hidden;
}

.log-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 10px 16px;
  background: none;
  border: none;
  cursor: pointer;
  color: var(--muted);
}
.log-toggle:hover { background: var(--surf-hi); }

.log-badge {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 1px 7px;
  margin-left: 2px;
}

.log-body {
  max-height: 300px;
  overflow-y: auto;
  padding: 8px 0 6px;
}

.tl-list { display:flex; flex-direction:column; padding: 0 12px; gap: 8px; }

.tl-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.tl-node {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 5px;
  background: var(--muted-d);
}

.tl-body { flex:1; }

.tl-badge {
  font-family: var(--mono);
  font-size: 8.5px;
  font-weight: 500;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 6px;
}

.tl-ts { font-family:var(--mono); font-size:9px; color:var(--muted-d); }
.tl-msg { font-size:11px; color:var(--muted); line-height:1.4; margin-top:2px; }

/* Node/badge colors */
.node-supervisor { background:#D4A843; }
.node-coordinator { background:#60A5FA; }
.node-grounding_check { background:#4ADE80; }
.node-critic { background:#FB923C; }
.node-delivery { background:#A3E635; }
.node-pipeline { background:var(--muted-d); }
.node-error { background:#F87171; }
.node-done { background:#34D399; }

.badge-supervisor { background:rgba(212,168,67,.13); color:#D4A843; }
.badge-coordinator { background:rgba(96,165,250,.13); color:#60A5FA; }
.badge-grounding_check { background:rgba(74,222,128,.13); color:#4ADE80; }
.badge-critic { background:rgba(251,146,60,.13); color:#FB923C; }
.badge-delivery { background:rgba(163,230,53,.13); color:#A3E635; }
.badge-pipeline { background:rgba(156,163,175,.1); color:#9CA3AF; }
.badge-error { background:rgba(248,113,113,.13); color:#F87171; }
.badge-done { background:rgba(52,211,153,.13); color:#34D399; }

/* ── Input bar ──────────────────────────────────────────────────── */
.input-bar {
  flex-shrink: 0;
  padding: 12px 24px 16px;
  border-top: 1px solid var(--bdr);
  background: rgba(11,9,6,.92);
  backdrop-filter: blur(14px);
}

.input-wrap { display:flex; flex-direction:column; gap:10px; }

.query-input {
  width: 100%;
  resize: none;
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  color: var(--text);
  font-family: var(--ui);
  font-size: 14px;
  line-height: 1.6;
  padding: 12px 16px;
  outline: none;
  caret-color: var(--acc);
  transition: border-color .15s, box-shadow .15s;
}
.query-input:focus { border-color: var(--acc); box-shadow: 0 0 0 3px rgba(212,168,67,.07); }
.query-input:disabled { opacity:.4; cursor:not-allowed; }
.query-input::placeholder { color:var(--muted-d); }

.input-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

/* Mode strip */
.mode-strip {
  display: flex;
  gap: 3px;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  padding: 3px;
}
.mode-btn {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 5px 10px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  cursor: pointer;
  font-family: var(--ui);
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  transition: background .2s var(--ease), color .2s var(--ease),
              border-color .2s var(--ease), transform .15s var(--ease);
  gap: 1px;
}
.mode-btn:hover:not(:disabled) {
  background: var(--surf-hi);
  color: var(--text);
  transform: translateY(-1px);
}
.mode-btn:active:not(:disabled) { transform: translateY(0); }
.mode-btn.active { background:var(--surf-hi); border-color:var(--bdr-hi); color:var(--acc); }
.mode-btn:disabled { opacity:.3; cursor:not-allowed; }
.mode-cost { font-family:var(--mono); font-size:8.5px; color:var(--muted-d); font-weight:400; }

/* Format pills */
.fmt-row { display:flex; gap:4px; }
.fmt-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 9px;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  color: var(--muted);
  font-family: var(--ui);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: border-color .2s var(--ease), color .2s var(--ease),
              background .2s var(--ease), transform .15s var(--ease),
              box-shadow .2s var(--ease);
}
.fmt-btn:hover:not(:disabled) {
  border-color: var(--bdr-hi);
  color: var(--text);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0,0,0,.2);
}
.fmt-btn:active:not(:disabled) { transform: translateY(0); box-shadow: none; }
.fmt-btn.active { background:rgba(212,168,67,.07); border-color:rgba(212,168,67,.45); color:var(--acc); }
.fmt-btn.active:hover:not(:disabled) { box-shadow: 0 2px 10px rgba(212,168,67,.15); }
.fmt-btn:disabled { opacity:.3; cursor:not-allowed; }
.fmt-glyph { font-family:var(--mono); font-size:10px; }

/* Send button */
.send-btn {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 36px;
  padding: 0 20px;
  background: var(--acc);
  border: none;
  border-radius: var(--r);
  cursor: pointer;
  margin-left: auto;
  transition: background .2s var(--ease), transform .15s var(--ease),
              box-shadow .2s var(--ease), opacity .15s;
  overflow: visible;
}
.send-btn:hover:not(:disabled) {
  background: var(--acc-dim, #B8902F);
  transform: translateY(-1px);
  box-shadow: 0 4px 18px rgba(212,168,67,.35);
}
.send-btn:active:not(:disabled) { transform: translateY(0); box-shadow: none; }
.send-btn:disabled { opacity:.3; cursor:not-allowed; }

.ring {
  position: absolute;
  inset: -5px;
  border: 1.5px solid rgba(212,168,67,.45);
  border-radius: calc(var(--r) + 5px);
  opacity: 0;
  pointer-events: none;
}
.send-btn.running .ring { animation: ring-exp 2s ease-out infinite; }
.send-btn.running .ring-2 { animation-delay: 1s; }
@keyframes ring-exp { 0%{transform:scale(.92);opacity:.75}100%{transform:scale(1.35);opacity:0} }

.send-text {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--ui);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  color: #0B0906;
}

.dots span { animation: dot-blink 1.3s ease-in-out infinite; }
.dots span:nth-child(2) { animation-delay:.22s; }
.dots span:nth-child(3) { animation-delay:.44s; }
@keyframes dot-blink { 0%,100%{opacity:1}50%{opacity:.15} }

.session-tag {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .1em;
  color: var(--muted-d);
  text-align: right;
}

/* ── Prose ──────────────────────────────────────────────────────── */
.prose {
  font-family: var(--prose);
  font-size: 16px;
  line-height: 1.8;
  color: var(--text);
  padding: 24px 28px;
}
.prose :deep(h1) { font-family:var(--prose); font-size:1.55rem; font-weight:600; line-height:1.2; margin:1.4em 0 .5em; letter-spacing:-.02em; color:#F5F0E8; }
.prose :deep(h1:first-child) { margin-top:0; }
.prose :deep(h2) { font-family:var(--prose); font-size:1.15rem; font-weight:600; margin:1.3em 0 .4em; color:var(--acc); }
.prose :deep(h3) { font-family:var(--ui); font-size:.78rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:1.2em 0 .35em; }
.prose :deep(p) { margin-bottom:.9em; }
.prose :deep(strong) { font-weight:600; color:#F5F0E8; }
.prose :deep(ul), .prose :deep(ol) { padding-left:1.5em; margin-bottom:.9em; }
.prose :deep(li) { margin-bottom:.25em; }
.prose :deep(a) { color:var(--acc); text-decoration:underline; text-underline-offset:3px; text-decoration-color:rgba(212,168,67,.35); }
.prose :deep(a:hover) { text-decoration-color:var(--acc); }
.prose :deep(code) { font-family:var(--mono); font-size:.8em; background:var(--bg); border:1px solid var(--bdr); border-radius:3px; padding:1px 5px; color:#C5B99A; }
.prose :deep(blockquote) { border-left:2px solid var(--acc); margin:.7em 0 .9em; padding-left:14px; color:var(--muted); font-style:italic; }
.prose :deep(hr) { border:none; border-top:1px solid var(--bdr); margin:1.8em 0; }
.prose :deep(table) { width:100%; border-collapse:collapse; font-family:var(--ui); font-size:13px; margin-bottom:.9em; }
.prose :deep(th), .prose :deep(td) { border:1px solid var(--bdr); padding:7px 12px; text-align:left; }
.prose :deep(th) { background:var(--bg); font-weight:600; color:var(--muted); font-family:var(--mono); font-size:10px; letter-spacing:.08em; text-transform:uppercase; }

/* ── Transitions ────────────────────────────────────────────────── */
.fade-enter-active, .fade-leave-active { transition: opacity .3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.rise-enter-active { transition: opacity .35s ease, transform .35s ease; }
.rise-enter-from { opacity:0; transform:translateY(14px); }

.slide-down-enter-active, .slide-down-leave-active { transition: opacity .18s, transform .18s; }
.slide-down-enter-from, .slide-down-leave-to { opacity:0; transform:translateY(-8px); }

.event-enter-active { transition: opacity .2s, transform .2s; }
.event-enter-from { opacity:0; transform:translateX(-8px); }

/* ── Responsive ─────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .panel { display: none; }
  .content { padding: 16px 16px 14px; }
  .input-bar { padding: 10px 14px 14px; }
  .input-controls { gap: 6px; }
  .fmt-btn { padding: 5px 6px; font-size: 10px; }
}
</style>
