<script setup>
import { ref, computed, nextTick } from 'vue'
import { marked } from 'marked'

// ── Config ───────────────────────────────────────────────────────────────
const MODES = [
  { value: 'fast',     label: 'Fast',     desc: '3 queries · ~$0.07' },
  { value: 'research', label: 'Research', desc: '5 queries · ~$0.30'  },
]

const FORMATS = [
  { value: 'markdown', label: 'Markdown', glyph: '#'  },
  { value: 'docx',     label: 'Word',     glyph: 'W'  },
  { value: 'pdf',      label: 'PDF',      glyph: '⬡'  },
  { value: 'pptx',     label: 'Slides',   glyph: '▣'  },
]

// ── State ────────────────────────────────────────────────────────────────
const query      = ref('')
const mode       = ref('fast')
const format     = ref('markdown')
const isRunning  = ref(false)
const sessionId  = ref(null)
const events     = ref([])
const result     = ref(null)
const error      = ref(null)
const copied     = ref(false)
const startTime  = ref(null)
const timelineEl = ref(null)
const stats      = ref({ cost: null, tokens: null, elapsed: null })

// ── Derived ──────────────────────────────────────────────────────────────
const hasContent     = computed(() => events.value.length > 0)
const renderedResult = computed(() =>
  result.value ? marked.parse(result.value) : ''
)

// ── Helpers ──────────────────────────────────────────────────────────────
function nowTS() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

function scrollTimeline() {
  nextTick(() => {
    if (timelineEl.value) {
      timelineEl.value.scrollTop = timelineEl.value.scrollHeight
    }
  })
}

// ── Research pipeline ────────────────────────────────────────────────────
async function startResearch() {
  if (!query.value.trim() || isRunning.value) return

  events.value   = []
  result.value   = null
  error.value    = null
  copied.value   = false
  stats.value    = { cost: null, tokens: null, elapsed: null }
  isRunning.value = true
  startTime.value = Date.now()

  let res
  try {
    res = await fetch('/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query:  query.value,
        mode:   mode.value,
        format: format.value,
      }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  } catch (err) {
    error.value    = err.message
    isRunning.value = false
    return
  }

  const { session_id } = await res.json()
  sessionId.value = session_id

  const es = new EventSource(`/stream/${session_id}`)

  es.addEventListener('status', e => {
    const d = JSON.parse(e.data)
    events.value.push({ phase: d.phase || 'pipeline', message: d.message, ts: nowTS() })
    scrollTimeline()
  })

  es.addEventListener('result', e => {
    const d = JSON.parse(e.data)
    result.value = d.final_output || ''
    stats.value  = {
      cost:    d.cost_usd ?? 0,
      tokens:  d.tokens   ?? 0,
      elapsed: ((Date.now() - startTime.value) / 1000).toFixed(1),
    }
  })

  es.addEventListener('error', e => {
    if (e.data) {
      try        { error.value = JSON.parse(e.data).message }
      catch (_)  { error.value = e.data }
    }
  })

  es.addEventListener('done', () => {
    es.close()
    isRunning.value = false
    events.value.push({ phase: 'done', message: 'Pipeline complete', ts: nowTS() })
    if (!stats.value.elapsed) {
      stats.value.elapsed = ((Date.now() - startTime.value) / 1000).toFixed(1)
    }
    scrollTimeline()
  })

  es.onerror = () => {
    es.close()
    isRunning.value = false
  }
}

// ── Actions ──────────────────────────────────────────────────────────────
function copyResult() {
  if (!result.value) return
  navigator.clipboard.writeText(result.value)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}

function downloadMarkdown() {
  if (!result.value) return
  const blob = new Blob([result.value], { type: 'text/markdown' })
  const url  = URL.createObjectURL(blob)
  const a    = Object.assign(document.createElement('a'), {
    href: url, download: 'research-brief.md',
  })
  a.click()
  URL.revokeObjectURL(url)
}

function downloadBinary() {
  if (!result.value?.startsWith('data:')) return
  const ext = { docx: 'docx', pdf: 'pdf', pptx: 'pptx' }[format.value] ?? format.value
  const a = Object.assign(document.createElement('a'), {
    href: result.value,
    download: `research-brief.${ext}`,
  })
  a.click()
}
</script>

<template>
  <div class="app" :class="{ running: isRunning }">

    <!-- Atmospheric dot grid -->
    <div class="bg-dots" aria-hidden="true" />

    <!-- Radial glow while running -->
    <Transition name="fade-slow">
      <div class="bg-glow" aria-hidden="true" v-if="isRunning" />
    </Transition>

    <!-- ── Header ──────────────────────────────────────────────────── -->
    <header class="header">
      <div class="brand">
        <svg class="brand-hex" width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M11 1.5L20.5 6.75V17.25L11 22.5L1.5 17.25V6.75L11 1.5Z"
                stroke="#D4A843" stroke-width="1.1"
                fill="rgba(212,168,67,0.05)" />
          <path d="M11 5.5L17.5 9.25V16.75L11 20.5L4.5 16.75V9.25L11 5.5Z"
                fill="rgba(212,168,67,0.07)" />
        </svg>
        <span class="brand-name">NOFRIN</span>
        <span class="brand-dot">·</span>
        <span class="brand-sub">Deep Research Agent</span>
      </div>

      <div class="header-right">
        <span class="version-chip">v2.0</span>
        <div class="status-badge" :class="{ active: isRunning }">
          <span class="status-pip" />
          {{ isRunning ? 'ACTIVE' : 'READY' }}
        </div>
      </div>
    </header>

    <!-- ── Main ────────────────────────────────────────────────────── -->
    <main class="main">

      <!-- Query card -->
      <section class="query-card" :class="{ compact: hasContent }">
        <label class="field-label" for="query-input">RESEARCH QUERY</label>
        <textarea
          id="query-input"
          v-model="query"
          class="query-input"
          :rows="hasContent ? 2 : 4"
          placeholder="What do you want to understand deeply?  ⌃↵ to begin"
          :disabled="isRunning"
          @keydown.ctrl.enter="startResearch"
        />

        <div class="controls-row">
          <!-- Mode -->
          <div class="ctl-block">
            <span class="field-label">MODE</span>
            <div class="mode-strip">
              <button
                v-for="m in MODES" :key="m.value"
                class="mode-btn"
                :class="{ active: mode === m.value }"
                :disabled="isRunning"
                @click="mode = m.value"
              >
                <span class="mode-name">{{ m.label }}</span>
                <span class="mode-cost">{{ m.desc }}</span>
              </button>
            </div>
          </div>

          <!-- Format -->
          <div class="ctl-block">
            <span class="field-label">OUTPUT FORMAT</span>
            <div class="format-row">
              <button
                v-for="f in FORMATS" :key="f.value"
                class="fmt-btn"
                :class="{ active: format === f.value }"
                :disabled="isRunning"
                @click="format = f.value"
              >
                <span class="fmt-glyph">{{ f.glyph }}</span>
                {{ f.label }}
              </button>
            </div>
          </div>

          <!-- Submit -->
          <div class="ctl-block submit-block">
            <button
              class="submit-btn"
              :class="{ running: isRunning }"
              :disabled="isRunning || !query.trim()"
              @click="startResearch"
            >
              <span class="ring ring-1" />
              <span class="ring ring-2" />
              <span class="submit-text">
                <template v-if="!isRunning">INITIATE RESEARCH</template>
                <template v-else>
                  RESEARCHING
                  <span class="dots">
                    <span>.</span><span>.</span><span>.</span>
                  </span>
                </template>
              </span>
            </button>
            <div class="session-tag" v-if="sessionId">
              SID·{{ sessionId.slice(0, 8).toUpperCase() }}
            </div>
          </div>
        </div>
      </section>

      <!-- Error -->
      <Transition name="slide-down">
        <div class="error-bar" v-if="error">
          <span class="error-sym">⚠</span>
          {{ error }}
          <button class="error-close" @click="error = null">×</button>
        </div>
      </Transition>

      <!-- Content — timeline + result -->
      <Transition name="rise">
        <div class="content-grid" v-if="hasContent">

          <!-- ── Timeline ──────────────────────────────────────── -->
          <aside class="tl-panel">
            <div class="panel-head">
              <span class="panel-label">PIPELINE LOG</span>
              <span class="event-badge">{{ events.length }}</span>
            </div>

            <div class="tl-scroll" ref="timelineEl">
              <TransitionGroup name="event" tag="div" class="tl-list">
                <div
                  v-for="(ev, i) in events"
                  :key="i"
                  class="tl-row"
                >
                  <div class="tl-gutter">
                    <div class="tl-node" :class="'node-' + ev.phase" />
                    <div
                      class="tl-rail"
                      v-if="i < events.length - 1 || isRunning"
                    />
                  </div>
                  <div class="tl-body">
                    <div class="tl-meta">
                      <span class="tl-badge" :class="'badge-' + ev.phase">
                        {{ ev.phase }}
                      </span>
                      <span class="tl-ts">{{ ev.ts }}</span>
                    </div>
                    <p class="tl-msg">{{ ev.message }}</p>
                  </div>
                </div>
              </TransitionGroup>

              <!-- Live pulse row -->
              <div class="tl-row tl-live" v-if="isRunning">
                <div class="tl-gutter">
                  <div class="tl-node node-live" />
                </div>
                <div class="tl-body">
                  <span class="live-text">processing</span>
                </div>
              </div>
            </div>
          </aside>

          <!-- ── Result ────────────────────────────────────────── -->
          <section class="result-panel" v-if="result || stats.cost !== null">

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

            <template v-if="result">
              <!-- Toolbar -->
              <div class="result-bar">
                <span class="panel-label">RESEARCH BRIEF</span>
                <div class="result-actions">
                  <button
                    class="action-btn"
                    :class="{ done: copied }"
                    @click="copyResult"
                  >
                    {{ copied ? '✓ COPIED' : 'COPY' }}
                  </button>
                  <button
                    class="action-btn"
                    v-if="format === 'markdown'"
                    @click="downloadMarkdown"
                  >
                    ↓ SAVE .MD
                  </button>
                  <button
                    class="action-btn"
                    v-if="format !== 'markdown' && result?.startsWith('data:')"
                    @click="downloadBinary"
                  >
                    ↓ SAVE .{{ format.toUpperCase() }}
                  </button>
                </div>
              </div>

              <!-- Prose body -->
              <div class="result-body prose" v-html="renderedResult" />
            </template>

          </section>

        </div>
      </Transition>

    </main>
  </div>
</template>

<style scoped>
/* ── Variables ────────────────────────────────────────────────────────── */
.app {
  --bg:         #0A0907;
  --surf:       #141310;
  --surf-hi:    #1E1C17;
  --bdr:        #2C2A24;
  --bdr-hi:     #3C3A32;
  --acc:        #D4A843;
  --acc-dim:    #B8902F;
  --acc-glow:   rgba(212, 168, 67, 0.12);
  --text:       #EDE9DF;
  --muted:      #8A8578;
  --muted-d:    #55524C;
  --ok:         #5BAD78;
  --err:        #C25E5E;
  --r:          6px;
  --r-lg:       10px;
  --ui:         'Syne', system-ui, sans-serif;
  --mono:       'JetBrains Mono', monospace;
  --prose:      'Newsreader', Georgia, serif;

  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: var(--ui);
  position: relative;
  overflow-x: hidden;
}

/* ── Background ─────────────────────────────────────────────────────── */
.bg-dots {
  position: fixed;
  inset: 0;
  background-image: radial-gradient(circle, var(--bdr) 1px, transparent 1px);
  background-size: 28px 28px;
  opacity: 0.35;
  pointer-events: none;
  z-index: 0;
}

.bg-glow {
  position: fixed;
  top: 40%;
  left: 50%;
  width: 700px;
  height: 700px;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle, rgba(212,168,67,0.035) 0%, transparent 68%);
  pointer-events: none;
  z-index: 0;
  animation: glow-breathe 4s ease-in-out infinite;
}

@keyframes glow-breathe {
  0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1);    }
  50%       { opacity: 1;   transform: translate(-50%, -50%) scale(1.18); }
}

/* ── Header ──────────────────────────────────────────────────────────── */
.header {
  position: relative;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 52px;
  padding: 0 28px;
  border-bottom: 1px solid var(--bdr);
  background: rgba(10,9,7,0.88);
  backdrop-filter: blur(14px);
}

/* Animated amber scan line when running */
.app.running .header::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent 0%, var(--acc) 50%, transparent 100%);
  animation: header-scan 2.8s ease-in-out infinite;
  transform-origin: left center;
}

@keyframes header-scan {
  0%   { clip-path: inset(0 100% 0 0); }
  45%  { clip-path: inset(0 0% 0 0);   }
  55%  { clip-path: inset(0 0% 0 0);   }
  100% { clip-path: inset(0 0% 0 100%); }
}

.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  user-select: none;
}

.brand-hex { flex-shrink: 0; }

.brand-name {
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.18em;
  color: var(--text);
}

.brand-dot { color: var(--muted-d); }

.brand-sub {
  font-size: 11px;
  font-weight: 400;
  color: var(--muted);
  letter-spacing: 0.04em;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.version-chip {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 2px 7px;
  letter-spacing: 0.08em;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.12em;
  color: var(--muted);
  transition: color 0.3s;
}

.status-badge.active { color: var(--acc); }

.status-pip {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--muted-d);
  transition: all 0.3s;
  flex-shrink: 0;
}

.status-badge.active .status-pip {
  background: var(--acc);
  box-shadow: 0 0 7px var(--acc);
  animation: pip-pulse 1.4s ease-in-out infinite;
}

@keyframes pip-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.25; }
}

/* ── Main layout ─────────────────────────────────────────────────────── */
.main {
  position: relative;
  z-index: 1;
  max-width: 1360px;
  margin: 0 auto;
  padding: 28px 28px 80px;
}

/* ── Query card ──────────────────────────────────────────────────────── */
.query-card {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  padding: 22px 24px;
  margin-bottom: 16px;
  transition: padding 0.25s ease;
}

.query-card.compact { padding: 14px 24px; }

.field-label {
  display: block;
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.14em;
  color: var(--muted);
  margin-bottom: 8px;
  user-select: none;
}

.query-input {
  width: 100%;
  resize: none;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  color: var(--text);
  font-family: var(--ui);
  font-size: 15px;
  line-height: 1.65;
  padding: 11px 14px;
  outline: none;
  caret-color: var(--acc);
  transition: border-color 0.15s, box-shadow 0.15s;
}

.query-input:focus {
  border-color: var(--acc);
  box-shadow: 0 0 0 3px rgba(212, 168, 67, 0.07);
}

.query-input:disabled   { opacity: 0.45; cursor: not-allowed; }
.query-input::placeholder { color: var(--muted-d); }

/* ── Controls ────────────────────────────────────────────────────────── */
.controls-row {
  display: flex;
  gap: 20px;
  margin-top: 14px;
  flex-wrap: wrap;
  align-items: flex-end;
}

.ctl-block {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.submit-block { margin-left: auto; }

/* ── Mode strip ──────────────────────────────────────────────────────── */
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
  gap: 2px;
  padding: 6px 12px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}

.mode-btn:hover:not(:disabled) { background: var(--surf-hi); }

.mode-btn.active {
  background: var(--surf-hi);
  border-color: var(--bdr-hi);
}

.mode-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.mode-name {
  font-family: var(--ui);
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.03em;
}

.mode-btn.active .mode-name { color: var(--acc); }

.mode-cost {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
  letter-spacing: 0.04em;
}

/* ── Format pills ────────────────────────────────────────────────────── */
.format-row {
  display: flex;
  gap: 4px;
}

.fmt-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  color: var(--muted);
  font-family: var(--ui);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  cursor: pointer;
  transition: all 0.15s;
}

.fmt-btn:hover:not(:disabled) {
  border-color: var(--bdr-hi);
  color: var(--text);
}

.fmt-btn.active {
  background: rgba(212, 168, 67, 0.07);
  border-color: rgba(212, 168, 67, 0.5);
  color: var(--acc);
}

.fmt-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.fmt-glyph {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  line-height: 1;
}

/* ── Submit button ───────────────────────────────────────────────────── */
.submit-btn {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 40px;
  padding: 0 26px;
  background: var(--acc);
  border: none;
  border-radius: var(--r);
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
  overflow: visible;
}

.submit-btn:hover:not(:disabled) { background: var(--acc-dim); }
.submit-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.submit-btn.running  { background: var(--acc-dim); }

.ring {
  position: absolute;
  inset: -5px;
  border: 1.5px solid rgba(212, 168, 67, 0.45);
  border-radius: calc(var(--r) + 5px);
  opacity: 0;
  pointer-events: none;
}

.submit-btn.running .ring {
  animation: ring-expand 2s ease-out infinite;
}

.submit-btn.running .ring-2 {
  animation-delay: 1s;
}

@keyframes ring-expand {
  0%   { transform: scale(0.92); opacity: 0.75; }
  100% { transform: scale(1.35); opacity: 0;    }
}

.submit-text {
  position: relative;
  z-index: 1;
  font-family: var(--ui);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: #0A0907;
  display: flex;
  align-items: center;
  gap: 4px;
}

/* Animated dots */
.dots span {
  animation: dot-blink 1.3s ease-in-out infinite;
}
.dots span:nth-child(2) { animation-delay: 0.22s; }
.dots span:nth-child(3) { animation-delay: 0.44s; }

@keyframes dot-blink {
  0%, 100% { opacity: 1;    }
  50%       { opacity: 0.15; }
}

.session-tag {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  color: var(--muted-d);
  text-align: center;
  margin-top: 5px;
}

/* ── Error bar ───────────────────────────────────────────────────────── */
.error-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: rgba(194, 94, 94, 0.09);
  border: 1px solid rgba(194, 94, 94, 0.28);
  border-radius: var(--r);
  color: #E88080;
  font-size: 13px;
  margin-bottom: 16px;
}

.error-sym  { font-size: 14px; flex-shrink: 0; }

.error-close {
  margin-left: auto;
  background: none;
  border: none;
  color: #E88080;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  padding: 0 2px;
  opacity: 0.7;
  transition: opacity 0.15s;
}
.error-close:hover { opacity: 1; }

/* ── Content grid ────────────────────────────────────────────────────── */
.content-grid {
  display: grid;
  grid-template-columns: 290px 1fr;
  gap: 16px;
  align-items: start;
}

/* ── Timeline panel ──────────────────────────────────────────────────── */
.tl-panel {
  position: sticky;
  top: 20px;
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  overflow: hidden;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 14px;
  border-bottom: 1px solid var(--bdr);
}

.panel-label {
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.14em;
  color: var(--muted);
}

.event-badge {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 1px 8px;
  min-width: 24px;
  text-align: center;
}

.tl-scroll {
  max-height: min(480px, calc(100vh - 220px));
  overflow-y: auto;
  padding: 10px 0 6px;
}

.tl-list {
  display: flex;
  flex-direction: column;
}

.tl-row {
  display: flex;
  padding: 3px 0;
}

.tl-gutter {
  width: 38px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 5px;
}

.tl-node {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--muted-d);
}

.tl-rail {
  width: 1px;
  flex: 1;
  min-height: 10px;
  background: var(--bdr);
  margin-top: 4px;
}

.tl-body {
  flex: 1;
  padding: 0 12px 10px 0;
  min-width: 0;
}

.tl-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 3px;
}

.tl-badge {
  font-family: var(--mono);
  font-size: 8.5px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 1px 6px;
  border-radius: 3px;
  white-space: nowrap;
}

.tl-ts {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted-d);
}

.tl-msg {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.45;
  overflow-wrap: break-word;
  word-break: break-word;
}

/* Live row */
.tl-live .tl-node { animation: live-glow 1.1s ease-in-out infinite; }

@keyframes live-glow {
  0%, 100% { opacity: 1;   box-shadow: 0 0 0 0   rgba(212,168,67,0.5); }
  50%       { opacity: 0.6; box-shadow: 0 0 0 4px rgba(212,168,67,0);   }
}

.live-text {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--acc);
  letter-spacing: 0.08em;
}

/* Node colors */
.node-supervisor      { background: #D4A843; }
.node-coordinator     { background: #60A5FA; }
.node-grounding_check { background: #4ADE80; }
.node-critic          { background: #FB923C; }
.node-delivery        { background: #A3E635; }
.node-pipeline        { background: var(--muted-d); }
.node-error           { background: #F87171; }
.node-done            { background: #34D399; }
.node-live            { background: var(--acc); }

/* Badge colors */
.badge-supervisor      { background: rgba(212,168,67,0.13); color: #D4A843; }
.badge-coordinator     { background: rgba(96,165,250,0.13); color: #60A5FA; }
.badge-grounding_check { background: rgba(74,222,128,0.13); color: #4ADE80; }
.badge-critic          { background: rgba(251,146,60,0.13); color: #FB923C; }
.badge-delivery        { background: rgba(163,230,53,0.13); color: #A3E635; }
.badge-pipeline        { background: rgba(156,163,175,0.1); color: #9CA3AF; }
.badge-error           { background: rgba(248,113,113,0.13); color: #F87171; }
.badge-done            { background: rgba(52,211,153,0.13);  color: #34D399; }

/* ── Result panel ────────────────────────────────────────────────────── */
.result-panel {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: var(--r-lg);
  overflow: hidden;
}

/* Stats strip */
.stats-strip {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid var(--bdr);
  background: rgba(10,9,7,0.5);
  gap: 0;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.stat-val {
  font-family: var(--mono);
  font-size: 17px;
  font-weight: 500;
  color: var(--acc);
  letter-spacing: 0.01em;
  line-height: 1;
}

.stat-key {
  font-family: var(--mono);
  font-size: 8.5px;
  letter-spacing: 0.14em;
  color: var(--muted-d);
}

.stat-div {
  width: 1px;
  height: 26px;
  background: var(--bdr);
  margin: 0 20px;
  flex-shrink: 0;
}

/* Result toolbar */
.result-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 20px;
  border-bottom: 1px solid var(--bdr);
}

.result-actions {
  display: flex;
  gap: 6px;
}

.action-btn {
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.1em;
  padding: 4px 10px;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  color: var(--muted);
  cursor: pointer;
  transition: all 0.15s;
}

.action-btn:hover { border-color: var(--bdr-hi); color: var(--text); }

.action-btn.done {
  border-color: rgba(91,173,120,0.5);
  color: var(--ok);
}

/* Research brief prose */
.result-body {
  padding: 28px 32px;
  overflow-y: auto;
  max-height: calc(100vh - 240px);
}

.prose {
  font-family: var(--prose);
  font-size: 16.5px;
  line-height: 1.82;
  color: var(--text);
}

.prose :deep(h1) {
  font-family: var(--prose);
  font-size: 1.65rem;
  font-weight: 600;
  line-height: 1.22;
  margin: 1.5em 0 0.55em;
  letter-spacing: -0.025em;
  color: #F5F0E8;
}

.prose :deep(h1:first-child) { margin-top: 0; }

.prose :deep(h2) {
  font-family: var(--prose);
  font-size: 1.2rem;
  font-weight: 600;
  margin: 1.4em 0 0.45em;
  color: var(--acc);
  letter-spacing: -0.01em;
}

.prose :deep(h3) {
  font-family: var(--ui);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.3em 0 0.4em;
}

.prose :deep(p)              { margin-bottom: 0.95em; }
.prose :deep(strong)         { font-weight: 600; color: #F5F0E8; }
.prose :deep(em)             { font-style: italic; color: #D0CBB8; }

.prose :deep(ul),
.prose :deep(ol)             { padding-left: 1.5em; margin-bottom: 0.95em; }
.prose :deep(li)             { margin-bottom: 0.3em; }

.prose :deep(a) {
  color: var(--acc);
  text-decoration: underline;
  text-underline-offset: 3px;
  text-decoration-color: rgba(212,168,67,0.35);
  transition: text-decoration-color 0.15s;
}
.prose :deep(a:hover) { text-decoration-color: var(--acc); }

.prose :deep(code) {
  font-family: var(--mono);
  font-size: 0.8em;
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: 3px;
  padding: 1px 5px;
  color: #C5B99A;
}

.prose :deep(pre) {
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: var(--r);
  padding: 16px;
  overflow-x: auto;
  margin-bottom: 1em;
}

.prose :deep(pre code) {
  background: none;
  border: none;
  padding: 0;
  font-size: 0.84em;
  color: var(--text);
}

.prose :deep(blockquote) {
  border-left: 2px solid var(--acc);
  margin: 0.75em 0 1em;
  padding-left: 16px;
  color: var(--muted);
  font-style: italic;
}

.prose :deep(hr) {
  border: none;
  border-top: 1px solid var(--bdr);
  margin: 2em 0;
}

.prose :deep(table) {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--ui);
  font-size: 13px;
  margin-bottom: 1em;
}

.prose :deep(th),
.prose :deep(td) {
  border: 1px solid var(--bdr);
  padding: 7px 12px;
  text-align: left;
}

.prose :deep(th) {
  background: var(--bg);
  font-weight: 600;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Transitions ─────────────────────────────────────────────────────── */

/* Timeline events */
.event-enter-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}
.event-enter-from {
  opacity: 0;
  transform: translateX(-10px);
}

/* Content grid (rise on first appear) */
.rise-enter-active {
  transition: opacity 0.38s ease, transform 0.38s ease;
}
.rise-enter-from {
  opacity: 0;
  transform: translateY(14px);
}

/* Error bar */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

/* Slow fade for bg glow */
.fade-slow-enter-active,
.fade-slow-leave-active {
  transition: opacity 1s ease;
}
.fade-slow-enter-from,
.fade-slow-leave-to {
  opacity: 0;
}

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width: 820px) {
  .main            { padding: 18px 16px 60px; }
  .header          { padding: 0 16px; }
  .brand-sub       { display: none; }
  .content-grid    { grid-template-columns: 1fr; }
  .tl-panel        { position: static; }
  .controls-row    { flex-direction: column; gap: 14px; }
  .submit-block    { margin-left: 0; }
  .result-body     { padding: 20px 18px; max-height: none; }
}
</style>
