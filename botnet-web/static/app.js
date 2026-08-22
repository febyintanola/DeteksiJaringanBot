const el = {
  form: document.getElementById('run-form'),
  url: document.getElementById('url'),
  maxComments: document.getElementById('max_comments'),
  analysisMode: document.getElementById('analysis_mode'),
  submit: document.getElementById('submit-button'),
  reset: document.getElementById('reset-button'),
  metrics: document.getElementById('metrics'),
  graph: document.getElementById('graph'),
  graphLegend: document.getElementById('graph-legend'),
  clusterFilter: document.getElementById('cluster-filter'),
  suspiciousOnly: document.getElementById('suspicious-only'),
  clusterTable: document.getElementById('cluster-table'),
  clusterDetail: document.getElementById('cluster-detail'),
  accountTable: document.getElementById('account-table'),
  processMetrics: document.getElementById('process-metrics'),
  processSummary: document.getElementById('process-summary'),
  status: document.getElementById('status-banner'),
  exportCsvBtn: document.getElementById('export-csv-button'),
  historyBtn: document.getElementById('history-toggle'),
  historyDrawer: document.getElementById('history-drawer'),
  historyList: document.getElementById('history-list'),
  toggleClusters: document.getElementById('toggle-clusters'),
  toggleAccounts: document.getElementById('toggle-accounts'),
};

const HISTORY_KEY = 'botnet-web-analysis-history';
const COLORS = ['#ff6b6b', '#2f7af6', '#22b573', '#f7a529', '#8b5cf6', '#34b5e5', '#ef5da8', '#14b8a6'];
const STAGES = [
  { key: 'fetch', label: 'Ambil komentar', color: '#22b573', start: 0, end: 0.18 },
  { key: 'parse', label: 'Parsing komentar', color: '#2f7af6', start: 0.18, end: 0.33 },
  { key: 'canopy', label: 'Canopy clustering', color: '#8b5cf6', start: 0.33, end: 0.48 },
  { key: 'graph', label: 'Build graph', color: '#f7a529', start: 0.48, end: 0.63 },
  { key: 'cluster', label: 'MST (Kruskal)', color: '#ef4444', start: 0.63, end: 0.78 },
  { key: 'score', label: 'Scoring', color: '#173f90', start: 0.78, end: 0.93 },
  { key: 'finalizing', label: 'Finalisasi hasil', color: '#64748b', start: 0.93, end: 1 },
];

const state = {
  cy: null,
  result: null,
  derived: null,
  activeJobId: null,
  pollingTimer: null,
  pollingInFlight: false,
  selectedClusterId: null,
  selectedNodeId: null,
  graphFilter: 'all',
  showSuspiciousOnly: false,
  showAllClusters: false,
  showAllAccounts: false,
  progressStatus: null,
  history: loadHistory(),
};

function escapeHtml(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function num(value, digits = 0) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toLocaleString('id-ID', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function score(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

function pct(value, digits = 0) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return `${num(clamp01(value) * 100, digits)}%`;
}

function sec(value) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  const s = Number(value);
  if (s < 60) return `${num(s, 1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${num(s - (m * 60), 1)}s`;
}

function avg(values = []) {
  const items = values.filter(value => Number.isFinite(Number(value))).map(Number);
  return items.length ? items.reduce((sum, value) => sum + value, 0) / items.length : 0;
}

function quantile(values = [], ratio = 0.5) {
  const items = values.filter(value => Number.isFinite(Number(value))).map(Number).sort((a, b) => a - b);
  if (!items.length) return 0;
  return items[Math.min(items.length - 1, Math.max(0, Math.floor((items.length - 1) * ratio)))];
}

function hexRgb(hex) {
  const value = String(hex || '').replace('#', '');
  const full = value.length === 3 ? value.split('').map(char => char + char).join('') : value.padEnd(6, '0').slice(0, 6);
  const int = Number.parseInt(full, 16);
  return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 };
}

function rgba(hex, alpha) {
  const color = hexRgb(hex);
  return `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
}

function trim(value, max = 160) {
  const text = String(value || '').trim();
  return text.length > max ? `${text.slice(0, max).trim()}...` : text;
}

function initials(value) {
  const text = String(value || '').replace(/^@/, '').split(/[\s._-]+/).filter(Boolean).slice(0, 2).map(part => part[0]?.toUpperCase() || '').join('');
  return text || 'AK';
}

function tokens(value) {
  return new Set(String(value || '').toLowerCase().match(/[a-z0-9_]+/g) || []);
}

function similarity(left, right) {
  const a = tokens(left);
  const b = tokens(right);
  if (!a.size || !b.size) return 0;
  let hits = 0;
  a.forEach(token => { if (b.has(token)) hits += 1; });
  return hits ? hits / new Set([...a, ...b]).size : 0;
}

function riskInfo(value) {
  if (value >= 0.75) return { label: 'Terduga Kuat', className: 'risk-high' };
  if (value >= 0.55) return { label: 'Perlu Tinjau', className: 'risk-medium' };
  return { label: 'Aktivitas Ringan', className: 'risk-low' };
}

const INDICATOR_LABELS = {
  'kepadatan cluster tinggi': 'Kepadatan klaster tinggi',
  'kepadatan klaster tinggi': 'Kepadatan klaster tinggi',
  'repetisi konten tinggi': 'Repetisi konten tinggi',
  'aktivitas temporal sinkron': 'Aktivitas temporal sinkron',
  'frekuensi komentar tinggi': 'Frekuensi komentar tinggi',
  'proporsi akun baru tinggi': 'Proporsi akun baru tinggi',
  'komentar mirip': 'Sampel komentar serupa',
  'waktu sinkron': 'Aktivitas temporal sinkron',
  'interaksi padat': 'Kepadatan klaster tinggi',
};

function normalizeIndicatorLabel(value) {
  const text = String(value || '').trim();
  if (!text || text.toLowerCase() === 'perlu tinjau manual') return '';
  const normalized = text.replace(/cluster/gi, 'klaster');
  const mapped = INDICATOR_LABELS[normalized.toLowerCase()];
  return mapped || `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`;
}

function clusterSuspicionLabel(summary) {
  return Number(summary?.score || 0) >= 0.5 ? 'Mencurigakan' : 'Normal';
}

function formatClusterSuspicionScore(summary) {
  return `${num(summary?.score || 0, 2)} (${clusterSuspicionLabel(summary)})`;
}

function clusterMainIndicators(summary) {
  const indicators = [];
  const addIndicator = value => {
    const label = normalizeIndicatorLabel(value);
    if (label && !indicators.includes(label)) indicators.push(label);
  };

  (summary?.reasons || []).forEach(addIndicator);
  if (Number(summary?.size || 0) >= 10 && Number(summary?.contentSimilarity || 0) >= 0.35) {
    addIndicator('Banyak akun dengan komentar serupa');
  }

  const scoredSignals = [
    { label: 'Repetisi konten tinggi', value: summary?.contentSimilarity },
    { label: 'Kepadatan klaster tinggi', value: summary?.density },
    { label: 'Aktivitas temporal sinkron', value: summary?.temporalSync },
    { label: 'Frekuensi komentar tinggi', value: summary?.highFrequency },
    { label: 'Proporsi akun baru tinggi', value: summary?.accountAge },
  ]
    .filter(item => Number.isFinite(Number(item.value)))
    .sort((left, right) => Number(right.value) - Number(left.value));

  scoredSignals.forEach(item => {
    if (indicators.length < 3 && Number(item.value) >= 0.3) addIndicator(item.label);
  });
  scoredSignals.forEach(item => {
    if (indicators.length < 3) addIndicator(item.label);
  });

  return indicators.slice(0, 3).join('; ') || 'Tidak ada indikator dominan';
}

function weightedClusterScore(signals = {}) {
  const components = [
    { weight: 0.25, value: signals.density },
    { weight: 0.25, value: signals.content },
    { weight: 0.20, value: signals.temporal },
    { weight: 0.15, value: signals.accountAge },
    { weight: 0.15, value: signals.highFrequency },
  ].filter(component => component.value != null && Number.isFinite(Number(component.value)));
  if (!components.length) return 0;
  const totalWeight = components.reduce((sum, component) => sum + component.weight, 0);
  return clamp01(components.reduce((sum, component) => sum + (component.weight * clamp01(component.value)), 0) / totalWeight);
}

function emptyCard(title, detail) {
  return `<article class="empty-card"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(detail)}</p></div></article>`;
}

function getClusterColor(clusterId) {
  return state.derived?.clusterColorMap?.[clusterId] || COLORS[0];
}

function destroyGraph() {
  if (state.cy) {
    state.cy.destroy();
    state.cy = null;
  }
}

function setSubmitDisabled(value) {
  el.submit.disabled = value;
}

function setStatus(type, title, detail, progress = 0) {
  const value = clamp01(progress);
  el.status.className = `status-banner status-${type}`;
  el.status.innerHTML = `<div class="status-topline"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(detail)}</p></div><span class="status-tag">${escapeHtml(pct(value))}</span></div><div class="status-progress"><span style="width:${value * 100}%"></span></div>`;
}

function clearStatus() {
  el.status.className = 'status-banner is-hidden';
  el.status.innerHTML = '';
}

function formatApiError(payload, response) {
  const detail = payload?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) return 'Permintaan tidak valid. Periksa kembali input yang dikirim.';
  return `Permintaan gagal (status ${response.status}).`;
}

function loadHistory() {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function saveHistory() {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
  } catch (error) {
    // Ignore storage issues.
  }
}

function renderHistory() {
  if (!state.history.length) {
    el.historyList.innerHTML = emptyCard('Belum ada riwayat analisis.', 'Hasil yang sudah dijalankan akan disimpan singkat di browser ini.');
    return;
  }

  el.historyList.innerHTML = state.history.map(item => `<article class="history-item"><div class="history-head"><div><strong>${escapeHtml(trim(item.url, 48))}</strong><p>${escapeHtml(new Date(item.createdAt).toLocaleString('id-ID', { dateStyle: 'medium', timeStyle: 'short' }))}</p></div><span class="score-chip">${escapeHtml(sec(item.metrics?.totalSec || 0))}</span></div><div class="history-meta"><span class="table-chip">${escapeHtml(num(item.maxComments))} komentar</span><span class="table-chip">${escapeHtml(num(item.metrics?.numNodes || 0))} node</span><span class="table-chip">${escapeHtml(num(item.metrics?.numClusters || 0))} cluster</span><span class="table-chip">${escapeHtml(num(item.metrics?.numSuspiciousUsers || 0))} suspicious</span></div><button type="button" class="table-action" data-history-id="${escapeHtml(item.id)}">Gunakan Lagi</button></article>`).join('');
}

function pushHistory(payload, result) {
  const entry = { id: `${Date.now()}`, createdAt: new Date().toISOString(), url: payload.url, maxComments: payload.max_comments, mode: payload.analysis_mode || 'pipeline', metrics: { numNodes: result?.metrics?.num_nodes || 0, numClusters: result?.metrics?.num_clusters || 0, numSuspiciousUsers: result?.metrics?.num_suspicious_users || 0, totalSec: result?.timings?.total_sec || 0 } };
  state.history = [entry, ...state.history.filter(item => item.url !== entry.url)].slice(0, 8);
  saveHistory();
  renderHistory();
}

function metricIcon(name) {
  const icons = {
    comments: '<svg viewBox="0 0 24 24"><path d="M21 12a7 7 0 0 1-7 7H7l-4 3V5a2 2 0 0 1 2-2h9a7 7 0 0 1 7 7z"></path></svg>',
    nodes: '<svg viewBox="0 0 24 24"><circle cx="8" cy="8" r="3"></circle><circle cx="17" cy="9" r="3"></circle><circle cx="12" cy="17" r="3"></circle><path d="M10.5 10.2 11.8 14"></path><path d="M14.2 14.1 15.5 11.5"></path></svg>',
    edges: '<svg viewBox="0 0 24 24"><circle cx="6" cy="12" r="2.2"></circle><circle cx="18" cy="6" r="2.2"></circle><circle cx="18" cy="18" r="2.2"></circle><path d="M8 11.3 16 7.1"></path><path d="M8 12.7 16 16.9"></path></svg>',
    clusters: '<svg viewBox="0 0 24 24"><circle cx="7" cy="8" r="2.5"></circle><circle cx="17" cy="8" r="2.5"></circle><circle cx="12" cy="16" r="2.5"></circle><path d="M9 9.5 10.8 13"></path><path d="M15 9.5 13.2 13"></path></svg>',
    shield: '<svg viewBox="0 0 24 24"><path d="M12 21c4.4 0 8-3.6 8-8V6l-8-3-8 3v7c0 4.4 3.6 8 8 8z"></path></svg>',
    signal: '<svg viewBox="0 0 24 24"><path d="M4 16 9 11 13 15 20 8"></path><path d="M20 8v5"></path><path d="M20 8h-5"></path></svg>',
    clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"></circle><path d="M12 8v4l3 2"></path></svg>',
  };
  return icons[name] || icons.clock;
}

function renderMetrics() {
  if (!state.derived) {
    el.metrics.innerHTML = [
      ['comments', '#2f7af6', 'Jumlah Komentar', '-', 'Belum ada analisis yang dijalankan.'],
      ['nodes', '#22b573', 'Jumlah Node', '-', 'Akun unik pada hasil.'],
      ['edges', '#8b5cf6', 'Jumlah Edge', '-', 'Total koneksi antar akun.'],
      ['clusters', '#f7a529', 'Jumlah Cluster', '-', 'Komunitas hasil clustering.'],
      ['shield', '#ef4444', 'Akun Suspicious', '-', 'Akun yang menonjol pada scoring.'],
      ['clock', '#22b573', 'Waktu Proses', '-', 'Durasi total pipeline.'],
    ].map(item => `<article class="metric-card"><div class="metric-icon" style="background:${rgba(item[1], 0.9)}">${metricIcon(item[0])}</div><div><span class="metric-label">${item[2]}</span><strong>${item[3]}</strong><p class="metric-note">${item[4]}</p></div></article>`).join('');
    return;
  }

  const suspiciousRatio = (state.result?.metrics?.num_suspicious_users || 0) / Math.max(1, state.result?.metrics?.num_nodes || 1);
  const items = [
    ['comments', '#2f7af6', 'Jumlah Komentar', num(state.derived.totalComments), 'Komentar publik yang ikut dianalisis.'],
    ['nodes', '#22b573', 'Jumlah Node', num(state.result?.metrics?.num_nodes), 'Akun unik pada graf interaksi.'],
    ['edges', '#8b5cf6', 'Jumlah Edge', num(state.result?.metrics?.num_edges), 'Total koneksi antar akun.'],
    ['clusters', '#f7a529', 'Jumlah Cluster', num(state.result?.metrics?.num_clusters), 'Komunitas hasil MST clustering.'],
    ['shield', '#ef4444', 'Akun Suspicious', num(state.result?.metrics?.num_suspicious_users), `${pct(suspiciousRatio, 1)} dari total akun`],
    ['clock', '#22b573', 'Waktu Proses', sec(state.result?.timings?.total_sec), 'Durasi eksekusi pipeline.'],
  ];
  el.metrics.innerHTML = items.map(item => `<article class="metric-card"><div class="metric-icon" style="background:${rgba(item[1], 0.9)}">${metricIcon(item[0])}</div><div><span class="metric-label">${item[2]}</span><strong>${item[3]}</strong><p class="metric-note">${item[4]}</p></div></article>`).join('');
}

function deriveData(result) {
  const nodes = result?.nodes || [];
  const edges = result?.edges || [];
  const clusters = result?.clusters || {};
  const details = result?.suspicious_details || [];
  const clusterStats = result?.cluster_stats || {};
  const suspiciousSet = new Set(result?.suspicious || []);
  const nodeMap = Object.fromEntries(nodes.map(node => [node.id, node]));
  const clusterMap = {};
  Object.entries(clusters).forEach(([clusterId, members]) => members.forEach(id => { clusterMap[id] = clusterId; }));
  const detailMap = Object.fromEntries(details.map(detail => [detail.user_id, detail]));
  const nodeScoreMap = {};
  nodes.forEach(node => { nodeScoreMap[node.id] = Number(node.suspicious_score || 0); });
  Object.entries(clusters).forEach(([clusterId, members]) => {
    const clusterScore = Number(clusterStats?.[clusterId]?.score || 0);
    members.forEach(id => { if (!nodeScoreMap[id]) nodeScoreMap[id] = clusterScore; });
  });
  details.forEach(detail => { nodeScoreMap[detail.user_id] = Number(detail.score || nodeScoreMap[detail.user_id] || 0); });
  const detailsByCluster = {};
  details.forEach(detail => {
    if (!detailsByCluster[detail.cluster_id]) detailsByCluster[detail.cluster_id] = [];
    detailsByCluster[detail.cluster_id].push(detail);
  });
  const allTimes = nodes.flatMap(node => [Number(node.first_timestamp || 0), Number(node.last_timestamp || 0)]).filter(Boolean);
  const globalSpan = allTimes.length >= 2 ? Math.max(1, Math.max(...allTimes) - Math.min(...allTimes)) : 1;
  const edgeRef = Math.max(1, quantile(edges.map(edge => Number(edge.weight || 0)), 0.9));
  const edgeBuckets = {};
  edges.forEach(edge => {
    const clusterId = clusterMap[edge.source];
    if (clusterId && clusterId === clusterMap[edge.target]) {
      if (!edgeBuckets[clusterId]) edgeBuckets[clusterId] = [];
      edgeBuckets[clusterId].push(edge);
    }
  });

  const summaries = Object.entries(clusters).map(([clusterId, members]) => {
    const memberIds = Array.isArray(members) ? members : [];
    const memberNodes = memberIds.map(id => nodeMap[id]).filter(Boolean);
    const bucket = edgeBuckets[clusterId] || [];
    const base = detailsByCluster[clusterId]?.[0];
    const clusterStat = clusterStats[clusterId] || {};
    const clusterSize = Math.max(
      Number(clusterStat.cluster_size || 0),
      Number(base?.cluster_size || 0),
      memberIds.length,
      memberNodes.length,
    );
    const possible = clusterSize > 1 ? (clusterSize * (clusterSize - 1)) / 2 : 1;
    const density = clusterStat.cluster_density ?? base?.cluster_density ?? clamp01((0.65 * (bucket.length / possible)) + (0.35 * clamp01(avg(bucket.map(edge => Number(edge.weight || 0))) / edgeRef)));
    const texts = memberNodes.map(node => node.text_sample).filter(Boolean).slice(0, 8);
    const textPairs = [];
    for (let index = 0; index < texts.length; index += 1) for (let cursor = index + 1; cursor < texts.length; cursor += 1) textPairs.push(similarity(texts[index], texts[cursor]));
    const times = memberNodes.flatMap(node => [Number(node.first_timestamp || 0), Number(node.last_timestamp || 0)]).filter(Boolean);
    const temporal = clusterStat.temporal_burst ?? base?.temporal_burst ?? (times.length >= 2 ? clamp01(1 - ((Math.max(...times) - Math.min(...times)) / globalSpan)) : 0);
    const content = clusterStat.content_repetition ?? base?.content_repetition ?? avg(textPairs);
    const high = clusterStat.high_frequency ?? base?.high_frequency ?? clamp01(avg(memberNodes.map(node => Number(node.comment_count || 0))) / Math.max(1, quantile(nodes.map(node => Number(node.comment_count || 0)), 0.9)));
    const scoreValue = clusterStat.score ?? base?.score ?? weightedClusterScore({ density, content, temporal, highFrequency: high, accountAge: clusterStat.account_age });
    const reasons = [...new Set([...(clusterStat.reasons || []), ...(((detailsByCluster[clusterId] || []).flatMap(detail => detail.reasons || []) || []).filter(Boolean))])];
    const topAccounts = [...memberNodes].sort((left, right) => (Number(suspiciousSet.has(right.id)) - Number(suspiciousSet.has(left.id))) || (Number(right.comment_count || 0) - Number(left.comment_count || 0))).slice(0, 4);
    const fallbackReasons = ['Komentar mirip', 'Waktu sinkron', 'Interaksi padat'].filter((_, index) => [content, temporal, density][index] >= 0.45).slice(0, 3);
    return { clusterId, size: clusterSize, memberIds, memberSet: new Set(memberIds), memberNodes, suspiciousMembers: memberNodes.filter(node => suspiciousSet.has(node.id)), suspiciousCount: memberNodes.filter(node => suspiciousSet.has(node.id)).length, score: scoreValue, density, contentSimilarity: content, temporalSync: temporal, highFrequency: high, accountAge: clusterStat.account_age, reasons: reasons.length ? reasons : (fallbackReasons.length ? fallbackReasons : ['Perlu tinjau manual']), topAccounts, sampleComments: [...new Set(memberNodes.map(node => trim(node.text_sample, 140)).filter(Boolean))].slice(0, 2), risk: riskInfo(scoreValue), stats: clusterStat };
  }).sort((left, right) => (right.score - left.score) || (right.suspiciousCount - left.suspiciousCount) || (right.size - left.size) || (Number(left.clusterId) - Number(right.clusterId)));

  const clusterColorMap = {};
  summaries.forEach((summary, index) => { clusterColorMap[summary.clusterId] = COLORS[index % COLORS.length]; summary.color = clusterColorMap[summary.clusterId]; });
  const fallback = [];
  summaries.slice(0, 8).forEach(summary => summary.topAccounts.slice(0, 2).forEach(node => fallback.push({ user_id: node.id, username: node.label || node.id, cluster_id: summary.clusterId, score: summary.score, content_repetition: summary.contentSimilarity, temporal_burst: summary.temporalSync, temporal_alignment: summary.stats?.temporal_alignment, high_frequency: summary.highFrequency, reasons: summary.reasons })));
  const seen = new Set();
  const accounts = (details.length ? details : fallback).filter(detail => !seen.has(detail.user_id) && seen.add(detail.user_id)).map(detail => ({ ...detail, username: detail.username || nodeMap[detail.user_id]?.label || detail.user_id, comment_count: Number(nodeMap[detail.user_id]?.comment_count || 0), text_sample: nodeMap[detail.user_id]?.text_sample || '' })).sort((left, right) => Number(right.score || 0) - Number(left.score || 0));
  return { nodeMap, clusterMap, detailMap, suspiciousSet, clusterSummaries: summaries, clusterColorMap, suspiciousAccounts: accounts, clusterStats, nodeScoreMap, totalComments: nodes.reduce((sum, node) => sum + Number(node.comment_count || 0), 0) };
}

function ensureSelections() {
  const summaries = state.derived?.clusterSummaries || [];
  if (!summaries.length) {
    state.selectedClusterId = null;
    state.selectedNodeId = null;
    return;
  }
  if (!state.selectedClusterId || !summaries.some(summary => summary.clusterId === state.selectedClusterId)) state.selectedClusterId = summaries[0].clusterId;
  if (!state.selectedNodeId || !state.derived.nodeMap[state.selectedNodeId]) state.selectedNodeId = summaries.find(summary => summary.clusterId === state.selectedClusterId)?.topAccounts[0]?.id || null;
}

function renderClusterFilter() {
  const summaries = state.derived?.clusterSummaries || [];
  el.clusterFilter.innerHTML = ['<option value="all">Semua Cluster</option>', ...summaries.slice(0, 18).map(summary => `<option value="${escapeHtml(summary.clusterId)}">Cluster ${escapeHtml(summary.clusterId)} (${escapeHtml(num(summary.size))} akun)</option>`)].join('');
  if (state.graphFilter !== 'all' && !summaries.some(summary => summary.clusterId === state.graphFilter)) state.graphFilter = 'all';
  el.clusterFilter.value = state.graphFilter;
}

function renderClusterTable() {
  const summaries = state.derived?.clusterSummaries || [];
  el.toggleClusters.textContent = state.showAllClusters ? 'Ringkas' : 'Lihat Semua';
  if (!summaries.length) {
    el.clusterTable.innerHTML = emptyCard('Belum ada data cluster.', 'Ringkasan cluster akan muncul setelah analisis selesai.');
    return;
  }
  const rows = (state.showAllClusters ? summaries : summaries.slice(0, 6)).map(summary => `<tr class="${summary.clusterId === state.selectedClusterId ? 'is-selected' : ''}" data-cluster-id="${escapeHtml(summary.clusterId)}"><td><span class="cluster-badge" style="background:${summary.color}">C${escapeHtml(summary.clusterId)}</span></td><td>${escapeHtml(num(summary.size))}</td><td><span class="risk-chip ${summary.risk.className}">${escapeHtml(score(summary.score))}</span></td><td>${escapeHtml(score(summary.density))}</td><td>${escapeHtml(score(summary.contentSimilarity))}</td><td>${escapeHtml(score(summary.temporalSync))}</td><td><button type="button" class="table-action" data-cluster-id="${escapeHtml(summary.clusterId)}">Lihat Detail</button></td></tr>`).join('');
  el.clusterTable.innerHTML = `<table class="data-table"><thead><tr><th>Cluster</th><th>Akun</th><th>Suspicious Score</th><th>Density</th><th>Content Similarity</th><th>Temporal Sync</th><th>Aksi</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderClusterDetail() {
  const summaries = state.derived?.clusterSummaries || [];
  if (!summaries.length) {
    el.clusterDetail.innerHTML = emptyCard('Belum ada cluster yang dipilih.', 'Pilih cluster atau jalankan analisis baru untuk melihat detailnya.');
    return;
  }
  const summary = summaries.find(item => item.clusterId === state.selectedClusterId) || summaries[0];
  const nodeId = summary.memberSet.has(state.selectedNodeId) ? state.selectedNodeId : (summary.topAccounts[0]?.id || null);
  const node = nodeId ? state.derived.nodeMap[nodeId] : null;
  const detail = nodeId ? state.derived.detailMap[nodeId] : null;
  const patternItems = [
    ...summary.reasons.map(reason => ({ text: reason, color: summary.color })),
    { text: `${num(summary.suspiciousCount)} akun paling menonjol`, color: '#ef4444' },
    { text: summary.sampleComments.length ? 'Ada sampel komentar serupa' : 'Komentar perlu dicek manual', color: '#22b573' },
  ].slice(0, 4);
  const scoreParts = [
    { label: 'Density', value: summary.density, weight: 0.25, color: summary.color },
    { label: 'Content Similarity', value: summary.contentSimilarity, weight: 0.25, color: '#2f7af6' },
    { label: 'Temporal Burst', value: summary.temporalSync, weight: 0.20, color: '#22b573' },
    { label: 'High Frequency', value: summary.highFrequency, weight: 0.15, color: '#f7a529' },
    { label: 'Akun Baru', value: summary.accountAge, weight: 0.15, color: '#8b5cf6' },
  ].filter(item => item.value != null).map(item => ({ ...item, contribution: clamp01(item.value) * item.weight }));
  const topDriver = scoreParts.reduce((best, item) => (!best || item.contribution > best.contribution ? item : best), null);

  const dominantAccounts = summary.topAccounts.length
    ? summary.topAccounts.map(item => `<button type="button" class="table-action" data-user-id="${escapeHtml(item.id)}" data-cluster-id="${escapeHtml(summary.clusterId)}">${escapeHtml(item.label || item.id)}</button>`).join('')
    : '<div class="comment-card">Belum ada akun dominan yang bisa ditampilkan.</div>';

  const sampleComments = summary.sampleComments.length
    ? summary.sampleComments.map(comment => `<div class="comment-card">${escapeHtml(comment)}</div>`).join('')
    : '<div class="comment-card">Belum ada sampel komentar yang cukup untuk cluster ini.</div>';

  el.clusterDetail.innerHTML = `
    <article class="detail-card">
      <div class="detail-topline">
        <div>
          <div class="chip-row">
            <span class="cluster-badge" style="background:${summary.color}">C${escapeHtml(summary.clusterId)}</span>
            <span class="risk-chip ${summary.risk.className}">${escapeHtml(summary.risk.label)}</span>
            ${summary.suspiciousCount ? `<span class="table-chip">${escapeHtml(num(summary.suspiciousCount))} akun ditandai</span>` : ''}
            ${topDriver ? `<span class="signal-chip signal-chip-strong">Driver utama: ${escapeHtml(topDriver.label)}</span>` : ''}
          </div>
          <h3>Cluster ${escapeHtml(summary.clusterId)}</h3>
          <p class="detail-copy">Ringkasan cluster berdasarkan pola koneksi, kemiripan komentar, dan sinkronisasi aktivitas.</p>
        </div>
        <span class="score-chip">${escapeHtml(score(summary.score))}</span>
      </div>

      <div class="detail-grid">
        <div class="detail-stat">
          <span>Jumlah Akun</span>
          <strong>${escapeHtml(num(summary.size))}</strong>
        </div>
        <div class="detail-stat">
          <span>Suspicious Score</span>
          <strong>${escapeHtml(score(summary.score))}</strong>
        </div>
        <div class="detail-stat">
          <span>Density</span>
          <strong>${escapeHtml(score(summary.density))}</strong>
        </div>
        <div class="detail-stat">
          <span>Temporal Sync</span>
          <strong>${escapeHtml(score(summary.temporalSync))}</strong>
        </div>
      </div>

      <section class="detail-section">
        <div class="detail-section-head">
          <h4>Pola Utama</h4>
          ${summary.canopyId ? `<span class="table-chip">Canopy ${escapeHtml(summary.canopyId)}</span>` : ''}
        </div>
        <div class="detail-list">
          ${patternItems.map(item => `<div class="detail-list-item"><span class="detail-list-bullet" style="background:${item.color}"></span><span>${escapeHtml(item.text)}</span></div>`).join('')}
        </div>
      </section>

      <section class="detail-section">
        <div class="detail-section-head">
          <h4>Komponen Score</h4>
          <span class="detail-section-copy">Skala 0 sampai 1, ditimbang sesuai formula scoring</span>
        </div>
        <div class="score-breakdown">
          ${scoreParts.map(item => `
            <div class="score-row ${topDriver?.label === item.label ? 'is-top' : ''}">
              <div class="score-row-head">
                <strong>${escapeHtml(item.label)}</strong>
                <span>${escapeHtml(score(item.value))} • bobot ${escapeHtml(pct(item.weight))}</span>
              </div>
              <div class="score-bar">
                <span style="width:${clamp01(item.value) * 100}%;background:${item.color}"></span>
              </div>
            </div>
          `).join('')}
        </div>
      </section>

      ${node ? `
        <section class="detail-section">
          <div class="detail-section-head">
            <h4>Akun Sorotan</h4>
            <span class="score-chip">${escapeHtml(score(detail?.score ?? summary.score))}</span>
          </div>

          <div class="spotlight-card">
            <div class="spotlight-head">
              <div class="spotlight-user">
                <span class="avatar-pill">${escapeHtml(initials(node.label || node.id))}</span>
                <div class="spotlight-user-copy">
                  <strong>${escapeHtml(node.label || node.id)}</strong>
                  <p class="detail-copy">${escapeHtml(node.id)}</p>
                </div>
              </div>
            </div>

            <div class="chip-row">
              <span class="table-chip">${escapeHtml(num(node.comment_count || 0))} komentar</span>
              <span class="table-chip">${escapeHtml(num(node.reply_count || 0))} reply</span>
              <span class="table-chip">${escapeHtml(num(node.mention_count || 0))} mention</span>
              <span class="table-chip">Score ${escapeHtml(score(node.suspicious_score ?? detail?.score ?? summary.score))}</span>
              ${detail?.node_prominence != null ? `<span class="table-chip">Prominence ${escapeHtml(score(detail.node_prominence))}</span>` : ''}
            </div>

            ${node.text_sample ? `<div class="comment-card">${escapeHtml(trim(node.text_sample, 180))}</div>` : '<div class="comment-card">Belum ada contoh komentar dari akun sorotan ini.</div>'}
          </div>
        </section>
      ` : ''}

      <section class="detail-section">
        <h4>Akun Dominan</h4>
        <div class="chip-row">${dominantAccounts}</div>
      </section>

      <section class="detail-section">
        <div class="detail-section-head">
          <h4>Contoh Komentar</h4>
          <span class="detail-section-copy">${escapeHtml(num(summary.sampleComments.length))} sampel</span>
        </div>
        <div class="comment-stack">${sampleComments}</div>
      </section>

      <div class="detail-button-row">
        <button type="button" class="table-action" data-focus-cluster="${escapeHtml(summary.clusterId)}">Fokus ke Subgraf</button>
        <button type="button" class="table-action" data-clear-filter>Reset Filter Graf</button>
      </div>
    </article>
  `;
}

function renderAccountTable() {
  const accounts = state.derived?.suspiciousAccounts || [];
  el.toggleAccounts.textContent = state.showAllAccounts ? 'Ringkas' : 'Lihat Semua Akun';
  if (!accounts.length) {
    el.accountTable.innerHTML = emptyCard('Belum ada akun yang menonjol.', 'Daftar akun suspicious akan diisi dari hasil scoring setelah analisis selesai.');
    return;
  }
  const rows = (state.showAllAccounts ? accounts : accounts.slice(0, 5)).map(account => `<tr class="${account.user_id === state.selectedNodeId ? 'is-selected' : ''}"><td><button type="button" class="spotlight-user" data-user-id="${escapeHtml(account.user_id)}" data-cluster-id="${escapeHtml(account.cluster_id)}" style="background:none;padding:0;border:0;text-align:left;"><span class="avatar-pill">${escapeHtml(initials(account.username || account.user_id))}</span><span>${escapeHtml(account.username || account.user_id)}</span></button></td><td><span class="cluster-badge" style="background:${getClusterColor(account.cluster_id)}">C${escapeHtml(account.cluster_id)}</span></td><td>${escapeHtml(num(account.comment_count || 0))}</td><td>${escapeHtml(score(account.content_repetition || 0))}</td><td>${escapeHtml(score(account.temporal_burst || 0))}</td><td><span class="risk-chip ${riskInfo(Number(account.score || 0)).className}">${escapeHtml(score(account.score || 0))}</span></td><td><div class="chip-row">${(account.reasons || []).slice(0, 3).map(reason => `<span class="signal-chip">${escapeHtml(reason)}</span>`).join('')}</div></td></tr>`).join('');
  el.accountTable.innerHTML = `<table class="data-table"><thead><tr><th>Username</th><th>Cluster</th><th>Frekuensi Komentar</th><th>Content Similarity</th><th>Temporal Score</th><th>Suspicious Score</th><th>Indikator</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderGraphLegend(items = []) {
  if (!items.length) {
    el.graphLegend.innerHTML = '';
    return;
  }
  el.graphLegend.innerHTML = [
    ...items.slice(0, 8).map(summary => `<span class="legend-pill"><span class="legend-dot" style="background:${summary.color}"></span><span>Cluster ${escapeHtml(summary.clusterId)}</span></span>`),
    '<span class="legend-pill legend-note"><span class="legend-dot" style="background:#fff;border:2px solid #ef4444"></span><span>Border merah = akun suspicious</span></span>',
    '<span class="legend-pill legend-note"><span class="legend-dot" style="width:14px;height:14px;background:rgba(23,63,144,0.18);border:1px solid rgba(23,63,144,0.38)"></span><span>Ukuran node = suspicious score</span></span>',
  ].join('');
}

function nodeSize(data) {
  const suspiciousScore = clamp01(Number(data.nodeScore || 0));
  return Math.max(18, Math.min(56, 18 + (suspiciousScore * 32)));
}

function buildPositions(nodes, clusterIds) {
  const positions = {};
  const groups = clusterIds.map(clusterId => state.derived.clusterSummaries.find(summary => summary.clusterId === clusterId)).filter(Boolean);
  if (!groups.length) return positions;
  const anchor = clusterIds.includes(state.selectedClusterId) ? state.selectedClusterId : groups[0].clusterId;
  const centers = { [anchor]: { x: 0, y: 0 } };
  const orbit = groups.filter(summary => summary.clusterId !== anchor);
  const orbitRadius = Math.max(260, 180 + orbit.length * 35);
  orbit.forEach((summary, index) => {
    const angle = ((Math.PI * 2) / Math.max(1, orbit.length)) * index - (Math.PI / 2);
    centers[summary.clusterId] = { x: Math.cos(angle) * orbitRadius, y: Math.sin(angle) * orbitRadius };
  });
  groups.forEach(summary => {
    const center = centers[summary.clusterId] || { x: 0, y: 0 };
    const members = nodes.filter(node => state.derived.clusterMap[node.id] === summary.clusterId).sort((left, right) => (Number(state.derived.suspiciousSet.has(right.id)) - Number(state.derived.suspiciousSet.has(left.id))) || ((state.derived.nodeScoreMap[right.id] || 0) - (state.derived.nodeScoreMap[left.id] || 0)) || (Number(right.comment_count || 0) - Number(left.comment_count || 0)));
    if (members.length === 1) {
      positions[members[0].id] = center;
      return;
    }
    members.forEach((node, index) => {
      if (index === 0 && state.derived.suspiciousSet.has(node.id)) {
        positions[node.id] = center;
        return;
      }
      const ring = Math.max(6, Math.ceil(Math.sqrt(members.length) * 2));
      const layer = 1 + Math.floor(index / ring);
      const slot = index % ring;
      const angle = ((Math.PI * 2) / ring) * slot + (layer * 0.42);
      const radius = 34 + (layer * 28) + ((slot % 2) * 8);
      positions[node.id] = { x: center.x + (Math.cos(angle) * radius), y: center.y + (Math.sin(angle) * radius) };
    });
  });
  return positions;
}

function renderGraph() {
  destroyGraph();
  if (!state.result || !state.derived) {
    el.graph.innerHTML = '<div class="placeholder large">Graf akan tampil di sini setelah analisis dijalankan.</div>';
    renderGraphLegend([]);
    return;
  }
  if (typeof window.cytoscape === 'undefined') {
    el.graph.innerHTML = '<div class="placeholder large">Library Cytoscape tidak berhasil dimuat. Coba refresh halaman ini.</div>';
    renderGraphLegend([]);
    return;
  }
  const clusterIds = state.graphFilter === 'all' ? state.derived.clusterSummaries.map(summary => summary.clusterId) : [state.graphFilter];
  const nodes = state.result.nodes.filter(node => clusterIds.includes(state.derived.clusterMap[node.id]) && (!state.showSuspiciousOnly || state.derived.suspiciousSet.has(node.id)));
  if (!nodes.length) {
    el.graph.innerHTML = '<div class="placeholder large">Tidak ada node yang cocok dengan filter saat ini.</div>';
    renderGraphLegend([]);
    return;
  }
  const ids = new Set(nodes.map(node => node.id));
  const edges = state.result.edges.filter(edge => ids.has(edge.source) && ids.has(edge.target));
  const visibleClusterIds = [...new Set(nodes.map(node => state.derived.clusterMap[node.id]))];
  const positions = buildPositions(nodes, visibleClusterIds);
  const groups = visibleClusterIds.map(clusterId => state.derived.clusterSummaries.find(summary => summary.clusterId === clusterId)).filter(Boolean);
  el.graph.innerHTML = '';
  state.cy = window.cytoscape({
    container: el.graph,
    elements: {
      nodes: nodes.map(node => ({ data: { ...node, clusterId: state.derived.clusterMap[node.id], suspicious: state.derived.suspiciousSet.has(node.id), nodeScore: Number(state.derived.nodeScoreMap[node.id] || 0) }, position: positions[node.id] || { x: 0, y: 0 } })),
      edges: edges.map((edge, index) => ({ data: { id: `${edge.source}-${edge.target}-${index}`, ...edge, sourceCluster: state.derived.clusterMap[edge.source], sameCluster: state.derived.clusterMap[edge.source] === state.derived.clusterMap[edge.target] } })),
    },
    style: [{ selector: 'node', style: { 'background-color': ele => getClusterColor(ele.data('clusterId')), 'border-width': ele => ele.data('suspicious') ? 4 : 2, 'border-color': ele => ele.data('suspicious') ? '#ef4444' : rgba(getClusterColor(ele.data('clusterId')), 0.95), 'width': ele => nodeSize(ele.data()), 'height': ele => nodeSize(ele.data()), 'label': ele => (ele.data('suspicious') || Number(ele.data('nodeScore') || 0) >= 0.5 || nodes.length <= 22) ? ele.data('label') : '', 'font-size': 10, 'font-family': 'Trebuchet MS', 'color': '#132849', 'text-valign': 'bottom', 'text-margin-y': 7, 'text-outline-width': 3, 'text-outline-color': '#ffffff', 'text-outline-opacity': 0.95, 'min-zoomed-font-size': 8 } }, { selector: 'node:selected', style: { 'overlay-color': '#173f90', 'overlay-opacity': 0.14, 'overlay-padding': 10 } }, { selector: 'edge', style: { 'curve-style': 'bezier', 'line-color': ele => ele.data('sameCluster') ? rgba(getClusterColor(ele.data('sourceCluster')), 0.24) : 'rgba(104,122,153,0.22)', 'width': ele => Math.max(1.2, Math.min(5, Number(ele.data('weight') || 0) * 9)), 'opacity': 0.85 } }],
    layout: { name: 'preset', fit: true, padding: 55 },
  });
  if (state.selectedNodeId && ids.has(state.selectedNodeId)) state.cy.getElementById(state.selectedNodeId).select();
  state.cy.on('tap', 'node', event => {
    const id = event.target.id();
    state.selectedNodeId = id;
    state.selectedClusterId = state.derived.clusterMap[id] || state.selectedClusterId;
    renderClusterTable();
    renderClusterDetail();
    renderAccountTable();
  });
  state.cy.on('tap', event => {
    if (event.target === state.cy) {
      state.selectedNodeId = null;
      renderClusterDetail();
      renderAccountTable();
    }
  });
  renderGraphLegend(groups);
}

function renderProcessRow(label, color, stateLabel, valueLabel, progressValue) {
  const value = clamp01(progressValue);
  return `<div class="process-row"><div class="process-row-main"><div class="process-name"><span class="process-dot" style="background:${color}"></span><span class="process-label">${escapeHtml(label)}</span></div><strong class="process-state">${escapeHtml(stateLabel)}</strong></div><div class="process-row-metrics"><div class="process-bar"><span style="width:${value * 100}%;background:${color}"></span></div><span class="process-value">${escapeHtml(valueLabel)}</span></div></div>`;
}

function renderProcessIdle() {
  el.processSummary.textContent = 'Siap dianalisis';
  el.processMetrics.innerHTML = STAGES.map(stage => renderProcessRow(stage.label, stage.color, 'Idle', '0%', 0)).join('');
}

function renderProcessProgress(status) {
  const overall = clamp01(status?.progress || 0);
  el.processSummary.textContent = status?.detail || `Berjalan ${pct(overall)}`;
  el.processMetrics.innerHTML = STAGES.map(stage => {
    const value = overall <= stage.start ? 0 : overall >= stage.end ? 1 : clamp01((overall - stage.start) / Math.max(0.001, stage.end - stage.start));
    const label = status?.stage === stage.key ? 'Aktif' : overall >= stage.end ? 'Selesai' : 'Menunggu';
    return renderProcessRow(stage.label, stage.color, label, pct(value), value);
  }).join('');
}

function renderProcessTimings(timings = {}) {
  const rows = [['Ambil komentar', timings.fetch_sec, '#22b573'], ['Parsing komentar', timings.parse_sec, '#2f7af6'], ['Canopy Clustering', timings.canopy_sec, '#8b5cf6'], ['Build Graph', timings.build_sec, '#f7a529'], ['MST (Kruskal)', timings.cluster_sec, '#ef4444'], ['Scoring', timings.score_sec, '#173f90']];
  const total = Math.max(0.001, Number(timings.total_sec || 0));
  el.processSummary.textContent = `Total ${sec(total)}`;
  el.processMetrics.innerHTML = rows.map(row => {
    const ratio = clamp01(Number(row[1] || 0) / total);
    return renderProcessRow(row[0], row[2], sec(row[1] || 0), pct(ratio), ratio);
  }).join('') + renderProcessRow('Total Waktu', '#0f2346', sec(total), '100%', 1);
}

function renderLoadingDashboard() {
  el.metrics.innerHTML = Array.from({ length: 6 }).map(() => '<article class="metric-card skeleton"></article>').join('');
  el.clusterTable.innerHTML = emptyCard('Memuat cluster...', 'Pipeline sedang menyusun ringkasan cluster.');
  el.clusterDetail.innerHTML = emptyCard('Menyiapkan detail cluster...', 'Detail cluster akan muncul otomatis setelah hasil utama siap.');
  el.accountTable.innerHTML = emptyCard('Memuat akun...', 'Sistem sedang menghitung akun yang paling menonjol.');
  el.graph.innerHTML = '<div class="placeholder loading"><div>Analisis sedang diproses...</div></div>';
  el.graphLegend.innerHTML = '';
  renderProcessProgress({ progress: 0.04, stage: 'fetch', detail: 'Menyiapkan pengambilan komentar...' });
}

function renderDashboard() {
  state.derived = state.result ? deriveData(state.result) : null;
  ensureSelections();
  renderMetrics();
  renderClusterFilter();
  renderClusterTable();
  renderClusterDetail();
  renderAccountTable();
  renderGraph();
  if (state.result?.timings) renderProcessTimings(state.result.timings);
  else renderProcessIdle();
  el.exportCsvBtn.disabled = !state.result;
}

function getErrorPresentation(message) {
  const lower = String(message || '').toLowerCase();
  const isTikTokBlocked = (
    lower.includes('anti-bot')
    || lower.includes('respons kosong')
    || lower.includes('terdeteksi atau dibatasi')
  );
  const isCommentUnavailable = (
    lower.includes('daftar komentarnya kosong')
    || lower.includes('tidak mengembalikan komentar publik')
    || lower.includes('tidak memiliki komentar publik')
    || lower.includes('komentar dinonaktifkan')
    || lower.includes('komentar dibatasi')
  );

  if (isTikTokBlocked) {
    return {
      title: 'Akses TikTok dibatasi',
      tableTitle: 'TikTok menolak pengambilan komentar.',
      hint: 'Komentar bisa saja ada, tetapi request dari aplikasi sedang terdeteksi atau dibatasi oleh TikTok. Perbarui `ms_token`, coba mode non-headless/WebKit, ganti jaringan/proxy, atau tunggu beberapa saat.',
    };
  }

  if (isCommentUnavailable) {
    return {
      title: 'Komentar tidak tersedia',
      tableTitle: 'Video belum bisa dianalisis.',
      hint: 'Video ini belum punya komentar publik, komentarnya dinonaktifkan, atau akses komentarnya dibatasi. Coba gunakan video lain yang komentarnya terlihat publik.',
    };
  }

  if (lower.includes('ms_token')) {
    return {
      title: 'Token TikTok perlu diperbarui',
      tableTitle: 'Analisis gagal dijalankan.',
      hint: 'Perbarui variabel lingkungan `ms_token` dengan cookie `msToken` TikTok yang valid, lalu jalankan ulang analisis.',
    };
  }

  if (lower.includes('playwright') || lower.includes('browser') || lower.includes('peramban')) {
    return {
      title: 'Dependensi peramban belum siap',
      tableTitle: 'Analisis gagal dijalankan.',
      hint: 'Pastikan dependensi peramban untuk TikTokApi sudah terpasang dengan benar, lalu coba lagi.',
    };
  }

  return {
    title: 'Terjadi kendala',
    tableTitle: 'Analisis gagal dijalankan.',
    hint: 'Periksa URL video, status komentar publik, dan konfigurasi pipeline sebelum mencoba lagi.',
  };
}

function renderError(message) {
  const info = getErrorPresentation(message);
  destroyGraph();
  renderMetrics();
  el.graph.innerHTML = '<div class="placeholder large">Graf belum bisa ditampilkan untuk analisis ini.</div>';
  el.graphLegend.innerHTML = '';
  el.clusterTable.innerHTML = emptyCard(info.tableTitle, message);
  el.clusterDetail.innerHTML = emptyCard('Detail cluster belum tersedia.', info.hint);
  el.accountTable.innerHTML = emptyCard('Daftar akun belum tersedia.', info.hint);
  renderProcessIdle();
}

function resetDashboard() {
  if (state.pollingTimer) {
    window.clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
  state.activeJobId = null;
  state.progressStatus = null;
  state.result = null;
  state.derived = null;
  state.selectedClusterId = null;
  state.selectedNodeId = null;
  state.graphFilter = 'all';
  state.showSuspiciousOnly = false;
  state.showAllClusters = false;
  state.showAllAccounts = false;
  el.suspiciousOnly.checked = false;
  el.url.value = '';
  el.maxComments.value = '1000';
  el.analysisMode.value = 'pipeline';
  clearStatus();
  setSubmitDisabled(false);
  renderDashboard();
  renderProcessIdle();
  el.exportCsvBtn.disabled = true;
}

async function pollJobStatus() {
  if (!state.activeJobId || state.pollingInFlight) return;
  state.pollingInFlight = true;
  try {
    const response = await fetch(`/api/status/${state.activeJobId}`);
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(formatApiError(errorPayload, response));
    }
    const status = await response.json();
    state.progressStatus = status;
    if (status.status === 'running' || status.status === 'pending') {
      setStatus('loading', 'Analisis sedang berjalan', status.detail || 'Pipeline sedang memproses komentar dan membangun graf.', status.progress || 0);
      renderProcessProgress(status);
      return;
    }
    if (state.pollingTimer) {
      window.clearInterval(state.pollingTimer);
      state.pollingTimer = null;
    }
    if (status.status === 'success' && status.result) {
      state.result = status.result;
      renderDashboard();
      setStatus('success', 'Analisis selesai', 'Pilih cluster atau klik node pada graf untuk melihat detailnya.', 1);
      pushHistory({ url: el.url.value.trim(), max_comments: Number(el.maxComments.value || 1000), analysis_mode: el.analysisMode.value }, status.result);
      setSubmitDisabled(false);
      return;
    }
    throw new Error(status.error || status.detail || 'Pipeline berakhir tanpa hasil yang bisa ditampilkan.');
  } catch (error) {
    if (state.pollingTimer) {
      window.clearInterval(state.pollingTimer);
      state.pollingTimer = null;
    }
    setSubmitDisabled(false);
    renderError(error.message);
    setStatus('error', getErrorPresentation(error.message).title, error.message, 1);
  } finally {
    state.pollingInFlight = false;
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const payload = { url: el.url.value.trim(), max_comments: Number(el.maxComments.value || 1000) };
  state.result = null;
  state.derived = null;
  state.selectedClusterId = null;
  state.selectedNodeId = null;
  state.graphFilter = 'all';
  state.showSuspiciousOnly = false;
  el.suspiciousOnly.checked = false;
  setSubmitDisabled(true);
  el.exportCsvBtn.disabled = true;
  renderLoadingDashboard();
  setStatus('loading', 'Analisis sedang dimulai', 'Aplikasi sedang membuat proses analisis dan menyiapkan pipeline.', 0.02);
  try {
    const response = await fetch('/api/run_async', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(formatApiError(errorPayload, response));
    }
    const data = await response.json();
    state.activeJobId = data.job_id;
    await pollJobStatus();
    state.pollingTimer = window.setInterval(pollJobStatus, 1200);
  } catch (error) {
    setSubmitDisabled(false);
    renderError(error.message);
    setStatus('error', getErrorPresentation(error.message).title, error.message, 1);
  }
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

function csvEscape(value) {
  const text = String(value ?? '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const safeText = /^[=+@]/.test(text) ? `'${text}` : text;
  return /[",\n]|^\s|\s$/.test(safeText) ? `"${safeText.replace(/"/g, '""')}"` : safeText;
}

function csvNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return '';
  return Number(value).toFixed(digits);
}

function buildCsv(rows, columns) {
  const header = columns.map(column => csvEscape(column.label)).join(',');
  const body = rows.map(row => columns.map(column => csvEscape(column.value(row))).join(','));
  return [header, ...body].join('\r\n');
}

function exportCsvResult() {
  if (!state.result || !state.derived) return;
  const exportedAt = new Date().toISOString();
  const baseName = `ringkasan-klaster-bot-tiktok-${exportedAt.replace(/[:.]/g, '-')}`;
  const columns = [
    { label: 'cluster_id', value: summary => summary.clusterId },
    { label: 'jumlah_akun', value: summary => Number(summary.size || 0) },
    { label: 'jumlah_akun_suspicious', value: summary => Number(summary.suspiciousCount || 0) },
    { label: 'suspicious_score', value: summary => csvNumber(summary.score, 4) },
    { label: 'kategori', value: summary => clusterSuspicionLabel(summary) },
    { label: 'density', value: summary => csvNumber(summary.density, 4) },
    { label: 'content_similarity', value: summary => csvNumber(summary.contentSimilarity, 4) },
    { label: 'temporal_sync', value: summary => csvNumber(summary.temporalSync, 4) },
    { label: 'high_frequency', value: summary => csvNumber(summary.highFrequency, 4) },
    { label: 'account_age', value: summary => csvNumber(summary.accountAge, 4) },
    { label: 'indikator_utama', value: summary => clusterMainIndicators(summary) },
    { label: 'akun_dominan', value: summary => (summary.topAccounts || []).map(account => account.label || account.id).join(' | ') },
    { label: 'sampel_komentar', value: summary => (summary.sampleComments || []).join(' | ') },
  ];
  const csv = buildCsv(state.derived.clusterSummaries, columns);

  downloadBlob(`${baseName}.csv`, `\uFEFF${csv}`, 'text/csv;charset=utf-8');
}

function setUrlValidationMessage() {
  if (!el.url.value.trim()) {
    el.url.setCustomValidity('URL video TikTok wajib diisi.');
    return;
  }
  el.url.setCustomValidity('Masukkan URL video TikTok yang valid.');
}

function setMaxCommentsValidationMessage() {
  const min = el.maxComments.min || '10';
  const max = el.maxComments.max || '10000';
  if (el.maxComments.validity.badInput) {
    el.maxComments.setCustomValidity('Jumlah komentar harus berupa angka.');
    return;
  }
  if (el.maxComments.validity.rangeUnderflow) {
    el.maxComments.setCustomValidity(`Jumlah komentar minimal ${min}.`);
    return;
  }
  if (el.maxComments.validity.rangeOverflow) {
    el.maxComments.setCustomValidity(`Jumlah komentar maksimal ${max}.`);
    return;
  }
  el.maxComments.setCustomValidity('Jumlah komentar tidak valid.');
}

function bindValidationMessages() {
  el.url.addEventListener('input', () => { el.url.setCustomValidity(''); });
  el.maxComments.addEventListener('input', () => { el.maxComments.setCustomValidity(''); });
  el.url.addEventListener('invalid', setUrlValidationMessage);
  el.maxComments.addEventListener('invalid', setMaxCommentsValidationMessage);
}

function focusCluster(clusterId, userId = null) {
  state.selectedClusterId = clusterId;
  state.selectedNodeId = userId;
  if (state.derived?.clusterSummaries?.some(summary => summary.clusterId === clusterId)) state.graphFilter = clusterId;
  renderClusterFilter();
  renderClusterTable();
  renderClusterDetail();
  renderAccountTable();
  renderGraph();
}

function handleGraphAction(action) {
  if (!state.cy) return;
  if (action === 'zoom-in') state.cy.zoom({ level: state.cy.zoom() * 1.2, renderedPosition: { x: el.graph.clientWidth / 2, y: el.graph.clientHeight / 2 } });
  if (action === 'zoom-out') state.cy.zoom({ level: state.cy.zoom() / 1.2, renderedPosition: { x: el.graph.clientWidth / 2, y: el.graph.clientHeight / 2 } });
  if (action === 'fit') state.cy.fit(undefined, 50);
  if (action === 'relayout') renderGraph();
}

function bindEvents() {
  bindValidationMessages();
  el.form.addEventListener('submit', handleSubmit);
  el.reset.addEventListener('click', resetDashboard);
  el.exportCsvBtn.addEventListener('click', exportCsvResult);
  el.historyBtn.addEventListener('click', () => { el.historyDrawer.hidden = false; });
  el.toggleClusters.addEventListener('click', () => { state.showAllClusters = !state.showAllClusters; renderClusterTable(); });
  el.toggleAccounts.addEventListener('click', () => { state.showAllAccounts = !state.showAllAccounts; renderAccountTable(); });
  el.clusterFilter.addEventListener('change', event => {
    state.graphFilter = event.target.value;
    if (state.graphFilter !== 'all') state.selectedClusterId = state.graphFilter;
    renderClusterTable();
    renderClusterDetail();
    renderGraph();
  });
  el.suspiciousOnly.addEventListener('change', event => {
    state.showSuspiciousOnly = Boolean(event.target.checked);
    renderGraph();
  });
  document.querySelectorAll('[data-scroll-target]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('is-active'));
      button.classList.add('is-active');
      document.getElementById(button.dataset.scrollTarget)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
  document.addEventListener('click', event => {
    const graphButton = event.target.closest('[data-graph-action]');
    if (graphButton) return void handleGraphAction(graphButton.dataset.graphAction);
    const userButton = event.target.closest('[data-user-id]');
    if (userButton) {
      state.selectedClusterId = userButton.dataset.clusterId || state.selectedClusterId;
      state.selectedNodeId = userButton.dataset.userId;
      if (state.graphFilter !== 'all' && state.graphFilter !== state.selectedClusterId) state.graphFilter = state.selectedClusterId;
      renderClusterFilter();
      renderClusterTable();
      renderClusterDetail();
      renderAccountTable();
      renderGraph();
      return;
    }
    const clusterButton = event.target.closest('[data-cluster-id]');
    if (clusterButton) return void focusCluster(clusterButton.dataset.clusterId);
    const focusButton = event.target.closest('[data-focus-cluster]');
    if (focusButton) return void focusCluster(focusButton.dataset.focusCluster, state.selectedNodeId);
    if (event.target.closest('[data-clear-filter]')) {
      state.graphFilter = 'all';
      renderClusterFilter();
      renderGraph();
      return;
    }
    const historyButton = event.target.closest('[data-history-id]');
    if (historyButton) {
      const item = state.history.find(entry => entry.id === historyButton.dataset.historyId);
      if (item) {
        el.url.value = item.url || '';
        el.maxComments.value = String(item.maxComments || 1000);
        el.analysisMode.value = item.mode || 'pipeline';
        el.historyDrawer.hidden = true;
        document.getElementById('analysis-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      return;
    }
    if (event.target.closest('[data-close-drawer]') && !el.historyDrawer.hidden) el.historyDrawer.hidden = true;
  });
}

function initialize() {
  bindEvents();
  renderHistory();
  renderMetrics();
  renderClusterTable();
  renderClusterDetail();
  renderAccountTable();
  renderGraph();
  renderProcessIdle();
}

initialize();
