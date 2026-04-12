const form = document.getElementById('run-form');
const graphDiv = document.getElementById('graph');
const metricsDiv = document.getElementById('metrics');
const tableDiv = document.getElementById('table');

function renderGraph(nodes, edges, clusters, suspicious, suspiciousDetails) {
  const clusterOf = {};
  Object.entries(clusters).forEach(([cid, members]) => {
    members.forEach(u => clusterOf[u] = cid);
  });
  const detailOf = Object.fromEntries((suspiciousDetails || []).map(d => [d.user_id, d]));

  const cy = cytoscape({
    container: graphDiv,
    style: [
      { selector: 'node', style: {
          'background-color': ele => suspicious.includes(ele.id()) ? '#e74c3c' : '#3498db',
          'label': 'data(label)',
          'font-size': 8,
        }
      },
      { selector: 'edge', style: {
          'width': ele => Math.max(1, Math.min(6, ele.data('weight') * 8)),
          'line-color': '#aaa'
        }
      }
    ],
    elements: {
      nodes: nodes.map(n => ({ data: n })),
      edges: edges.map(e => ({ data: { id: `${e.source}-${e.target}`, ...e } }))
    },
    layout: { name: 'cose', animate: false }
  });

  cy.on('tap', 'node', (evt) => {
    const n = evt.target.data();
    const cid = clusterOf[n.id];
    const det = detailOf[n.id];
    let reason = 'Tidak terduga.';
    const activity = `Komentar: ${n.comment_count ?? 0} | Reply: ${n.reply_count ?? 0} | Mention: ${n.mention_count ?? 0} | Thread: ${n.thread_count ?? 0}`;
    const canopyLabel = n.canopy ? `Canopy: ${n.canopy}` : 'Canopy: -';
    if (det) {
      const topPeers = (det.top_edges || []).map(e => `${e.peer_label} (${e.weight.toFixed(3)})`).join(', ') || '-';
      const reasons = (det.reasons || []).join(', ') || 'indikasi koordinasi cluster';
      reason = `Skor: ${det.score.toFixed(3)} | Kepadatan: ${det.cluster_density?.toFixed?.(3) ?? '-'} | Repetisi konten: ${det.content_repetition?.toFixed?.(3) ?? '-'} | Burst: ${det.temporal_burst?.toFixed?.(3) ?? '-'} | Frekuensi tinggi: ${det.high_frequency?.toFixed?.(3) ?? '-'} | Alasan utama: ${reasons} | Koneksi utama: ${topPeers}`;
    }
    const snippet = n.text_sample ? `\nContoh komentar:\n\"${(n.text_sample || '').substring(0, 240)}\"` : '';
    alert(`Akun: ${n.label}\nID: ${n.id}\nCluster: ${cid}\n${canopyLabel}\nStatus: ${suspicious.includes(n.id) ? 'Terduga bot' : 'Tidak terduga'}\nAktivitas: ${activity}\n${reason}${snippet}`);
  });
}

function renderMetrics(metrics, timings) {
  const canopyMetrics = metrics?.num_canopies
    ? ` | <b>Canopies</b>: ${metrics.num_canopies} (avg ${metrics.avg_canopy_size?.toFixed?.(1) ?? '-'}, max ${metrics.max_canopy_size ?? '-'})`
    : '';
  const canopyEdgeRatio = metrics?.intra_canopy_edge_ratio != null
    ? ` | <b>Intra-canopy edge ratio</b>: ${metrics.intra_canopy_edge_ratio.toFixed(3)}`
    : '';
  const scoreSummary = `
      <b>Suspicious clusters</b>: ${metrics.num_suspicious_clusters ?? '-'} |
      <b>Suspicious users</b>: ${metrics.num_suspicious_users ?? '-'} |
      <b>Cutoff</b>: ${metrics.suspicious_score_cutoff?.toFixed?.(3) ?? '-'}
      <br/>
      <b>Avg cluster density</b>: ${metrics.avg_cluster_density?.toFixed?.(3) ?? '-'} |
      <b>Avg content repetition</b>: ${metrics.avg_content_repetition?.toFixed?.(3) ?? '-'} |
      <b>Avg temporal burst</b>: ${metrics.avg_temporal_burst?.toFixed?.(3) ?? '-'} |
      <b>Avg high frequency</b>: ${metrics.avg_high_frequency?.toFixed?.(3) ?? '-'} |
      <b>Account-age coverage</b>: ${metrics.account_age_coverage?.toFixed?.(3) ?? '-'}
  `;
  metricsDiv.innerHTML = `
    <div class="card">
      <b>Modularity</b>: ${metrics.modularity?.toFixed(4) ?? '-'} |
      <b>Conductance (mean)</b>: ${metrics.conductance_mean?.toFixed(4) ?? '-'}
      <br/>
      <b>Nodes</b>: ${metrics.num_nodes ?? '-'} | <b>Edges</b>: ${metrics.num_edges ?? '-'} | <b>Clusters</b>: ${metrics.num_clusters ?? '-'} | <b>Components</b>: ${metrics.num_components ?? '-'}${canopyMetrics}${canopyEdgeRatio}
      <br/>
      <b>Avg degree</b>: ${metrics.avg_degree?.toFixed(2) ?? '-'} | <b>Density</b>: ${metrics.density?.toFixed(4) ?? '-'}
    </div>
    <div class="card">
      ${scoreSummary}
    </div>
    <div class="card">
      <b>Timings</b> - Fetch: ${timings?.fetch_sec?.toFixed?.(3) ?? '-'} s, Parse: ${timings?.parse_sec?.toFixed?.(3) ?? '-'} s, Canopy: ${timings?.canopy_sec?.toFixed?.(3) ?? '-'} s, Build: ${timings?.build_sec?.toFixed?.(3) ?? '-'} s, MST: ${timings?.cluster_sec?.toFixed?.(3) ?? '-'} s, Score: ${timings?.score_sec?.toFixed?.(3) ?? '-'} s, Total: ${timings?.total_sec?.toFixed?.(3) ?? '-'} s
    </div>`;
}

function renderTable(clusters, suspicious, nodes, suspiciousDetails, canopies) {
  const nameMap = Object.fromEntries(nodes.map(n => [n.id, n.label]));
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const detailOf = Object.fromEntries((suspiciousDetails || []).map(d => [d.user_id, d]));

  const allMembers = Object.values(clusters).flat();
  const nonGlobal = allMembers.filter(u => !suspicious.includes(u));
  const nonUnique = Array.from(new Set(nonGlobal));

  const parts = [];

  parts.push(`
    <div class="card">
      <h3>Tidak Terduga (Semua Cluster)</h3>
      <div>${nonUnique.length} akun</div>
      <div>${nonUnique.map(u => nameMap[u] || u).slice(0, 50).join(', ') || '-'}</div>
      ${nonUnique.length > 50 ? `<div>+${nonUnique.length - 50} lainnya</div>` : ''}
    </div>
  `);

  Object.entries(clusters).forEach(([cid, members]) => {
    const sus = members.filter(m => suspicious.includes(m));
    if (sus.length === 0) return;
    const susList = sus.map(u => {
      const det = detailOf[u];
      const nodeInfo = nodeMap[u] || {};
      const topPeers = det && (det.top_edges || []).map(e => `${e.peer_label} (${e.weight.toFixed(2)})`).join(', ') || '-';
      const reasons = det && (det.reasons || []).join(', ') || 'indikasi koordinasi tinggi';
      const alasan = det
        ? `Skor tinggi (${det.score.toFixed(3)}), kepadatan cluster ${det.cluster_density.toFixed(3)}, repetisi konten ${det.content_repetition.toFixed(3)}, temporal burst ${det.temporal_burst.toFixed(3)}, frekuensi tinggi ${det.high_frequency.toFixed(3)}, alasan utama: ${reasons}, koneksi utama: ${topPeers}`
        : 'Pola koneksi internal tinggi.';
      const snippet = nodeInfo.text_sample ? nodeInfo.text_sample.replace(/</g, '&lt;') : '';
      return `
        <li>
          <b>${nameMap[u] || u}</b>
          <div>Alasan: ${alasan}</div>
          ${snippet ? `<div>Kutipan komentar: "${snippet}"</div>` : ''}
        </li>`;
    }).join('') || '<li>-</li>';

    parts.push(`
      <div class="card">
        <h3>Cluster ${cid} - Terduga (${sus.length})</h3>
        <ul>${susList}</ul>
      </div>
    `);
  });

  const canopyParts = Object.entries(canopies || {}).map(([canopyId, members]) => {
    return `<li><b>${canopyId}</b> - ${members.length} akun</li>`;
  }).join('');
  const canopySection = canopyParts
    ? `<div class="card"><h3>Ringkasan Canopy</h3><ul>${canopyParts}</ul></div>`
    : '';
  tableDiv.innerHTML = parts.join('\n') + canopySection;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  metricsDiv.textContent = 'Memproses...';
  tableDiv.innerHTML = '';
  graphDiv.innerHTML = '';

  const payload = {
    url: document.getElementById('url').value,
    max_comments: Number(document.getElementById('max_comments')?.value || 200),
  };

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderMetrics(data.metrics || {}, data.timings || {});
    renderGraph(data.nodes, data.edges, data.clusters, data.suspicious, data.suspicious_details || []);
    renderTable(data.clusters, data.suspicious, data.nodes, data.suspicious_details || [], data.canopies || {});
  } catch (err) {
    metricsDiv.innerHTML = `<div class="card error">Error: ${err.message}</div>`;
  }
});

window.addEventListener('DOMContentLoaded', () => {
  const helper = document.querySelector('.helper');
  if (helper) {
    helper.innerHTML = 'Tempel URL video TikTok di atas. Setelah Analisis, panel kiri menampilkan graf interaksi, dan panel kanan merangkum cluster serta akun terduga.';
  }
});
