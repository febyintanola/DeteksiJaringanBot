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
      reason = `Skor: ${det.score.toFixed(3)} | Avg cluster weight: ${det.cluster_avg_weight.toFixed(3)} | Degree: ${det.degree} | Koneksi utama: ${topPeers}`;
    }
    alert(`Akun: ${n.label}\nID: ${n.id}\nCluster: ${cid}\n${canopyLabel}\nStatus: ${suspicious.includes(n.id) ? 'Terduga bot' : 'Tidak terduga'}\nAktivitas: ${activity}\n${reason}`);
  });
}

function renderMetrics(metrics, timings) {
  const canopyMetrics = metrics?.num_canopies
    ? ` | <b>Canopies</b>: ${metrics.num_canopies} (avg ${metrics.avg_canopy_size?.toFixed?.(1) ?? '-'}, max ${metrics.max_canopy_size ?? '-'})`
    : '';
  const canopyEdgeRatio = metrics?.intra_canopy_edge_ratio != null
    ? ` | <b>Intra-canopy edge ratio</b>: ${metrics.intra_canopy_edge_ratio.toFixed(3)}`
    : '';
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
      <b>Timings</b> — Fetch: ${timings?.fetch_sec?.toFixed?.(3) ?? '-'} s, Parse: ${timings?.parse_sec?.toFixed?.(3) ?? '-'} s, Canopy: ${timings?.canopy_sec?.toFixed?.(3) ?? '-'} s, Build: ${timings?.build_sec?.toFixed?.(3) ?? '-'} s, MST: ${timings?.cluster_sec?.toFixed?.(3) ?? '-'} s, Score: ${timings?.score_sec?.toFixed?.(3) ?? '-'} s, Total: ${timings?.total_sec?.toFixed?.(3) ?? '-'} s
    </div>`;
}

function renderTable(clusters, suspicious, nodes, suspiciousDetails, canopies) {
  const nameMap = Object.fromEntries(nodes.map(n => [n.id, n.label]));
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  const detailOf = Object.fromEntries((suspiciousDetails || []).map(d => [d.user_id, d]));
  const parts = [];
  Object.entries(clusters).forEach(([cid, members]) => {
    const sus = members.filter(m => suspicious.includes(m));
    const non = members.filter(m => !suspicious.includes(m));
    const susList = sus.map(u => {
      const det = detailOf[u];
      const nodeInfo = nodeMap[u] || {};
      if (!det) {
        return `<li>${nameMap[u] || u} — komentar ${nodeInfo.comment_count ?? 0}, reply ${nodeInfo.reply_count ?? 0}, canopy ${nodeInfo.canopy ?? '-'}</li>`;
      }
      const topPeers = (det.top_edges || []).map(e => `${e.peer_label} (${e.weight.toFixed(2)})`).join(', ') || '-';
      return `<li><b>${nameMap[u] || u}</b> — skor ${det.score.toFixed(3)}, avg w ${det.cluster_avg_weight.toFixed(3)}, degree ${det.degree}, strength ${det.strength.toFixed(3)}, komentar ${nodeInfo.comment_count ?? 0}, reply ${nodeInfo.reply_count ?? 0}, canopy ${nodeInfo.canopy ?? '-'}<br/>Koneksi utama: ${topPeers}</li>`;
    }).join('') || '<li>-</li>';
    parts.push(`
      <div class="card">
        <h3>Cluster ${cid}</h3>
        <div><b>Terduga bot</b> (${sus.length}):
          <ul>${susList}</ul>
        </div>
        <div><b>Tidak terduga</b> (${non.length}): ${non.map(u => nameMap[u]).join(', ') || '-'}</div>
      </div>
    `);
  });
  const canopyParts = Object.entries(canopies || {}).map(([canopyId, members]) => {
    return `<li><b>${canopyId}</b> — ${members.length} akun</li>`;
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
    max_comments: Number(document.getElementById('max_comments').value),
    alpha_mention: Number(document.getElementById('alpha').value),
    beta_reply: Number(document.getElementById('beta').value),
    gamma_content: Number(document.getElementById('gamma').value),
    delta_thread: Number(document.getElementById('delta').value) || 0,
    k_neighbors: Number(document.getElementById('k').value),
    canopy_t1: Number(document.getElementById('t1').value),
    canopy_t2: Number(document.getElementById('t2').value),
    use_ann: true,
    use_mst_overlay: document.getElementById('mst').checked,
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

// Tour edukatif MST & Canopy
const tourBtn = document.getElementById('tour');
if (tourBtn) {
  tourBtn.addEventListener('click', () => {
    const driver = window.driver.js.driver({
      showProgress: true,
      steps: [
        { element: '#url', popover: { title: 'Input URL', description: 'Masukkan URL video TikTok yang ingin dianalisis.' } },
        { element: '#alpha', popover: { title: 'α (mention)', description: 'Bobot pengaruh relasi mention antar akun.' } },
        { element: '#beta', popover: { title: 'β (reply)', description: 'Bobot interaksi balas-membalas untuk edge reply.' } },
        { element: '#gamma', popover: { title: 'γ (konten)', description: 'Bobot kesamaan konten (v1: token overlap, nanti TF‑IDF/embedding).' } },
        { element: '#delta', popover: { title: 'δ (thread)', description: 'Bobot co-thread untuk akun yang aktif di thread yang sama.' } },
        { element: '#t1', popover: { title: 'Canopy T1/T2', description: 'Pre-clustering untuk mempercepat—T1 longgar, T2 ketat (implementasi bertahap).' } },
        { element: '#graph', popover: { title: 'Graf & MST', description: 'Graf akun dengan cluster berwarna. MST overlay menyorot backbone (minimum spanning tree).' } },
        { element: '#table', popover: { title: 'Ringkasan Hasil', description: 'Daftar akun terduga bot per cluster berdasarkan skor komposit.' } },
      ]
    });
    driver.drive();
  });
}
