<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Social Viral Prediction | GT–TGN–SCN</title>

  <!-- Tailwind / Chart.js / PapaParse -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

  <!-- Cytoscape + fcose layout -->
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.umd.js"></script>

  <!-- (선택) 한글 웹폰트 -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap" rel="stylesheet">

  <style>
    body{ font-family: "Noto Sans KR", system-ui, -apple-system, Segoe UI, Roboto, "Malgun Gothic", sans-serif; }
    .card{
      background: rgba(255,255,255,0.8);
      -webkit-backdrop-filter: blur(8px);
      backdrop-filter: blur(8px);
      border-radius: 1rem;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,.1), 0 4px 6px -4px rgba(0,0,0,.1);
      padding: 1.25rem;
      border: 1px solid #f1f5f9;
    }
    .kpi{ font-size:1.5rem; line-height:2rem; font-weight:600; }
    .kpi-sub{ font-size:.75rem; line-height:1rem; color:#64748b; }
    .pill{
      display:inline-flex; align-items:center; gap:.35rem;
      padding:.125rem .5rem; border-radius:9999px;
      font-size:.75rem; font-weight:500;
      background:#f1f5f9; color:#334155;
    }
    .tbl th,.tbl td{ padding:.5rem .75rem; font-size:.875rem; }
    .tbl thead th{ background:#f8fafc; color:#475569; font-weight:600; }
    .tbl tbody tr:nth-child(odd){ background:#fff; }
    .tbl tbody tr:nth-child(even){ background:rgba(248,250,252,.5); }
    .hidden-init{ display:none; }
    /* 네트워크 info bubble */
    #netInfo{
      position:absolute; right:1rem; top:1rem; z-index:20;
      background:#ffffffcc; border:1px solid #e2e8f0; border-radius:.75rem;
      padding:.5rem .75rem; box-shadow:0 10px 15px -3px rgba(0,0,0,.1);
      font-size:.85rem; line-height:1.1rem; color:#334155;
      max-width:360px;
    }
  </style>
</head>
<body class="bg-slate-100 text-slate-900">
  <!-- Header / Hero -->
  <header class="bg-gradient-to-r from-indigo-600 to-violet-600 text-white">
    <div class="max-w-6xl mx-auto px-5 py-10">
      <h1 class="text-3xl md:text-4xl font-bold tracking-tight">소셜 바이럴 예측 (GT–TGN–SCN)</h1>
      <p class="mt-3 max-w-3xl text-white/90">
        시계열 그래프(TGN) + 전역 관계(Transformer) + 국소 전파(SCN)로 소셜 콘텐츠의 바이럴 여부를 예측합니다.
        아래에서 <span class="underline decoration-white/60">CSV 업로드</span> 시 결과가 표시됩니다.
      </p>
      <div class="mt-5 flex flex-wrap gap-3 items-center">
        <label class="pill cursor-pointer">
          <input id="csvInput" type="file" accept=".csv" class="hidden" />
          CSV 업로드 (results)
        </label>
        <span id="uploadHint" class="text-sm text-white/80">파일을 선택하세요</span>
      </div>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-5 mt-6 pb-24">
    <!-- KPI cards -->
    <section id="kpiSection" class="grid md:grid-cols-3 gap-4 hidden-init">
      <div class="card"><div class="kpi" id="kpi-auc">—</div><div class="kpi-sub">AUC (Validation)</div></div>
      <div class="card"><div class="kpi" id="kpi-acc">—</div><div class="kpi-sub">Accuracy</div></div>
      <div class="card"><div class="kpi"><span id="kpi-prec">—</span> / <span id="kpi-recall">—</span> / <span id="kpi-f1">—</span></div><div class="kpi-sub">Precision / Recall / F1</div></div>
    </section>

    <!-- Dataset Stats -->
    <section id="statSection" class="grid md:grid-cols-3 gap-4 mt-4 hidden-init">
      <div class="card"><div class="kpi" id="kpi-total">—</div><div class="kpi-sub">전체 예측 수</div></div>
      <div class="card"><div class="kpi" id="kpi-actual-rate">—</div><div class="kpi-sub">실제 바이럴 비율</div></div>
      <div class="card"><div class="kpi" id="kpi-pred-rate">—</div><div class="kpi-sub">예측 바이럴 비율</div></div>
    </section>

    <!-- Chart -->
    <section id="chartSection" class="card mt-6 hidden-init">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-lg font-semibold">소스타입별 AUC (상위 10)</h2>
        <span class="pill">CSV 업로드 후 표시</span>
      </div>
      <canvas id="aucChart" height="120"></canvas>
      <p class="text-xs text-slate-500 mt-2">* 단일 클래스 그룹(AUC 불가)은 제외됩니다.</p>
    </section>

    <!-- Source-type Table -->
    <section id="tableSection" class="card mt-6 overflow-x-auto hidden-init">
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-lg font-semibold">소스타입별 성능</h2>
        <span class="pill">Validation 기준</span>
      </div>
      <table class="tbl w-full" id="stTable">
        <thead>
          <tr>
            <th class="text-left">source_type</th>
            <th class="text-right">AUC</th>
            <th class="text-right">Acc</th>
            <th class="text-right">n</th>
            <th class="text-right">PosRate</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>

    <!-- Network Graph -->
    <section id="netSection" class="card mt-6 hidden-init relative">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h2 class="text-lg font-semibold">노드 관계 네트워크</h2>
        <div class="flex flex-wrap items-center gap-2 text-sm">
          <label class="pill">모드
            <select id="netMode" class="bg-transparent outline-none">
              <option value="top5">소스타입별 Top-5</option>
              <option value="all">전체</option>
              <option value="pred1">Predicted=1</option>
              <option value="act1">Actual=1</option>
            </select>
          </label>
          <label class="pill">확률 ≥ <span id="probVal">0.00</span>
            <input id="probMin" type="range" min="0" max="1" step="0.01" value="0" class="ml-2">
          </label>
          <button id="btnFit" class="pill">맞춤보기</button>
          <button id="btnPng" class="pill">PNG 저장</button>
        </div>
      </div>
      <div id="netLegend" class="flex flex-wrap gap-2 mb-2"></div>
      <div id="net" style="height: 600px;"></div>
      <div id="netInfo" class="hidden-init"></div>
    </section>
  </main>

  <script>
    /* ========= Helpers ========= */
    const show = (id) => document.getElementById(id).classList.remove('hidden-init');
    const hide = (id) => document.getElementById(id).classList.add('hidden-init');
    const fmt = (x, d=4) => (x==null || isNaN(x)) ? "NA" : Number(x).toFixed(d);
    const pct = (x) => (x==null || isNaN(x)) ? "NA" : (100*x).toFixed(2)+"%";

    function metricsFromArrays(yTrue, yPred, yScore) {
      const n = yTrue.length;
      let tp=0, fp=0, tn=0, fn=0;
      for (let i=0;i<n;i++){
        if (yTrue[i]===1 && yPred[i]===1) tp++;
        else if (yTrue[i]===0 && yPred[i]===1) fp++;
        else if (yTrue[i]===0 && yPred[i]===0) tn++;
        else if (yTrue[i]===1 && yPred[i]===0) fn++;
      }
      const acc = n>0 ? (tp+tn)/n : 0;
      const prec = (tp+fp)>0 ? tp/(tp+fp) : 0;
      const rec = (tp+fn)>0 ? tp/(tp+fn) : 0;
      const f1 = (prec+rec)>0 ? 2*prec*rec/(prec+rec) : 0;
      const posRate = n>0 ? yTrue.reduce((s,v)=>s+v,0)/n : 0;
      const predRate = n>0 ? yPred.reduce((s,v)=>s+v,0)/n : 0;
      const auc = computeAUC(yTrue, yScore);
      return {acc, prec, rec, f1, auc, posRate, predRate, n};
    }

    function computeAUC(yTrue, yScore) {
      const n = yTrue.length;
      let P = 0, N = 0;
      for (const y of yTrue) y ? P++ : N++;
      if (P===0 || N===0) return null;
      const idx = [...Array(n).keys()].sort((a,b)=> yScore[b]-yScore[a]);
      let tp = 0, fp = 0, prevScore = Infinity;
      const pts = [[0,0]];
      for (let k=0;k<n;k++){
        const i = idx[k];
        if (yScore[i] !== prevScore) {
          pts.push([fp/N, tp/P]);
          prevScore = yScore[i];
        }
        if (yTrue[i]===1) tp++; else fp++;
      }
      pts.push([1,1]);
      let area = 0;
      for (let i=1;i<pts.length;i++){
        const [x1,y1]=pts[i-1], [x2,y2]=pts[i];
        area += (x2-x1)*(y1+y2)/2;
      }
      return area;
    }

    function renderOverall({auc,acc,prec,rec,f1,n,posRate,predRate}) {
      show('kpiSection'); show('statSection');
      document.getElementById('kpi-auc').textContent   = fmt(auc,4);
      document.getElementById('kpi-acc').textContent   = fmt(acc,4);
      document.getElementById('kpi-prec').textContent  = fmt(prec,4);
      document.getElementById('kpi-recall').textContent= fmt(rec,4);
      document.getElementById('kpi-f1').textContent    = fmt(f1,4);
      document.getElementById('kpi-total').textContent = n.toLocaleString();
      document.getElementById('kpi-actual-rate').textContent = pct(posRate);
      document.getElementById('kpi-pred-rate').textContent   = pct(predRate);
    }

    function renderTable(rows) {
      show('tableSection');
      const tbody = document.querySelector('#stTable tbody');
      tbody.innerHTML = '';
      rows.forEach(r=>{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="text-left">${r.st}</td>
          <td class="text-right">${r.auc==null?'NA':fmt(r.auc,4)}</td>
          <td class="text-right">${fmt(r.acc,4)}</td>
          <td class="text-right">${r.n.toLocaleString()}</td>
          <td class="text-right">${pct(r.posRate)}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    let aucChart;
    function renderChart(topRows) {
      show('chartSection');
      const ctx = document.getElementById('aucChart');
      const labels = topRows.map(d=>d.st);
      const data = topRows.map(d=> (d.auc==null?0:d.auc));
      if (aucChart) aucChart.destroy();
      aucChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'AUC', data }] },
        options: {
          responsive: true,
          scales: { y: { beginAtZero: true, max: 1 } },
          plugins: { legend: { display: false } }
        }
      });
    }

    /* ========= Network Graph ========= */
    let cy, parsedRows = [];

    const BASE_COLORS = [
      '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
      '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
      '#4e79a7','#f28e2b','#59a14f','#e15759','#edc949',
      '#af7aa1','#ff9da7','#9c755f','#bab0ab','#76b7b2'
    ];

    function colorMap(types){
      const map = new Map();
      types.forEach((t,i)=>{
        if(i < BASE_COLORS.length) map.set(t, BASE_COLORS[i]);
        else{
          const h = (i*37)%360; map.set(t, `hsl(${h} 70% 50%)`);
        }
      });
      map.set('NA','#94a3b8');
      return map;
    }

    function buildTop5(rows){
      const g = new Map();
      for(const r of rows){
        const key = r.src_source_type || 'NA';
        if(!g.has(key)) g.set(key, []);
        g.get(key).push(r);
      }
      const out = [];
      for(const [k,arr] of g.entries()){
        arr.sort((a,b)=> Number(b.prediction_probability)-Number(a.prediction_probability));
        out.push(...arr.slice(0,5));
      }
      return out;
    }

    function dominantTypeForNodes(rows){
      const bag = new Map();
      const push = (id, st)=>{
        if(!bag.has(id)) bag.set(id, new Map());
        const m = bag.get(id); m.set(st, (m.get(st)||0)+1);
      };
      for(const r of rows){
        const sst = String(r.src_source_type||'NA');
        const dst = String(r.dst_source_type||'NA');
        push(String(r.src_node_id), sst);
        push(String(r.dst_node_id), dst);
      }
      const node2type = new Map();
      for(const [id, m] of bag.entries()){
        let best = 'NA', cnt = -1;
        for(const [st,c] of m.entries()){ if(c>cnt){cnt=c; best=st;} }
        node2type.set(id, best);
      }
      return node2type;
    }
    // 노드ID -> video_id (src/dst 모두 고려, 다수결) 매핑
    function buildNode2Video(rows){
      const bag = new Map(); // node_id -> (video_id -> count)
      const push = (nid, vid) => {
        if(!nid || !vid) return;
        const id = String(nid);
        const v  = String(vid).trim();
        if(!v) return;
        if(!bag.has(id)) bag.set(id, new Map());
        const m = bag.get(id); m.set(v, (m.get(v)||0)+1);
      };
      for (const r of rows){
        push(r.src_node_id, r.src_video_id);
        push(r.dst_node_id, r.dst_video_id);
      }
      const map = new Map();
      for (const [nid, m] of bag.entries()){
        let best=null, cnt=-1;
        for (const [vid, c] of m.entries()){
          if (c > cnt){ cnt = c; best = vid; }
        }
        map.set(nid, best);
      }
      return map;
    }

    function buildElements(rows){
      const mode = document.getElementById('netMode').value;
      const pmin = Number(document.getElementById('probMin').value);

      let filtered = rows.filter(r => Number(r.prediction_probability) >= pmin);
      if(mode==='pred1') filtered = filtered.filter(r => Number(r.predicted_viral)===1);
      if(mode==='act1')  filtered = filtered.filter(r => Number(r.actual_viral)===1);
      if(mode==='top5')  filtered = buildTop5(filtered);

      // 노드 타입 다수결
      const node2type = dominantTypeForNodes(filtered);
      const node2video = buildNode2Video(filtered);

      // 엣지/노드 구성
      const nodes = new Map();
      const edges = [];
      const deg = new Map();
      const pairCount = new Map();

      for(const r of filtered){
        const s = String(r.src_node_id), t = String(r.dst_node_id);
        const sType = node2type.get(s)||'NA';
        const tType = node2type.get(t)||'NA';

        if(!nodes.has(s)) nodes.set(s, { id:s, label:s, st:sType });
        if(!nodes.has(t)) nodes.set(t, { id:t, label:t, st:tType });

        const key = `${s}->${t}`;
        const idx = (pairCount.get(key)||0)+1; pairCount.set(key, idx);

        edges.push({
          data: {
            id: `${key}#${idx}`,
            source: s, target: t,
            prob: Number(r.prediction_probability)||0,
            pred: Number(r.predicted_viral)||0,
            act: Number(r.actual_viral)||0
          }
        });

        deg.set(s, (deg.get(s)||0)+1);
        deg.set(t, (deg.get(t)||0)+1);
      }

      const types = Array.from(new Set(Array.from(nodes.values()).map(n=>n.st)));
      const cmap = colorMap(types);

      const elements = [];
      for(const n of nodes.values()){
      const d   = deg.get(n.id)||0;
      const vid = node2video.get(n.id) || "";
      const url = vid ? `https://www.youtube.com/watch?v=${vid}` : "";

      elements.push({
        data: {
          id: n.id,
          label: n.label,
          st: n.st,
          color: cmap.get(n.st)||'#94a3b8',
          size: 14 + Math.min(24, d*2),
          video: vid,   // ← 추가
          url: url      // ← 추가
        }
      });
    }

      for(const e of edges) elements.push(e);

      // 범례 갱신
      const legend = document.getElementById('netLegend');
      legend.innerHTML = '';
      types.sort().forEach(st=>{
        const chip = document.createElement('span');
        chip.className = 'pill';
        chip.style.background = '#fff';
        chip.style.border = `1px solid ${cmap.get(st)}`;
        chip.style.color = '#334155';
        chip.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:9999px;background:${cmap.get(st)};margin-right:6px"></span>${st}`;
        legend.appendChild(chip);
      });

      return { elements, nodeCount: nodes.size, edgeCount: edges.length };
    }

    function layoutOptions(name){
      return (name === 'fcose')
        ? { name: 'fcose', quality: 'proof', animate: false, randomize: false, nodeRepulsion: 4500 }
        : { name: 'cose',  animate: false, randomize: false };
    }

    function renderNetwork(rows){
      show('netSection');
      document.getElementById('probVal').textContent =
        Number(document.getElementById('probMin').value).toFixed(2);

      const { elements, nodeCount, edgeCount } = buildElements(rows);

      // 빈 상태 안내
      const info = document.getElementById('netInfo');
      if(edgeCount === 0){
        info.innerHTML = '표시할 엣지가 없습니다.<br>모드를 <b>전체</b>로 바꾸거나, 확률 슬라이더를 내려보세요.';
        show('netInfo');
      } else {
        info.innerHTML = `노드 ${nodeCount.toLocaleString()} · 엣지 ${edgeCount.toLocaleString()}`;
        show('netInfo');
      }

      if(cy){ cy.destroy(); }

      // 일단 cy 생성 (레이아웃은 나중에 시도)
      cy = cytoscape({
        container: document.getElementById('net'),
        elements,
        wheelSensitivity: 0.2,
        style: [
          { selector: 'node',
            style: {
              'background-color': 'data(color)',
              'label': 'data(label)',
              'font-size': 8, 'color': '#334155',
              'text-valign': 'center', 'text-halign': 'center',
              'width': 'data(size)', 'height': 'data(size)'
            }},
          { selector: 'edge',
            style: {
              'curve-style': 'bezier',
              'line-color': '#94a3b8',
              'opacity': 0.35,
              'width': 'mapData(prob, 0, 1, 1, 7)',
              'target-arrow-shape': 'none'
            }}
        ],
        layout: { name: 'preset' } // 레이아웃은 아래에서 시도
      });

      // fcose 시도 → 실패하면 cose로 폴백
      let use = 'fcose';
      try{
        const test = cy.layout(layoutOptions('fcose'));
        if (!test || typeof test.run !== 'function') use = 'cose';
      } catch(e){
        use = 'cose';
      }
      try{
        cy.layout(layoutOptions(use)).run();
      } catch(e){
        // 최종 폴백
        if(use !== 'cose'){
          try { cy.layout(layoutOptions('cose')).run(); } catch(_) {}
        }
      }

      // --- 노드 클릭: 정보 + YouTube 링크 열기 버튼 (video_id 텍스트 없음) ---
      cy.on('tap', 'node', (e)=>{
        const n   = e.target;
        const deg = n.connectedEdges().length;
        const st  = n.data('st');

        // 1) 우선 노드 data에 url이 있으면 사용
        let url = n.data('url');

        // 2) url이 없으면 parsedRows에서 해당 node_id의 video_id를 찾아 생성 (fallback)
        if (!url) {
          const nid = String(n.data('id'));
          const rows = window.parsedRows || [];
          let vid = null;
          for (const r of rows) {
            if (String(r.src_node_id) === nid && r.src_video_id) { vid = r.src_video_id; break; }
            if (String(r.dst_node_id) === nid && r.dst_video_id) { vid = r.dst_video_id; break; }
          }
          if (vid) url = `https://www.youtube.com/watch?v=${vid}`;
        }

        // 3) info 박스 출력 (링크 버튼만)
        let html = `<b>노드</b> ${n.data('id')} <span class="pill" style="margin-left:.35rem">${st}</span><br>
                    차수: ${deg}`;
        if (url) {
          html += `&nbsp; <a href="${url}" target="_blank" rel="noopener"
                  class="pill" style="text-decoration:none;">YouTube 열기 ↗</a>`;
        } else {
          html += `<br><em>열 수 있는 링크가 없습니다</em>`;
        }
        info.innerHTML = html;
        show('netInfo');
      });

      // (선택) 더블클릭 시 즉시 새 탭 열기
      cy.on('dblclick', 'node', (e)=>{
        // 위와 동일한 fallback 로직을 재사용
        let url = e.target.data('url');
        if (!url) {
          const nid = String(e.target.data('id'));
          const rows = window.parsedRows || [];
          let vid = null;
          for (const r of rows) {
            if (String(r.src_node_id) === nid && r.src_video_id) { vid = r.src_video_id; break; }
            if (String(r.dst_node_id) === nid && r.dst_video_id) { vid = r.dst_video_id; break; }
          }
          if (vid) url = `https://www.youtube.com/watch?v=${vid}`;
        }
      if (url) window.open(url, '_blank', 'noopener');
      });

      cy.on('tap', 'edge', (e)=>{
        const ed = e.target;
        info.innerHTML = `<b>엣지</b> ${ed.data('source')} → ${ed.data('target')}<br>
                          prob: ${Number(ed.data('prob')).toFixed(3)} |
                          pred: ${ed.data('pred')} |
                          act: ${ed.data('act')}`;
      });
    }

    // 컨트롤 이벤트
    document.getElementById('netMode').addEventListener('change', ()=> renderNetwork(parsedRows));
    document.getElementById('probMin').addEventListener('input', (e)=>{
      document.getElementById('probVal').textContent = Number(e.target.value).toFixed(2);
      renderNetwork(parsedRows);
    });
    document.getElementById('btnFit').addEventListener('click', ()=> { if(cy) cy.fit(); });
    document.getElementById('btnPng').addEventListener('click', ()=>{
      if(!cy) return;
      const png = cy.png({ full: true, scale: 2 });
      const a = document.createElement('a'); a.href = png; a.download = 'network.png'; a.click();
    });

    /* ========= CSV 업로드 핸들러 ========= */
    document.getElementById('csvInput').addEventListener('change', (e)=>{
      const file = e.target.files[0];
      if (!file) return;
      document.getElementById('uploadHint').textContent = `${file.name} 업로드 중…`;

      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (res)=>{
          const rows = res.data;
          // 누락/NaN 방지 변환
          const yTrue = [], yPred = [], yScore = [], st = [];
          for (const r of rows) {
            const yt = Number(r.actual_viral);
            const ys = Number(r.prediction_probability);
            const yp = Number(r.predicted_viral ?? (ys>=0.5 ? 1 : 0));
            const sst = r.src_source_type || 'Unknown';
            if (!Number.isFinite(yt) || !Number.isFinite(ys)) continue;
            yTrue.push(yt); yPred.push(yp); yScore.push(ys); st.push(sst);
          }
          if (yTrue.length === 0) {
            alert('유효한 행이 없습니다. CSV 컬럼을 확인하세요.');
            document.getElementById('uploadHint').textContent = 'CSV 업로드 실패';
            return;
          }

          const overall = metricsFromArrays(yTrue, yPred, yScore);
          renderOverall(overall);

          // 소스타입 집계
          const map = new Map();
          for (let i=0;i<st.length;i++){
            const key = st[i];
            if (!map.has(key)) map.set(key, {yT:[], yP:[], yS:[]});
            map.get(key).yT.push(yTrue[i]);
            map.get(key).yP.push(yPred[i]);
            map.get(key).yS.push(yScore[i]);
          }
          const rowsAgg = [];
          for (const [key, obj] of map.entries()){
            const m = metricsFromArrays(obj.yT, obj.yP, obj.yS);
            rowsAgg.push({ st: key, ...m });
          }

          // 차트: AUC 상위 10
          const haveAUC = rowsAgg.filter(r=> r.auc!=null);
          const top = haveAUC.sort((a,b)=> b.auc - a.auc).slice(0,10);
          renderChart(top);

          // 표: 표본수 내림차순
          rowsAgg.sort((a,b)=> b.n - a.n);
          renderTable(rowsAgg);

          // 네트워크
          parsedRows = rows;
          renderNetwork(parsedRows);

          document.getElementById('uploadHint').textContent =
            `${file.name} · ${rows.length.toLocaleString()} rows`;
        }
      });
    });
  </script>
   <!-- ▼ [ADD-ON] 소스타입 필터: 기존 코드 수정 없이 추가 -->
  <script>
  (function(){
    let selectedType = ""; // ""=전체

    function ensureTypeFilterUI(){
      const controls = document.querySelector('#netSection .flex.flex-wrap.items-center.gap-2.text-sm');
      if(!controls) return;
      if(document.getElementById('typeFilterWrap')) return;

      const wrap = document.createElement('label');
      wrap.className = 'pill';
      wrap.id = 'typeFilterWrap';
      wrap.innerHTML = `
        타입
        <select id="typeFilter" class="bg-transparent outline-none">
          <option value="">전체</option>
        </select>
        <button id="typeClear" type="button" class="ml-1">초기화</button>
      `;
      controls.prepend(wrap);

      wrap.addEventListener('change', (e)=>{
        if(e.target && e.target.id==='typeFilter'){
          selectedType = e.target.value || "";
          window._renderWithType();
        }
      });
      document.getElementById('typeClear').addEventListener('click', ()=>{
        selectedType = "";
        const sel = document.getElementById('typeFilter'); if(sel) sel.value="";
        window._renderWithType();
      });
    }

    function collectTypes(rows){
      const set = new Set();
      for(const r of rows || []){
        if(r.src_source_type) set.add(String(r.src_source_type));
        if(r.dst_source_type) set.add(String(r.dst_source_type));
      }
      return Array.from(set).sort();
    }

    function populateTypeOptions(rows){
      ensureTypeFilterUI();
      const sel = document.getElementById('typeFilter'); if(!sel) return;
      const cur = sel.value;
      const types = collectTypes(window.parsedRows || rows || []);
      sel.innerHTML = `<option value="">전체</option>` + types.map(t=>`<option value="${t}">${t}</option>`).join('');
      sel.value = types.includes(cur) ? cur : "";
    }

    // 기존 renderNetwork를 래핑(override). 기존 호출부는 그대로 사용 가능.
    window._origRenderNetwork = window.renderNetwork;

    window._renderWithType = function(){
      const rows = window.parsedRows || [];
      const type = (document.getElementById('typeFilter')?.value) || "";
      const filtered = type
        ? rows.filter(r => String(r.src_source_type)===type || String(r.dst_source_type)===type)
        : rows;
      window._origRenderNetwork(filtered);
    };

    window.renderNetwork = function(rows){
      populateTypeOptions(rows);     // 첫 렌더에서 UI/옵션 생성
      window.parsedRows = rows;      // 기존 전역 그대로 유지
      window._renderWithType();      // 현재 선택(없으면 전체)로 렌더
    };
  })();
  </script>
  <!-- ▲ [ADD-ON] 끝 -->
</body>
</html>
