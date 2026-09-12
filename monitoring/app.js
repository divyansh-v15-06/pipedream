const REFRESH_MS = 5000;
let latest = null;

const $ = (id) => document.getElementById(id);

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  return `${minutes}m ${String(secs).padStart(2, '0')}s`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString('en-US');
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let number = bytes;
  let unit = 0;
  while (number >= 1024 && unit < units.length - 1) { number /= 1024; unit += 1; }
  return `${number.toFixed(number >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatTime(value) {
  if (!value) return 'Not started';
  const date = new Date(value);
  return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function setBar(id, value) {
  $(id).style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
}

function stateLabel(status) {
  return { running: 'RUNNING', completed: 'COMPLETE', exited: 'STOPPED', missing: 'OFFLINE' }[status] || String(status || 'UNKNOWN').toUpperCase();
}

function render(data) {
  latest = data;
  const container = data.container || {};
  const run = data.run || {};
  const resources = data.resources || {};
  const docker = resources.docker || {};
  const system = resources.system || {};
  const gpu = resources.gpu;
  const activeSeed = run.activeSeed == null ? 'No active seed' : `Seed ${run.activeSeed} is learning`;

  $('connectionLabel').textContent = 'live telemetry';
  $('runStatus').textContent = stateLabel(container.status);
  $('runStatusMeta').textContent = container.running ? `seed ${run.activeSeed ?? '—'} · ${run.completedSeeds || 0}/${run.totalSeeds || 0} complete` : 'Container is not running';
  $('elapsed').textContent = formatDuration(container.elapsedSeconds);
  $('startedAt').textContent = `started ${formatTime(container.startedAt)}`;
  $('eta').textContent = formatDuration(run.etaSeconds);
  $('containers').textContent = container.running ? '01' : '00';
  $('containerMeta').textContent = container.name || 'Docker workload';
  $('heroDescription').textContent = container.running
    ? `PPO is training across ${run.totalSeeds || 0} seeds on the development pilot. The dashboard refreshes every ${REFRESH_MS / 1000} seconds.`
    : 'The training container is offline. Start a run to stream live telemetry here.';
  $('overallPercent').textContent = `${Number(run.progressPercent || 0).toFixed(1)}%`;
  setBar('overallProgress', run.progressPercent);
  $('overallSteps').textContent = `${formatNumber(run.totalSteps)} / ${formatNumber((run.totalSeeds || 0) * (run.totalTimesteps || 0))} timesteps`;
  $('activeSeed').textContent = activeSeed;
  $('containerName').textContent = container.name || '—';
  $('outputDir').textContent = run.outputDir || '—';
  $('lastUpdated').textContent = `updated ${new Date().toLocaleTimeString()}`;

  const load = system.loadAverage?.[0] || 0;
  const cpu = docker.cpuPercent || 0;
  $('cpuValue').textContent = `${cpu.toFixed(1)}%`;
  $('cpuMeta').textContent = `${load.toFixed(2)} load · ${system.cpuCount || '—'} host threads`;
  setBar('cpuBar', Math.min(100, cpu));
  const memoryPercent = docker.memoryPercent || 0;
  $('memoryValue').textContent = memoryPercent ? `${memoryPercent.toFixed(1)}%` : '—';
  $('memoryMeta').textContent = docker.memoryBytes ? `${formatBytes(docker.memoryBytes)} of ${formatBytes(docker.memoryLimitBytes)}` : 'Container memory unavailable';
  setBar('memoryBar', memoryPercent);
  if (gpu) {
    const gpuPercent = gpu.memoryUsedMiB * 100 / Math.max(1, gpu.memoryTotalMiB);
    $('gpuValue').textContent = `${gpuPercent.toFixed(1)}%`;
    $('gpuMeta').textContent = `${Math.round(gpu.memoryUsedMiB)} / ${Math.round(gpu.memoryTotalMiB)} MiB · ${gpu.name}`;
    setBar('gpuBar', gpuPercent);
  } else {
    $('gpuValue').textContent = 'idle';
    $('gpuMeta').textContent = 'No GPU telemetry available';
    setBar('gpuBar', 0);
  }

  const seeds = data.seeds || [];
  $('seedGrid').innerHTML = seeds.map((seed) => `
    <div class="seed-card ${seed.status === 'training' ? 'active' : ''} ${seed.status === 'completed' ? 'done' : ''}">
      <div class="seed-top"><span class="seed-name">SEED ${seed.seed}</span><span class="seed-status">${seed.status}</span></div>
      <div class="seed-progress"><div class="progress-track"><span style="width:${seed.progressPercent || 0}%"></span></div></div>
      <div class="seed-stat"><span>${formatNumber(seed.step)} / ${formatNumber(seed.totalTimesteps)}</span><strong>${seed.fps ? `${seed.fps.toFixed(0)} fps` : 'queued'}</strong></div>
    </div>`).join('');
}

function renderLogs(data) {
  const lines = data.lines || [];
  const window = $('logWindow');
  if (!lines.length) { window.innerHTML = '<span class="log-muted">No container output yet.</span>'; return; }
  window.innerHTML = lines.slice(-12).map((line) => `<span>${escapeHtml(line)}</span>`).join('');
  window.scrollTop = window.scrollHeight;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
}

async function refresh() {
  try {
    const [statusResponse, logsResponse] = await Promise.all([fetch('/api/status'), fetch('/api/logs')]);
    if (!statusResponse.ok || !logsResponse.ok) throw new Error('monitor API unavailable');
    render(await statusResponse.json());
    renderLogs(await logsResponse.json());
  } catch (error) {
    $('connectionLabel').textContent = 'disconnected';
    $('heroDescription').textContent = 'The monitor server cannot reach its local telemetry source. Retrying automatically…';
  }
}

$('refreshButton').addEventListener('click', refresh);
refresh();
setInterval(refresh, REFRESH_MS);
