// Mull web interface. Uses only the public job API, so it works against any Mull server.
"use strict";

const $ = (id) => document.getElementById(id);
const FINISHED = ["done", "failed", "cancelled"];
let selected = null;
let events = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function seconds(job) {
  if (!job.started) return "";
  const end = job.finished || Date.now() / 1000;
  const elapsed = end - job.started;
  const rate = job.tokens && elapsed > 0 ? `, ${(job.tokens / elapsed).toFixed(1)} tok/s` : "";
  return `${elapsed.toFixed(1)} s${rate}`;
}

async function refreshHealth() {
  const box = $("health");
  try {
    const h = await api("/health");
    const running = h.running_job ? `running #${h.running_job}` : "idle";
    box.textContent = `${h.model || "model"} · ${running} · ${h.jobs.queued} queued · ` +
      `${h.free_ram_gb} of ${h.total_ram_gb} GB RAM free`;
    box.classList.remove("down");
  } catch (error) {
    box.textContent = `server unreachable (${error.message})`;
    box.classList.add("down");
  }
}

function jobItem(job) {
  const li = document.createElement("li");
  li.dataset.id = job.id;
  li.classList.toggle("selected", job.id === selected);
  const id = document.createElement("span");
  id.textContent = `#${job.id}`;
  const prompt = document.createElement("span");
  prompt.className = "prompt";
  prompt.textContent = job.prompt;
  const state = document.createElement("span");
  state.className = `state ${job.state}`;
  state.textContent = job.state;
  li.append(id, prompt, state);
  li.onclick = () => select(job.id);
  return li;
}

async function refreshJobs() {
  const state = $("filter").value;
  const list = $("jobs");
  try {
    const jobs = await api(`/jobs?limit=100${state ? `&state=${state}` : ""}`);
    list.replaceChildren(...(jobs.length ? jobs.map(jobItem) : [emptyItem()]));
  } catch {
    // health line already shows the server is down
  }
}

function emptyItem() {
  const li = document.createElement("li");
  li.className = "empty";
  li.textContent = "No jobs yet.";
  return li;
}

function showJob(job) {
  $("detail").hidden = false;
  $("detail-title").textContent = `Job #${job.id}`;
  const opts = job.options || {};
  $("detail-meta").textContent = [`${job.state}`, `${job.tokens} tokens`, seconds(job),
    opts.thinking ? "thinking on" : "", opts.temperature ? `temperature ${opts.temperature}` : ""]
    .filter(Boolean).join(" · ");
  $("detail-prompt").textContent = (opts.system ? `[system] ${opts.system}\n\n` : "") + job.prompt;
  $("reasoning-box").hidden = !job.reasoning;
  $("detail-reasoning").textContent = job.reasoning || "";
  $("detail-output").textContent = job.output || (job.state === "queued" ? "Waiting in the queue…" : "");
  $("detail-error").textContent = job.error || "";
  $("cancel").disabled = FINISHED.includes(job.state);
  $("delete").disabled = !FINISHED.includes(job.state);
}

function select(id) {
  selected = id;
  if (events) events.close();
  document.querySelectorAll("#jobs li").forEach((li) => li.classList.toggle("selected", +li.dataset.id === id));
  events = new EventSource(`/jobs/${id}/events`);
  events.addEventListener("job", (event) => {
    const job = JSON.parse(event.data);
    showJob(job);
    if (FINISHED.includes(job.state)) { events.close(); refreshJobs(); }
  });
  events.onerror = () => events.close();
}

function formBody(form) {
  const data = new FormData(form);
  const body = {
    prompt: data.get("prompt"),
    max_new_tokens: Number(data.get("max_new_tokens")),
    temperature: Number(data.get("temperature")),
    thinking: data.get("thinking") === "on",
  };
  const system = data.get("system").trim();
  if (system) body.system = system;
  const stop = data.get("stop").split(",").map((s) => s.trim()).filter(Boolean);
  if (stop.length) body.stop = stop;
  return body;
}

$("submit").onsubmit = async (event) => {
  event.preventDefault();
  $("submit-error").textContent = "";
  try {
    const { id } = await api("/jobs", { method: "POST", body: JSON.stringify(formBody(event.target)) });
    event.target.prompt.value = "";
    await refreshJobs();
    select(id);
  } catch (error) {
    $("submit-error").textContent = error.message;
  }
};

$("cancel").onclick = () => selected && api(`/jobs/${selected}/cancel`, { method: "POST" }).then(refreshJobs);
$("delete").onclick = async () => {
  if (!selected) return;
  await api(`/jobs/${selected}`, { method: "DELETE" }).catch((e) => alert(e.message));
  selected = null;
  $("detail").hidden = true;
  refreshJobs();
};
$("filter").onchange = refreshJobs;

refreshHealth();
refreshJobs();
setInterval(refreshHealth, 5000);
setInterval(refreshJobs, 3000);
