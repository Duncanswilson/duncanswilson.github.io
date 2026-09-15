"""Create a portable, offline HTML report from recorded simulation runs.

The report never runs or changes the model: controls inspect the supplied traces.
All source labels are treated as untrusted text, including inside embedded JSON.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .types import SimulationResult


_LABELS = {
    "mean_rate_hz": "Mean population rate",
    "saturation_fraction": "Population at firing ceiling",
    "dopamine_concentration": "Dopamine concentration",
    "dopamine_sensitivity": "Dopamine sensitivity",
    "dopamine_reward_proxy": "Dopamine response proxy",
    "npf_exposure_proxy": "NPF exposure proxy",
    "responsiveness_hz": "Probe responsiveness",
    "dopamine_population_rate_hz": "Dopamine neuron rate",
    "probe_on": "Probe stimulus",
    "dopamine_exposure_auc": "Dopamine exposure (area under curve)",
    "dopamine_concentration_mean": "Mean dopamine concentration",
    "dopamine_population_rate_hz_mean": "Mean dopamine neuron rate",
    "dopamine_reward_proxy_mean": "Mean dopamine response proxy",
    "npf_exposure_proxy_mean": "Mean NPF exposure proxy",
    "peak_saturation_fraction": "Peak fraction at firing ceiling",
    "mean_saturation_fraction": "Mean fraction at firing ceiling",
    "mean_responsiveness_hz": "Mean probe responsiveness",
    "peak_responsiveness_hz": "Peak probe responsiveness",
    "final_dopamine_sensitivity": "Final dopamine sensitivity",
    "final_dopamine_concentration": "Final dopamine concentration",
}


def _json_value(value: Any) -> Any:
    """Normalize NumPy/dataclass values and replace non-finite floats with null."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Report data contains a non-JSON value: {type(value).__name__}")


def _script_json(value: Any) -> str:
    """Escape HTML-significant characters so JSON cannot close its script tag."""
    return (
        json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " ").capitalize())


def _number(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 100000:
            return f"{value:.4g}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _metric_table(runs: list[dict[str, Any]]) -> str:
    keys = list(dict.fromkeys(key for run in runs for key in run["metrics"]))
    heads = "".join(
        '<th scope="col" data-run="{}">{}</th>'.format(
            index, html.escape(str(run["config"].get("preset", f"Run {index + 1}")))
        )
        for index, run in enumerate(runs)
    )
    rows = []
    for key in keys:
        cells = "".join(
            '<td data-run="{}">{}</td>'.format(index, html.escape(_number(run["metrics"].get(key))))
            for index, run in enumerate(runs)
        )
        rows.append(f'<tr><th scope="row" title="{html.escape(key, quote=True)}">{html.escape(_label(key))}</th>{cells}</tr>')
    return '<table id="metrics-table"><caption class="sr-only">Recorded metrics for every run</caption><thead><tr><th scope="col">Recorded metric</th>' + heads + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _visualizer_link(target: Path) -> str:
    """Offer package navigation only where the local visualizer is available."""
    package_root = Path(__file__).resolve().parent.parent
    visualizer = package_root / "visualizer" / "index.html"
    if not target.is_relative_to(package_root / "results") or not visualizer.is_file():
        return ""
    relative = Path(os.path.relpath(visualizer, target.parent)).as_posix()
    return f'<a href="{html.escape(relative, quote=True)}" style="color:var(--teal);text-underline-offset:3px">View fly room visualization →</a>'


def write_report(
    results: list[SimulationResult],
    graph_metadata: dict,
    output_path: Path,
    optimization: dict | None = None,
) -> Path:
    """Write a self-contained interactive report and return its absolute path.

    Traces are discovered from each result; unknown trace/metric names are shown
    with human-readable labels. The report embeds the complete recorded data for
    JSON export, so no network connection or local server is required. Non-finite
    values become JSON null and appear as gaps, never as fabricated zeros.
    """
    if not results:
        raise ValueError("A report needs at least one simulation result")
    runs = [_json_value(result) for result in results]
    payload = {
        "schema": "flyreward-report-v1",
        "graph_metadata": _json_value(graph_metadata),
        "runs": runs,
        "optimization": _json_value(optimization),
        "labels": _LABELS,
    }
    target = Path(output_path).expanduser().resolve()
    replacements = {"__PAYLOAD__": _script_json(payload), "__METRIC_TABLE__": _metric_table(runs),
                    "__VISUALIZER_LINK__": _visualizer_link(target)}
    # A single substitution pass cannot reinterpret token-like source labels.
    document = re.sub(r"__PAYLOAD__|__METRIC_TABLE__|__VISUALIZER_LINK__", lambda match: replacements[match.group(0)], _HTML)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>MaleCNS reward simulation</title>
<style>
:root{--paper:#f4f3ed;--surface:#fffef9;--ink:#19342f;--muted:#64716a;--line:#dce1d7;--teal:#197b6d;--tealwash:#e4f0e9;--amber:#966419;--amberwash:#fbefdb;--radius:16px}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select{font:inherit}button,select,summary{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #c78f2f;outline-offset:3px}button{color:inherit}main{max-width:1280px;margin:auto;padding:42px 40px 28px}header{border-bottom:1px solid var(--line);padding-bottom:30px;margin-bottom:24px}.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:.17em;font-weight:750;color:var(--teal);margin:0 0 10px}.headerline{display:flex;align-items:flex-start;justify-content:space-between;gap:28px}h1{font-size:clamp(29px,3vw,44px);font-weight:650;line-height:1.13;letter-spacing:-.045em;margin:0 0 14px}h2{font-size:20px;font-weight:650;line-height:1.25;letter-spacing:-.025em;margin:0}h3{font-size:14px;margin:0 0 7px;font-weight:700}p{margin:0}.subhead{color:var(--muted);max-width:730px}.button{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:10px 15px;font-size:13px;white-space:nowrap;font-weight:600}.button:hover{border-color:var(--teal);background:var(--tealwash)}.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:19px}.tag{font-size:12px;font-weight:600;border-radius:99px;padding:4px 11px;border:1px solid var(--line);background:var(--surface);max-width:100%;overflow-wrap:anywhere}.tag.evidence{color:var(--teal);background:var(--tealwash);border-color:#c9ded2}.tag.demo{color:var(--amber);background:var(--amberwash);border-color:#ead8b2}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:24px}.card{padding:18px 20px;border:1px solid var(--line);background:var(--surface);border-radius:var(--radius)}.card-title{font-size:12px;color:var(--muted);margin-bottom:9px}.card-value{font-size:30px;line-height:1.25;letter-spacing:-.045em;font-variant-numeric:tabular-nums}.card-note{font-size:11px;color:var(--muted);margin-top:5px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:24px;margin-bottom:22px;overflow:hidden}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:22px}.small{font-size:12px;color:var(--muted)}.panel-head .small{margin-top:5px}.controls{display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap}.control-label{display:flex;flex-direction:column;font-size:11px;gap:5px;color:var(--muted);font-weight:600}select{min-width:190px;max-width:100%;padding:8px 34px 8px 10px;background:var(--surface);border:1px solid var(--line);border-radius:8px;color:var(--ink);font-size:12px}.check{display:flex;align-items:center;gap:7px;font-size:12px;margin:0 0 8px;white-space:nowrap}.check input{accent-color:var(--teal)}.chart-wrap{position:relative;margin:0 -8px}#chart{display:block;width:100%;height:auto;min-height:225px;overflow:visible}#chart text{font-family:inherit}#chart-status{padding:0 12px;font-size:12px;color:var(--muted)}.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:15px}.legend button{border:1px solid transparent;border-radius:7px;background:transparent;font-size:12px;padding:6px 10px;display:flex;gap:7px;align-items:center}.legend button[aria-pressed="true"]{border-color:var(--line);background:var(--paper);font-weight:650}.legend button:hover{background:var(--tealwash)}.swatch{width:17px;height:3px;border-radius:2px}.chart-foot{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line);margin-top:17px;padding-top:12px;font-size:11px;color:var(--muted)}.table-scroll{overflow:auto;margin:0 -24px;padding:0 24px 3px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}th,td{padding:11px 14px;border-bottom:1px solid var(--line)}thead th{font-weight:650;color:var(--muted);background:var(--paper)}th:first-child{text-align:left;min-width:240px;padding-left:0}tbody th{font-weight:500;color:var(--muted)}thead th:first-child{padding-left:12px}tbody tr:last-child>*{border-bottom:0}td.is-selected,thead th.is-selected{background:var(--tealwash);color:var(--ink)}.columns{display:grid;grid-template-columns:1fr 1fr;gap:28px}.note{font-size:13px;color:var(--muted)}.note strong{color:var(--ink);font-weight:600}.notice{margin-top:16px;border-left:3px solid #c79648;padding:9px 14px;background:var(--amberwash);border-radius:0 7px 7px 0;font-size:12px;color:#765725}.details-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}details{border:1px solid var(--line);border-radius:9px;padding:11px 14px;min-width:0}summary{font-size:12px;font-weight:600}pre{font-size:11px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;margin:14px 0 0;color:var(--muted)}#optimization-panel[hidden]{display:none}.search-body{display:grid;grid-template-columns:1fr 1fr;gap:28px}.search-body details{margin:0}footer{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:11px;padding:0 2px}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}noscript p{border:1px solid var(--line);background:var(--amberwash);padding:15px;margin-bottom:18px}
[hidden]{display:none!important}.button:disabled{opacity:.42;cursor:default}.button:disabled:hover{background:var(--surface);border-color:var(--line)}
@media(max-width:900px){main{padding:28px 20px}.panel-head{flex-direction:column}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.headerline{gap:14px}.headerline .button{margin-top:0}.columns,.search-body{gap:20px}.controls{width:100%}.control-label{flex:1}select{width:100%}}
@media(max-width:580px){main{padding:24px 12px}.headerline{display:block}.headerline .button{margin-top:20px}header{padding-bottom:22px}.cards{gap:9px}.card{padding:15px 13px}.card-value{font-size:25px}.panel{padding:18px}.panel-head{gap:16px}.columns,.details-row,.search-body{grid-template-columns:1fr}.columns{gap:22px}.controls{display:grid;grid-template-columns:1fr}.check{margin:3px 0 0}.chart-foot,footer{display:block}.table-scroll{margin:0 -18px;padding:0 18px 3px}.tag{font-size:11px}.legend{gap:2px}.legend button{padding:5px 7px}}
@media print{body{background:white}main{padding:0}.button,.controls,details{display:none}.panel,.cards{break-inside:avoid}.table-scroll{overflow:visible}#chart{min-height:0}.card{padding:12px}.card-value{font-size:24px}footer{margin-top:15px}}
</style>
</head>
<body><main>
<header>
<p class="eyebrow">Fly reward lab · Recorded experiment</p>
<div class="headerline"><div><h1>MaleCNS reward simulation</h1><p class="subhead">Explore how modeled dopamine persistence, neural drive, and an abstract NPF signal change a connectome-based network over time.</p></div><button class="button" id="download" type="button">Export recorded JSON ↓</button></div>
<div class="tags"><span class="tag" id="dataset-tag">Connectivity input</span><span class="tag" id="graph-tag">Recorded graph</span><span class="tag" id="runs-tag">Recorded runs</span><span class="tag">Model output · Pleasure is not measured</span></div>
</header>
<noscript><p>The comparison table is available below. Enable JavaScript to inspect recorded traces and export the embedded data.</p></noscript>
<section class="cards" id="summary" aria-label="Selected run metrics" aria-live="polite"></section>
<section class="panel" aria-labelledby="dynamics-heading">
<div class="panel-head"><div><h2 id="dynamics-heading">Network dynamics</h2><p class="small">Recorded traces · <span id="active-run">Selected run</span></p></div><div class="controls"><label class="control-label">Inspect signal<select id="trace-select" aria-label="Signal to plot"></select></label><label class="control-label">Selected run<select id="run-select" aria-label="Selected simulation run"></select></label><label class="check"><input type="checkbox" id="overlay" checked><span id="overlay-label">Compare all runs</span></label></div></div>
<div class="chart-wrap"><svg id="chart" viewBox="0 0 1100 370" role="img" aria-label="Recorded simulation time series"><title>Recorded simulation time series</title></svg><p id="chart-status" role="status"></p></div>
<div class="legend" id="legend" aria-label="Select a recorded run"></div>
<div class="chart-foot"><span id="units-note">Time in seconds. Dopamine and NPF use arbitrary model units.</span><span>Controls inspect saved data; they do not rerun the simulation.</span></div>
</section>
<section class="panel" aria-labelledby="comparison-heading"><div class="panel-head"><div><h2 id="comparison-heading">Compare every run</h2><p class="small" id="table-window-note">All recorded summary metrics. The selected run is highlighted.</p></div><nav id="run-pager" hidden aria-label="Comparison run groups"><button class="button" type="button" id="previous-group">← Previous</button> <span class="small" id="page-status" aria-live="polite"></span> <button class="button" type="button" id="next-group">Next →</button></nav></div><div class="table-scroll">__METRIC_TABLE__</div></section>
<section class="panel" id="optimization-panel" hidden aria-labelledby="search-heading"><div class="panel-head"><div><h2 id="search-heading">Best tested configuration</h2><p class="small">A bounded parameter search, not a proven global maximum.</p></div></div><div class="search-body"><div><h3 id="optimization-winner">Recorded search result</h3><p class="note" id="optimization-note">This search ranks simulated outcomes using the recorded objective. Its optimum depends on model assumptions, parameter bounds, and the tested candidates.</p><pre id="winner-parameters"></pre></div><details><summary>Inspect full search result</summary><pre id="optimization-json"></pre></details></div></section>
<section class="panel" aria-labelledby="provenance-heading"><div class="panel-head"><div><h2 id="provenance-heading">What comes from data, and what is modeled</h2><p class="small">A compact provenance record for interpreting these results.</p></div></div><div class="columns"><div><h3>Connectivity evidence</h3><p class="note" id="connectivity-note">The graph provenance is recorded below. Anatomical connection counts alone do not determine synaptic strength, receptor effects, or neural dynamics.</p></div><div><h3>Physiology assumptions</h3><p class="note">Firing rates, dopamine release and clearance, sensitivity, and tolerance are <strong>model assumptions</strong>. NPF is an <strong>abstract global proxy</strong>, without verified cell or receptor mapping. These are not measured drug responses.</p></div></div><p class="notice">Higher dopamine exposure or a larger reward proxy is a numerical model outcome. This report does not establish subjective pleasure or biological validation.</p><div class="details-row"><details><summary>Graph provenance and input metadata</summary><pre id="graph-json"></pre></details><details><summary>Selected run parameters and model metadata</summary><pre id="run-json"></pre></details></div></section>
<footer><span>Fly reward lab · Self-contained experiment record</span>__VISUALIZER_LINK__<span id="footer-detail">No external scripts, fonts, or services.</span></footer>
</main>
<script id="experiment-data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
(() => {
const data = JSON.parse(document.getElementById("experiment-data").textContent);
const runs = data.runs;
const colors = ["#197b6d","#b87828","#7076ad","#b05e61","#4c8fa8","#7d8d46","#ad71a1","#675e50"];
const byId = id => document.getElementById(id);
const labels = data.labels || {};
const label = key => labels[key] || key.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
const runName = (run, index) => String(run.config.preset || `Run ${index + 1}`);
const number = value => {
 if (value == null || typeof value !== "number" || !Number.isFinite(value)) return "Not recorded";
 if (value === 0) return "0";
 if (Math.abs(value) < .001 || Math.abs(value) >= 100000) return value.toExponential(3);
 return value.toLocaleString("en-US", {maximumFractionDigits:4});
};
const shortNumber = value => {
 if (value === 0) return "0";
 if (Math.abs(value) < .001 || Math.abs(value) >= 10000) return value.toExponential(1);
 return value.toLocaleString("en-US", {maximumFractionDigits:3});
};
const traceKeys = [...new Set(runs.flatMap(run => Object.keys(run.traces)))];
const winnerPreset = data.optimization?.winner?.preset;
let selectedRun = Math.max(0,runs.findIndex(run => run.config.preset === winnerPreset));
const pageSize = 8;
let visibleStart = Math.floor(selectedRun / pageSize) * pageSize;
const visibleIndices = () => Array.from({length:Math.min(pageSize,runs.length-visibleStart)},(_,index)=>visibleStart+index);
let selectedTrace = traceKeys.includes("dopamine_concentration") ? "dopamine_concentration" : traceKeys[0];
let compareAll = true;
const option = (text, value) => {const el = document.createElement("option"); el.textContent = text; el.value = value; return el;};
runs.forEach((run, index) => byId("run-select").append(option(runName(run,index), index)));
byId("run-select").value = String(selectedRun);
if(runs.length > pageSize){byId("run-pager").hidden=false;byId("overlay-label").textContent="Compare this group";}
traceKeys.forEach(key => byId("trace-select").append(option(label(key), key)));
byId("trace-select").value = selectedTrace || "";
byId("trace-select").disabled = traceKeys.length === 0;
byId("runs-tag").textContent = `${runs.length} recorded ${runs.length === 1 ? "run" : "runs"}`;
byId("graph-json").textContent = JSON.stringify(data.graph_metadata, null, 2);
const gm = data.graph_metadata || {};
const sourceText = [gm.source_label,gm.dataset,gm.source,gm.graph_kind,gm.kind].filter(v => typeof v === "string").join(" · ");
const isSynthetic = gm.synthetic === true || gm.is_demo === true || gm.is_synthetic === true || /synthetic|demo/i.test(sourceText);
const isMaleCNS = /male.?cns/i.test(sourceText);
const tag = byId("dataset-tag");
tag.textContent = isSynthetic ? "Synthetic demo graph" : (isMaleCNS ? "MaleCNS connectivity input" : "Connectivity input · See provenance");
tag.classList.add(isSynthetic ? "demo" : "evidence");
if (isSynthetic) byId("connectivity-note").textContent = "This is a synthetic demonstration graph, not downloaded MaleCNS anatomy. It exercises the simulator and comparison workflow; its results do not provide evidence about the actual fly connectome.";
else if (isMaleCNS) byId("connectivity-note").textContent = "This run uses imported MaleCNS connectivity. The exact source, graph filtering, and retained population are recorded below. Synapse counts and transmitter annotations constrain the model; they do not measure receptor effects or physiological strengths.";
const nodeCount = gm.retained_neurons ?? gm.n_neurons ?? gm.neurons ?? gm.node_count ?? gm.nodes;
const edgeCount = gm.directed_pairs ?? gm.n_edges ?? gm.edges ?? gm.edge_count ?? gm.connections;
const stats = [];
if (typeof nodeCount === "number") stats.push(`${nodeCount.toLocaleString("en-US")} neurons`);
if (typeof edgeCount === "number") stats.push(`${edgeCount.toLocaleString("en-US")} connections`);
byId("graph-tag").textContent = stats.join(" · ") || (sourceText || "Graph details in provenance");
if (data.optimization) {
 byId("optimization-panel").hidden = false;
 byId("optimization-json").textContent = JSON.stringify(data.optimization, null, 2);
 const objective = data.optimization.objective ?? data.optimization.objective_name;
 if (typeof objective === "string") byId("optimization-note").textContent = `Recorded objective: ${label(objective)}. This search ranks outcomes only within its model, parameter bounds, and tested candidates. The complete search result is available alongside this note.`;
 const winner = data.optimization.winner;
 if (winner && typeof winner === "object") {
  byId("optimization-winner").textContent = `${String(winner.preset || "Winning candidate")} · Score ${number(winner.score)}`;
  byId("winner-parameters").textContent = JSON.stringify(winner.config || {},null,2);
 }
 if (typeof data.optimization.definition === "string") byId("optimization-note").textContent += ` Score definition: ${data.optimization.definition}`;
}
function summaries() {
 const current = runs[selectedRun];
 byId("active-run").textContent = runName(current, selectedRun);
 byId("run-json").textContent = JSON.stringify({config:current.config,metadata:current.metadata}, null, 2);
 const localDopamine = typeof current.metadata?.dopamine_recipient_neuron_count === "number";
 byId("units-note").textContent=localDopamine?"Time in seconds. Dopamine traces average over anatomical recipients; concentrations use model units.":"Time in seconds. Dopamine and NPF use arbitrary model units.";
 const cards = [
  ["dopamine_exposure_auc","Dopamine exposure",localDopamine?"Recipient mean · Model units × seconds":"Model units × seconds"],
  ["mean_rate_hz","Mean population rate","Hz · Mean over recorded run"],
  ["final_dopamine_sensitivity","Final dopamine sensitivity",localDopamine?"Recipient mean · Adaptation state":"Dimensionless · Adaptation state"],
  ["peak_saturation_fraction","Peak saturation","Fraction of neurons at firing ceiling"]
 ];
 byId("summary").replaceChildren();
 cards.forEach(([key,title,note]) => {
  const card = document.createElement("div"); card.className = "card";
  [["card-title",title],["card-value",number(current.metrics[key])],["card-note",note]].forEach(([cls,value])=>{const el=document.createElement("p");el.className=cls;el.textContent=value;card.append(el);});
  byId("summary").append(card);
 });
 const visible = visibleIndices();
 document.querySelectorAll("#metrics-table [data-run]").forEach(el => {const index=Number(el.dataset.run);el.classList.toggle("is-selected",index === selectedRun);el.hidden=!visible.includes(index);});
 if(runs.length > pageSize){
  byId("table-window-note").textContent=`All metrics for runs ${visibleStart+1}–${visibleStart+visible.length} of ${runs.length}. The chart compares this group. Select any run above to jump to its group.`;
  byId("page-status").textContent=`${Math.floor(visibleStart/pageSize)+1} / ${Math.ceil(runs.length/pageSize)}`;
  byId("previous-group").disabled=visibleStart===0;
  byId("next-group").disabled=visibleStart+pageSize>=runs.length;
 }
 byId("legend").replaceChildren();
 runs.forEach((run,index) => {
  if(!visible.includes(index))return;
  const button = document.createElement("button");button.type="button";button.setAttribute("aria-pressed",String(index === selectedRun));
  const swatch=document.createElement("span");swatch.className="swatch";swatch.style.background=colors[index%colors.length];swatch.setAttribute("aria-hidden","true");
  const text=document.createElement("span");text.textContent=runName(run,index);button.append(swatch,text);
  button.addEventListener("click",()=>{selectedRun=index;byId("run-select").value=String(index);render();});
  byId("legend").append(button);
 });
}
const svgNS="http://www.w3.org/2000/svg";
function svgElement(name, attrs, text) {const el=document.createElementNS(svgNS,name);Object.entries(attrs||{}).forEach(([key,value])=>el.setAttribute(key,String(value)));if(text !== undefined)el.textContent=text;return el;}
function chart() {
 const svg=byId("chart");svg.replaceChildren();
 svg.append(svgElement("title",{},`${label(selectedTrace || "No signal")}, recorded time series`));
 const indices=compareAll ? visibleIndices() : [selectedRun];
 const series=indices.map(index=>({index,run:runs[index],values:runs[index].traces[selectedTrace]||[]}));
 let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity,count=0;
 series.forEach(({run,values})=>values.forEach((value,i)=>{const time=run.time[i];if(typeof value==="number"&&Number.isFinite(value)&&typeof time==="number"&&Number.isFinite(time)){xMin=Math.min(xMin,time);xMax=Math.max(xMax,time);yMin=Math.min(yMin,value);yMax=Math.max(yMax,value);count++;}}));
 if(!count){svg.append(svgElement("text",{x:550,y:180,"text-anchor":"middle",fill:"#64716a","font-size":15},"No finite recorded values for this signal."));byId("chart-status").textContent="Choose another signal or run to inspect recorded data.";return;}
 if(xMin===xMax)xMax=xMin+1;
 if(yMin>=0)yMin=0;
 if(yMin===yMax)yMax=yMin+1;
 const pad=(yMax-yMin)*.06;yMax+=pad;if(yMin<0)yMin-=pad;
 const left=78,right=24,top=25,bottom=49,width=1100-left-right,height=370-top-bottom;
 const X=value=>left+(value-xMin)/(xMax-xMin)*width,Y=value=>top+(yMax-value)/(yMax-yMin)*height;
 for(let i=0;i<=4;i++){
  const v=yMin+(yMax-yMin)*i/4,y=Y(v);
  svg.append(svgElement("line",{x1:left,y1:y,x2:left+width,y2:y,stroke:"#e2e7dd","stroke-width":1}));
  svg.append(svgElement("text",{x:left-12,y:y+4,"text-anchor":"end",fill:"#64716a","font-size":11},shortNumber(v)));
 }
 for(let i=0;i<=5;i++){
  const v=xMin+(xMax-xMin)*i/5,x=X(v);
  svg.append(svgElement("text",{x,y:top+height+26,"text-anchor":"middle",fill:"#64716a","font-size":11},shortNumber(v)));
 }
 svg.append(svgElement("text",{x:left,y:12,fill:"#64716a","font-size":11},label(selectedTrace)));
 svg.append(svgElement("text",{x:left+width,y:364,"text-anchor":"end",fill:"#64716a","font-size":11},"Time (s)"));
 // Draw the selected run last, and retain gaps for non-finite samples.
 series.sort((a,b)=>Number(a.index===selectedRun)-Number(b.index===selectedRun));
 series.forEach(({index,run,values})=>{
  let path="",open=false,finiteSamples=0,lastPoint=null;
  values.forEach((value,i)=>{const time=run.time[i];if(typeof value!=="number"||!Number.isFinite(value)||typeof time!=="number"||!Number.isFinite(time)){open=false;return;}path+=`${open?"L":"M"}${X(time).toFixed(2)},${Y(value).toFixed(2)} `;open=true;finiteSamples++;lastPoint=[X(time),Y(value)];});
  const line=svgElement("path",{d:path,fill:"none",stroke:colors[index%colors.length],"stroke-width":index===selectedRun?2.8:1.8,opacity:index===selectedRun?1:.55,"stroke-linejoin":"round","stroke-linecap":"round"});
  line.append(svgElement("title",{},runName(run,index)));svg.append(line);
  if(finiteSamples===1&&lastPoint)svg.append(svgElement("circle",{cx:lastPoint[0],cy:lastPoint[1],r:4,fill:colors[index%colors.length]}));
 });
 const selectedAvailable=Array.isArray(runs[selectedRun].traces[selectedTrace]);
 byId("chart-status").textContent=selectedAvailable?"":`The selected run did not record ${label(selectedTrace).toLowerCase()}; available comparison traces are shown.`;
 svg.setAttribute("aria-label",`${label(selectedTrace)} over ${shortNumber(xMin)} to ${shortNumber(xMax)} seconds. ${compareAll ? `${indices.length} recorded runs in the current group are compared.` : runName(runs[selectedRun],selectedRun)+" is shown."} Numerical summaries are available in the comparison table.`);
}
function render(){summaries();chart();}
byId("run-select").addEventListener("change",event=>{selectedRun=Number(event.target.value);visibleStart=Math.floor(selectedRun/pageSize)*pageSize;render();});
function changePage(direction){visibleStart=Math.max(0,Math.min(Math.floor((runs.length-1)/pageSize)*pageSize,visibleStart+direction*pageSize));selectedRun=visibleStart;byId("run-select").value=String(selectedRun);render();}
byId("previous-group").addEventListener("click",()=>changePage(-1));
byId("next-group").addEventListener("click",()=>changePage(1));
byId("trace-select").addEventListener("change",event=>{selectedTrace=event.target.value;chart();});
byId("overlay").addEventListener("change",event=>{compareAll=event.target.checked;chart();});
byId("download").addEventListener("click",()=>{
 const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download="flyreward-recorded-experiment.json";document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
})();
</script>
</body></html>'''
