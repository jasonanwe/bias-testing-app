/*
 * AI Bias Testing Tool
 * Disparate impact analysis for AI systems
 *
 * MIT License
 *
 * Copyright (c) 2026 Jason Wemer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Repository: https://github.com/jasonanwe/bias-testing-app
 *
 * No network calls. All computation, including the plain-language summary, runs
 * locally in the browser. Nothing entered here is transmitted anywhere.
 */

import React, { useState, useCallback, useMemo } from "react";

const STEPS = ["System Info", "Comparisons", "Metrics", "Statistics", "Context", "Classification", "Report"];

const STEP_INFO = [
  ["System Info", "The system name, owner, register ID, test date, and whether this is pre-deployment, recurring, or incident-triggered."],
  ["Comparisons", "Each pair of groups you are comparing, defined by protected class, with the favourable and unfavourable outcome counts for each. Add as many comparisons as the use case needs. Gather these before you start."],
  ["Metrics", "Calculated for you. Selection rates and the disparate impact ratio for every comparison, against the four-fifths rule."],
  ["Statistics", "Calculated for you. Fisher's Exact, chi-square, and two-proportion z-test run on each comparison, plus a multiplicity-adjusted p-value shown for context."],
  ["Context", "Four questions about the decision: its type, who it affects, how many decisions per year, and how reversible it is. Answered once for the system."],
  ["Classification", "A classification for each comparison and an overall result for the system, plus a plain-language summary."],
  ["Report", "A single printable record covering every comparison, for your AI inventory and governance review."],
];

const LEVELS = ["None", "Low", "Moderate", "Critical"];
const DASH = "\u2014";

const SEV_COLORS = {
  None: { bg: "#e8f5e9", text: "#1b5e20", border: "#4caf50" },
  Low: { bg: "#fff8e1", text: "#e65100", border: "#ffa000" },
  Moderate: { bg: "#fff3e0", text: "#bf360c", border: "#e65100" },
  Critical: { bg: "#ffebee", text: "#b71c1c", border: "#c62828" },
};
const NEUTRAL = { bg: "#f9fafb", text: "#6b7280", border: "#d1d5db" };

const CONTEXT_LABELS = {
  decisionType: { label: "Decision type", options: ["Informational (Standard)", "Influences human decision (Elevated, +1)", "Directly determines outcomes (High, +2)"], scores: [0, 1, 2] },
  affectedPop: { label: "Affected population", options: ["Internal staff only (Standard)", "General public (Elevated, +1)", "Vulnerable populations (High, +2)"], scores: [0, 1, 2] },
  scale: { label: "Scale of impact", options: ["< 100 decisions/year (Standard)", "100 - 10,000 decisions/year (Elevated, +1)", "> 10,000 decisions/year (High, +2)"], scores: [0, 1, 2] },
  reversibility: { label: "Reversibility", options: ["Easily reversed (Standard)", "Reversible with effort (Elevated, +1)", "Difficult to reverse (High, +2)"], scores: [0, 1, 2] },
};

/* ---------------- statistics ---------------- */

function logFactorial(n) {
  let r = 0;
  for (let i = 2; i <= n; i++) r += Math.log(i);
  return r;
}

function fisherExact(a, b, c, d) {
  const n = a + b + c + d;
  const logDenom = logFactorial(a + b) + logFactorial(c + d) + logFactorial(a + c) + logFactorial(b + d) - logFactorial(n);
  const pCutoff = logDenom - (logFactorial(a) + logFactorial(b) + logFactorial(c) + logFactorial(d));
  let p = 0;
  const maxI = Math.min(a + b, a + c);
  for (let i = 0; i <= maxI; i++) {
    const j = a + b - i, k = a + c - i, l = c + d - k;
    if (j < 0 || k < 0 || l < 0) continue;
    const logP = logDenom - (logFactorial(i) + logFactorial(j) + logFactorial(k) + logFactorial(l));
    if (logP <= pCutoff + 1e-10) p += Math.exp(logP);
  }
  return Math.min(p, 1);
}

function chiSquare(a, b, c, d) {
  const n = a + b + c + d;
  if (n === 0) return { valid: false, stat: 0, p: 1 };
  const e = [(a + b) * (a + c) / n, (a + b) * (b + d) / n, (c + d) * (a + c) / n, (c + d) * (b + d) / n];
  if (e.some(x => x < 5)) return { valid: false, stat: 0, p: 1 };
  const o = [a, b, c, d];
  let stat = 0;
  for (let i = 0; i < 4; i++) stat += Math.pow(o[i] - e[i], 2) / e[i];
  return { valid: true, stat, p: 1 - chi2CDF(stat) };
}

function chi2CDF(x) {
  if (x <= 0) return 0;
  return erf(Math.sqrt(x / 2));
}

function erf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function zTest(p1, n1, p2, n2) {
  if (n1 < 30 || n2 < 30) return { valid: false, z: 0, p: 1 };
  const pHat = (p1 * n1 + p2 * n2) / (n1 + n2);
  const se = Math.sqrt(pHat * (1 - pHat) * (1 / n1 + 1 / n2));
  if (se === 0) return { valid: false, z: 0, p: 1 };
  const z = (p1 - p2) / se;
  return { valid: true, z, p: 2 * (1 - normalCDF(Math.abs(z))) };
}

function normalCDF(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function classifyDI(ratio) {
  if (ratio >= 0.90) return "None";
  if (ratio >= 0.80) return "Low";
  if (ratio >= 0.65) return "Moderate";
  return "Critical";
}

function fmtP(p) { return p < 0.001 ? "< 0.001" : p.toFixed(4); }
function pLabel(p) { return p <= 0.01 ? "(highly significant)" : p <= 0.05 ? "(significant)" : p <= 0.10 ? "(marginal)" : "(not significant)"; }
function pct(x) { return (x * 100).toFixed(1) + "%"; }

function listJoin(items) {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return items[0] + " and " + items[1];
  return items.slice(0, -1).join(", ") + ", and " + items[items.length - 1];
}

/*
 * Holm-Bonferroni step-down adjustment. Advisory only: the raw p-value
 * continues to drive the finding.
 */
function holmAdjust(ps) {
  const m = ps.length;
  if (m === 0) return [];
  const order = ps.map((p, i) => ({ p, i })).sort((x, y) => x.p - y.p);
  const out = new Array(m);
  let running = 0;
  order.forEach((o, k) => {
    const v = Math.min(1, (m - k) * o.p);
    running = Math.max(running, v);
    out[o.i] = running;
  });
  return out;
}

/* ---------------- per-comparison computation ---------------- */

function newComparison() {
  return { label: "", aName: "Group A", bName: "Group B", aPos: "", aNeg: "", bPos: "", bNeg: "" };
}

function compStats(c) {
  const a = parseInt(c.aPos) || 0, b = parseInt(c.aNeg) || 0, cc = parseInt(c.bPos) || 0, d = parseInt(c.bNeg) || 0;
  const n1 = a + b, n2 = cc + d;
  const ready = n1 > 0 && n2 > 0;
  const rateA = n1 > 0 ? a / n1 : 0, rateB = n2 > 0 ? cc / n2 : 0;
  const di = rateB > 0 ? (rateA < rateB ? rateA / rateB : rateB / rateA) : 0;
  const fp = ready ? fisherExact(a, b, cc, d) : 1;
  const cr = ready ? chiSquare(a, b, cc, d) : { valid: false, stat: 0, p: 1 };
  const zr = ready ? zTest(rateA, n1, rateB, n2) : { valid: false, z: 0, p: 1 };
  const bestP = Math.min(fp, cr.valid ? cr.p : 1, zr.valid ? zr.p : 1);
  return {
    a, b, c: cc, d, n1, n2, ready, rateA, rateB, di, fp, cr, zr, bestP,
    q: ready ? classifyDI(di) : null,
    lowG: rateA < rateB ? c.aName : c.bName,
    hiG: rateA < rateB ? c.bName : c.aName,
    lowR: Math.min(rateA, rateB),
    hiR: Math.max(rateA, rateB),
    small: ready && (n1 < 30 || n2 < 30),
    aName: c.aName, bName: c.bName,
    name: (c.label && c.label.trim()) ? c.label.trim() : (c.aName + " vs " + c.bName),
  };
}

function reviewRec(fc) {
  if (fc === "Critical") return {
    days: 30, interval: "After remediation, before return to service",
    text: "Re-test after remediation and before the system returns to service. Once cleared, test quarterly for at least two consecutive cycles before resuming the standard cadence.",
  };
  if (fc === "Moderate") return {
    days: 90, interval: "Within 90 days",
    text: "Re-test within 90 days to verify the remediation plan has closed the disparity. Then resume the system's scheduled cadence.",
  };
  if (fc === "Low") return {
    days: 180, interval: "Next scheduled cycle",
    text: "Re-test at the next scheduled cycle. If a Low finding persists across two consecutive cycles, escalate the finding to Moderate.",
  };
  return {
    days: 180, interval: "Standard cadence",
    text: "Continue your organization\u2019s standard testing schedule. A common baseline is quarterly for high-impact systems and semi-annually for elevated systems.",
  };
}

function nextDue(testDate, days) {
  const base = testDate ? new Date(testDate + "T00:00:00") : new Date();
  base.setDate(base.getDate() + days);
  return base.toLocaleDateString();
}

/*
 * Plain-language summary, generated from the figures alone. Deterministic and
 * local: no model, no API key, no network call. The same inputs always produce
 * the same wording, which is what an auditable governance record needs.
 */
function buildNarrative(d) {
  const { stats, adj, driving, contextMax, context, overall, sysName, testType, testDate } = d;
  const ready = stats.filter(s => s.ready);
  if (ready.length === 0) return [];
  const paras = [];
  const system = sysName || "the system under test";
  const m = ready.length;

  paras.push(
    "This " + testType.toLowerCase() + " bias test examined " + system + " across " +
    (m === 1 ? "one comparison" : m + " comparisons") + ": " + listJoin(ready.map(s => s.name)) + "."
  );

  const per100 = Math.round(driving.di * 100);
  let gap = (m === 1 ? "" : "The comparison driving the result is " + driving.name + ". ") +
    driving.lowG + " received favourable outcomes at " + pct(driving.lowR) + " against " +
    pct(driving.hiR) + " for " + driving.hiG + ". The disparate impact ratio is " + driving.di.toFixed(3) +
    ", which means that for every 100 " + driving.hiG + " receiving a favourable outcome at their group's rate, roughly " +
    per100 + " " + driving.lowG + " did. ";
  if (driving.di >= 0.90) gap += "That sits above 0.90 and shows no meaningful separation between the two groups under the four-fifths rule.";
  else if (driving.di >= 0.80) gap += "That clears the 0.80 four-fifths threshold, though the margin is narrow enough to watch across future cycles.";
  else gap += "That falls below the 0.80 threshold set by the four-fifths rule, placing the quantitative finding in the " + driving.q + " range and marking the disparity as one requiring explanation.";
  paras.push(gap);

  if (m > 1) {
    const flagged = ready.filter(s => s.di < 0.80);
    const clear = ready.filter(s => s.di >= 0.80);
    let other = "Across all " + m + " comparisons, " +
      (flagged.length === 0 ? "none fell below the 0.80 threshold"
        : flagged.length === m ? "every one fell below the 0.80 threshold"
          : flagged.length + " fell below the 0.80 threshold (" + listJoin(flagged.map(s => s.name)) + ") and " + clear.length + " cleared it") + ". ";
    const smalls = ready.filter(s => s.small);
    if (smalls.length > 0) {
      other += listJoin(smalls.map(s => s.name)) + (smalls.length === 1 ? " has" : " have") +
        " a group under 30 observations, so " + (smalls.length === 1 ? "that result carries" : "those results carry") + " limited statistical power. ";
    }
    other += "The overall classification takes the most severe result across comparisons, since a disparity affecting one group is not offset by parity in another.";
    paras.push(other);
  }

  let sig;
  if (driving.bestP <= 0.01) sig = "Statistical testing puts the probability of a gap this size arising by chance at " + fmtP(driving.bestP) + ". That is a highly significant result, so the disparity should be treated as real rather than as noise, and it raises the finding by one level.";
  else if (driving.bestP <= 0.05) sig = "Statistical testing puts the probability of a gap this size arising by chance at " + fmtP(driving.bestP) + ", significant at the conventional 0.05 threshold. The quantitative finding stands as measured.";
  else if (driving.bestP <= 0.10) sig = "Statistical testing returns " + fmtP(driving.bestP) + ", a marginal result. The gap may be real, but this sample cannot establish it confidently, so the finding is treated cautiously and a follow-up test with more data is warranted.";
  else sig = "Statistical testing returns " + fmtP(driving.bestP) + ", which is not significant. A gap this size could reasonably occur by chance at these sample sizes, so the finding is reduced by one level.";
  if (driving.small) sig += " One or both groups in this comparison have fewer than 30 observations, so Fisher's Exact Test carried the result and the test has limited power.";
  paras.push(sig);

  if (m > 1) {
    const a = adj[driving.index];
    const sigRaw = ready.filter(s => s.bestP <= 0.05).length;
    paras.push(
      "Running " + m + " comparisons raises the chance that at least one crosses the 0.05 threshold by chance alone. " +
      "A Holm-Bonferroni adjustment is reported alongside each raw p-value for that reason. " +
      (a != null ? ("For the driving comparison the raw p-value of " + fmtP(driving.bestP) + " adjusts to " + fmtP(a) + ". ") : "") +
      sigRaw + " of " + m + " comparisons " + (sigRaw === 1 ? "is" : "are") + " significant on the raw p-value. " +
      "The adjustment is advisory and does not change the classification, which continues to follow the raw p-value under your organization\u2019s classification policy. " +
      "It is shown so reviewers can weigh how many tests were run when reading the result."
    );
  }

  const drivers = [];
  if (context.decisionType === 2) drivers.push("it directly determines outcomes rather than informing them");
  else if (context.decisionType === 1) drivers.push("it influences a human decision");
  if (context.affectedPop === 2) drivers.push("it affects vulnerable populations");
  else if (context.affectedPop === 1) drivers.push("it affects the general public");
  if (context.scale === 2) drivers.push("it operates above 10,000 decisions a year");
  else if (context.scale === 1) drivers.push("it operates at 100 to 10,000 decisions a year");
  if (context.reversibility === 2) drivers.push("its outcomes are difficult to reverse");
  else if (context.reversibility === 1) drivers.push("its outcomes are reversible only with effort");

  if (contextMax === 0) paras.push("The context assessment placed the system in the standard band on every factor, so no severity adjustment was applied.");
  else paras.push(
    "The context assessment raised the severity by " + (contextMax === 1 ? "one level" : "two levels") +
    " because " + listJoin(drivers) + ". The same numerical gap carries more weight in a setting like this than it would in a low-stakes one. " +
    "Context is a property of the system, so the same adjustment applies to every comparison."
  );

  const rec = reviewRec(overall);
  let concl;
  if (overall === "Critical") concl = "Taken together, the overall finding is Critical. A disparity of this size in a setting of this consequence warrants suspending the system pending remediation, with governance review before it returns to service and notification to leadership. " + rec.text;
  else if (overall === "Moderate") concl = "Taken together, the overall finding is Moderate. The system can continue operating with enhanced monitoring, but a remediation plan is due within 30 days and a human review checkpoint should be added for affected decisions. " + rec.text;
  else if (overall === "Low") concl = "Taken together, the overall finding is Low. No interruption to the system is required, but the finding and its root cause should be documented and carried into the next review. " + rec.text;
  else concl = "Taken together, the overall finding is None. No disparity requiring action was identified in " + (m === 1 ? "this comparison" : "any of these comparisons") + ". Document the result and continue monitoring. " + rec.text;
  concl += " The target date for the next test is " + nextDue(testDate, rec.days) + ".";
  paras.push(concl);

  paras.push("This summary is generated from the figures above. It is a screening result rather than a legal conclusion, and the four-fifths rule is a threshold for further inquiry rather than a definitive standard. Involve counsel before acting on a finding.");
  return paras;
}

export default function BiasTestingTool() {
  const [step, setStep] = useState(0);
  const [aheadOpen, setAheadOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [agencyName, setAgencyName] = useState("");
  const [sysName, setSysName] = useState("");
  const [regId, setRegId] = useState("");
  const [owner, setOwner] = useState("");
  const [testDate, setTestDate] = useState(new Date().toISOString().slice(0, 10));
  const [testType, setTestType] = useState("Pre-Deployment");
  const [comparisons, setComparisons] = useState([newComparison()]);
  const [context, setContext] = useState({ decisionType: 0, affectedPop: 0, scale: 0, reversibility: 0 });

  const setComp = (i, field, value) => {
    setComparisons(prev => prev.map((c, k) => k === i ? { ...c, [field]: value } : c));
  };
  const addComparison = () => setComparisons(prev => [...prev, newComparison()]);
  const removeComparison = (i) => setComparisons(prev => prev.length > 1 ? prev.filter((c, k) => k !== i) : prev);

  const stats = useMemo(() => comparisons.map(compStats), [comparisons]);
  const readyStats = stats.filter(s => s.ready);
  const hasData = readyStats.length > 0;
  const contextMax = Math.max(context.decisionType, context.affectedPop, context.scale, context.reversibility);

  const adj = useMemo(() => {
    const idx = [], ps = [];
    stats.forEach((s, i) => { if (s.ready) { idx.push(i); ps.push(s.bestP); } });
    const a = holmAdjust(ps);
    const out = new Array(stats.length).fill(null);
    idx.forEach((i, k) => { out[i] = a[k]; });
    return out;
  }, [stats]);

  const finalClassFor = useCallback((s) => {
    if (!s.ready) return null;
    if (s.q === "None") return "None";
    let i = LEVELS.indexOf(s.q);
    i = Math.min(i + contextMax, 3);
    if (s.bestP > 0.10) i = Math.max(i - 1, 0);
    else if (s.bestP <= 0.01) i = Math.min(i + 1, 3);
    return LEVELS[i];
  }, [contextMax]);

  const overall = useMemo(() => {
    let best = -1;
    stats.forEach(s => { const f = finalClassFor(s); if (f) best = Math.max(best, LEVELS.indexOf(f)); });
    return best < 0 ? null : LEVELS[best];
  }, [stats, finalClassFor]);

  const driving = useMemo(() => {
    let out = null, outIdx = -1, bestIdx = -1, bestDi = 2;
    stats.forEach((s, i) => {
      if (!s.ready) return;
      const li = LEVELS.indexOf(finalClassFor(s));
      if (li > bestIdx || (li === bestIdx && s.di < bestDi)) { bestIdx = li; bestDi = s.di; out = s; outIdx = i; }
    });
    return out ? { ...out, index: outIdx } : null;
  }, [stats, finalClassFor]);

  const narrative = useMemo(() => {
    if (!hasData) return [];
    return buildNarrative({ stats, adj, driving, contextMax, context, overall, sysName, testType, testDate });
  }, [hasData, stats, adj, driving, contextMax, context, overall, sysName, testType, testDate]);

  const rec = hasData ? reviewRec(overall) : null;
  const oc = hasData ? SEV_COLORS[overall] : NEUTRAL;
  const canProceed = () => { if (step === 0) return !!(sysName && owner); if (step === 1) return hasData; return true; };

  const copyNarrative = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(narrative.join("\n\n")).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  const handleSavePdf = () => {
    const w = window.open("", "_blank");
    w.document.write(
      '<html><head><title>Bias Test Report - ' + sysName + '</title><style>' +
      '*,*::before,*::after{-webkit-print-color-adjust:exact !important;print-color-adjust:exact !important}' +
      '@media print{@page{margin:14mm}}' +
      'body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;color:#1a1a1a;font-size:12.5px;line-height:1.55}' +
      'table{border-collapse:collapse;width:100%;margin:6px 0}th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;font-size:11.5px}' +
      'th{background:#f3f4f6}.font-mono{font-family:monospace}.font-medium{font-weight:500}.font-bold{font-weight:700}' +
      '.text-center{text-align:center}.text-xs{font-size:11px}.text-sm{font-size:12.5px}.text-lg{font-size:17px}' +
      '.text-gray-500{color:#6b7280}.text-gray-600{color:#4b5563}.text-gray-700{color:#374151}.text-gray-800{color:#1f2937}' +
      '.text-gray-900{color:#111827}.text-gray-400{color:#9ca3af}.bg-gray-50{background:#f9fafb}' +
      '.uppercase{text-transform:uppercase}.tracking-wider{letter-spacing:.05em}div{margin-bottom:4px}p{margin:0 0 6px}' +
      '</style></head><body>' + document.getElementById("report-content").innerHTML + "</body></html>"
    );
    w.document.close();
    w.print();
  };

  const inputStyle = "w-full px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent";
  const selectStyle = "w-full px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400";
  const labelStyle = "block text-sm font-medium text-gray-700 mb-1";

  const previewBanner = (
    <div className="rounded-lg border border-amber-400 border-l-4 bg-amber-50 p-3 text-sm text-amber-900">
      <span className="block font-semibold mb-0.5">Preview only {DASH} these are not your results.</span>
      You are looking ahead before entering data. Every figure on this page is a placeholder produced from empty
      counts, including any classification shown. Go to step 2, Comparisons, and enter your outcome counts to see
      real results.
    </div>
  );

  const ComparisonTable = ({ showAdj }) => (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-100 text-xs text-gray-700">
            <th className="px-2 py-2 text-left font-medium">Comparison</th>
            <th className="px-2 py-2 text-left font-medium">Rates</th>
            <th className="px-2 py-2 text-left font-medium">DI ratio</th>
            <th className="px-2 py-2 text-left font-medium">Quantitative</th>
            {showAdj && <th className="px-2 py-2 text-left font-medium">p (raw)</th>}
            {showAdj && <th className="px-2 py-2 text-left font-medium">p (Holm)</th>}
            <th className="px-2 py-2 text-left font-medium">Classification</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s, i) => {
            if (!s.ready) return (
              <tr key={i} className="border-t border-gray-100 text-gray-400 italic">
                <td className="px-2 py-2">{s.name}</td>
                <td className="px-2 py-2" colSpan={showAdj ? 6 : 3}>Incomplete, no counts entered</td>
              </tr>
            );
            const f = finalClassFor(s);
            const sv = SEV_COLORS[f];
            return (
              <tr key={i} className="border-t border-gray-100">
                <td className="px-2 py-2 text-gray-800">{s.name}</td>
                <td className="px-2 py-2 font-mono">{pct(s.rateA)} / {pct(s.rateB)}</td>
                <td className="px-2 py-2 font-mono">{s.di.toFixed(3)}</td>
                <td className="px-2 py-2">{s.q}</td>
                {showAdj && <td className="px-2 py-2 font-mono">{fmtP(s.bestP)}</td>}
                {showAdj && <td className="px-2 py-2 font-mono">{adj[i] != null ? fmtP(adj[i]) : DASH}</td>}
                <td className="px-2 py-2">
                  <span className="inline-block px-2 py-0.5 rounded-full text-xs font-semibold border"
                    style={{ backgroundColor: sv.bg, color: sv.text, borderColor: sv.border }}>{f}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  const renderStep = () => {
    switch (step) {
      case 0: return (<div className="space-y-4">
        <p className="text-sm text-gray-600">Identify the AI system under test and the organization conducting the assessment. These apply to every comparison.</p>
        <div><label className={labelStyle}>Organization name</label><input className={inputStyle} value={agencyName} onChange={e => setAgencyName(e.target.value)} placeholder="Your organization" /></div>
        <div><label className={labelStyle}>AI system name *</label><input className={inputStyle} value={sysName} onChange={e => setSysName(e.target.value)} placeholder="e.g., Benefits Eligibility Review Tool" /></div>
        <div><label className={labelStyle}>System ID or register ID</label><input className={inputStyle} value={regId} onChange={e => setRegId(e.target.value)} placeholder="e.g., AI-2026-001" /></div>
        <div><label className={labelStyle}>AI system owner *</label><input className={inputStyle} value={owner} onChange={e => setOwner(e.target.value)} placeholder="Name and title" /></div>
        <div className="grid grid-cols-2 gap-4">
          <div><label className={labelStyle}>Test date</label><input type="date" className={inputStyle} value={testDate} onChange={e => setTestDate(e.target.value)} /></div>
          <div><label className={labelStyle}>Test type</label><select className={selectStyle} value={testType} onChange={e => setTestType(e.target.value)}><option>Pre-Deployment</option><option>Recurring</option><option>Incident-Triggered</option></select></div>
        </div>
      </div>);

      case 1: return (<div className="space-y-4">
        <p className="text-sm text-gray-600">Each comparison is one pair of groups tested on one outcome. Groups should be defined by protected class (e.g., race, ethnicity, sex, age, disability status) per applicable civil rights requirements. Add a comparison for each pair the use case requires.</p>
        {comparisons.map((c, i) => {
          const s = stats[i];
          return (
            <div key={i} className="rounded-xl border border-gray-200 bg-gray-50/40 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-blue-700">Comparison {i + 1}</span>
                {comparisons.length > 1 && (
                  <button onClick={() => removeComparison(i)} className="text-xs text-gray-400 hover:text-red-700 hover:bg-red-50 px-2 py-0.5 rounded">Remove</button>
                )}
              </div>
              <div><label className={labelStyle}>Label (optional)</label><input className={inputStyle} value={c.label} onChange={e => setComp(i, "label", e.target.value)} placeholder="e.g., Race: Black vs White" /></div>
              <div className="grid grid-cols-2 gap-4">
                <div><label className={labelStyle}>Group A name</label><input className={inputStyle} value={c.aName} onChange={e => setComp(i, "aName", e.target.value)} placeholder="e.g., Hispanic" /></div>
                <div><label className={labelStyle}>Group B name</label><input className={inputStyle} value={c.bName} onChange={e => setComp(i, "bName", e.target.value)} placeholder="e.g., Non-Hispanic" /></div>
              </div>
              <div className="rounded-xl border border-gray-200 overflow-hidden bg-white">
                <table className="w-full text-sm">
                  <thead><tr className="bg-gray-100">
                    <th className="px-3 py-2 text-left font-medium text-gray-700"></th>
                    <th className="px-3 py-2 text-center font-medium text-gray-700">Positive outcome<br /><span className="font-normal text-gray-500 text-xs">(approved, selected, etc.)</span></th>
                    <th className="px-3 py-2 text-center font-medium text-gray-700">Negative outcome<br /><span className="font-normal text-gray-500 text-xs">(denied, not selected, etc.)</span></th>
                    <th className="px-3 py-2 text-center font-medium text-gray-700">Total</th>
                  </tr></thead>
                  <tbody>
                    <tr className="border-t border-gray-100">
                      <td className="px-3 py-2 font-medium text-gray-700">{c.aName}</td>
                      <td className="px-3 py-2"><input type="number" min="0" className={inputStyle + " text-center"} value={c.aPos} onChange={e => setComp(i, "aPos", e.target.value)} placeholder="0" /></td>
                      <td className="px-3 py-2"><input type="number" min="0" className={inputStyle + " text-center"} value={c.aNeg} onChange={e => setComp(i, "aNeg", e.target.value)} placeholder="0" /></td>
                      <td className="px-3 py-2 text-center font-medium text-gray-700">{s.n1}</td>
                    </tr>
                    <tr className="border-t border-gray-100">
                      <td className="px-3 py-2 font-medium text-gray-700">{c.bName}</td>
                      <td className="px-3 py-2"><input type="number" min="0" className={inputStyle + " text-center"} value={c.bPos} onChange={e => setComp(i, "bPos", e.target.value)} placeholder="0" /></td>
                      <td className="px-3 py-2"><input type="number" min="0" className={inputStyle + " text-center"} value={c.bNeg} onChange={e => setComp(i, "bNeg", e.target.value)} placeholder="0" /></td>
                      <td className="px-3 py-2 text-center font-medium text-gray-700">{s.n2}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              {s.ready && <p className="text-xs text-gray-500">Total observations: {s.n1 + s.n2}.{s.small ? " One or both groups have fewer than 30 observations, so Fisher's Exact Test will carry this comparison." : ""}</p>}
            </div>
          );
        })}
        <button onClick={addComparison} className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors">+ Add comparison</button>
        {comparisons.length > 1 && <p className="text-xs text-gray-500">Running several comparisons raises the chance that one crosses the significance threshold by chance. The Statistics step reports a multiplicity-adjusted p-value alongside the raw one for that reason.</p>}
      </div>);

      case 2: return (<div className="space-y-5">
        {!hasData && previewBanner}
        <p className="text-sm text-gray-600">Calculated from your contingency tables. The Disparate Impact Ratio uses the four-fifths rule from the Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4).</p>
        {!hasData
          ? <p className="text-sm text-gray-600">Selection rates and disparate impact ratios appear here once you enter outcome counts on step 2, Comparisons.</p>
          : (<>
            <ComparisonTable showAdj={false} />
            <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700 space-y-2">
              <div className="font-medium">Interpretation</div>
              <p>{driving.lowG} has a selection rate of {pct(driving.lowR)}, compared to {driving.hiG} at {pct(driving.hiR)}, in {driving.name}. The ratio is {driving.di.toFixed(3)}, which is {driving.di >= 0.80 ? "at or above" : "below"} the 0.80 threshold.</p>
              {driving.di < 0.80 && <p className="font-medium" style={{ color: SEV_COLORS[driving.q].text }}>This is the comparison driving the result and it falls in the {driving.q} zone. Statistical significance testing on the next step will confirm whether the disparity is likely real or could be due to chance.</p>}
            </div>
            <div className="rounded-xl border border-gray-200 p-4 text-sm">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Threshold reference</div>
              <table className="w-full text-sm"><thead><tr className="text-gray-500 text-xs">
                <th className="text-left py-1">Metric</th><th className="text-center py-1">Pass (&ge;0.90)</th><th className="text-center py-1">Low (0.80-0.89)</th><th className="text-center py-1">Moderate (0.65-0.79)</th><th className="text-center py-1">Critical (&lt;0.65)</th>
              </tr></thead><tbody><tr><td className="py-1 text-gray-700">DI Ratio</td>
                {LEVELS.map(cl => <td key={cl} className="text-center py-1"><span className={"inline-block w-5 h-5 rounded-full " + (driving.q === cl ? "ring-2 ring-offset-1" : "opacity-30")} style={{ backgroundColor: SEV_COLORS[cl].border }} /></td>)}
              </tr></tbody></table>
            </div>
          </>)}
      </div>);

      case 3: {
        const m = readyStats.length;
        return (<div className="space-y-5">
          {!hasData && previewBanner}
          <p className="text-sm text-gray-600">All three standard significance tests run on every comparison. Each determines whether an observed disparity could have occurred by chance.</p>
          {!hasData
            ? <p className="text-sm text-gray-600">Significance results appear here once you enter outcome counts on step 2, Comparisons.</p>
            : (<>
              <ComparisonTable showAdj={true} />
              {stats.map((s, i) => {
                if (!s.ready) return null;
                const tests = [
                  { n: "Fisher's Exact", p: s.fp, valid: true, stat: null, sl: null },
                  { n: "Chi-Square", p: s.cr.p, valid: s.cr.valid, stat: s.cr.stat, sl: "\u03C7\u00B2" },
                  { n: "Two-Proportion Z", p: s.zr.p, valid: s.zr.valid, stat: s.zr.z, sl: "Z" },
                ];
                return (
                  <div key={i} className="rounded-xl border border-gray-200 p-4 space-y-2">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="font-medium text-gray-800">{s.name}</div>
                      <span className={"px-3 py-1 rounded-full text-xs font-medium " + (s.bestP <= 0.05 ? "bg-red-100 text-red-800" : s.bestP <= 0.10 ? "bg-amber-100 text-amber-800" : "bg-green-100 text-green-800")}>best p = {fmtP(s.bestP)} {pLabel(s.bestP)}</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {tests.map(t => (
                        <span key={t.n} className={"rounded px-2 py-0.5 text-xs font-mono bg-gray-100 text-gray-700 " + (t.valid ? "" : "opacity-50")}>
                          {t.n}: {t.valid ? fmtP(t.p) + (t.stat !== null ? " (" + t.sl + " = " + t.stat.toFixed(3) + ")" : "") : "not applicable"}
                        </span>
                      ))}
                    </div>
                    <div className="text-xs text-gray-500">Holm-adjusted p = {adj[i] != null ? fmtP(adj[i]) : DASH}{s.small ? ". One or both groups are under 30 observations, so Fisher's Exact carries this comparison and power is limited." : "."}</div>
                  </div>
                );
              })}
              <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700">
                <div className="font-medium mb-1">On running {m} comparison{m === 1 ? "" : "s"}</div>
                <p>{m === 1
                  ? "A single comparison needs no multiplicity adjustment. The raw p-value governs the finding."
                  : "Each additional comparison raises the chance that one crosses the 0.05 threshold by chance alone. With " + m + " comparisons that chance is roughly " + Math.round((1 - Math.pow(0.95, m)) * 100) + " percent if no real disparity exists anywhere. The Holm-Bonferroni column adjusts for this."}</p>
                <p className="mt-2"><span className="font-medium">The adjustment is advisory.</span> Classification continues to follow the raw p-value. A finding is never reduced solely because other comparisons were tested. The adjusted value is shown so reviewers can weigh how many tests were run when reading the result.</p>
              </div>
            </>)}
        </div>);
      }

      case 4: return (<div className="space-y-5">
        <p className="text-sm text-gray-600">Assess the real-world context of the AI system. The highest-scoring factor determines the severity adjustment. Context is a property of the system, so the same adjustment applies to every comparison.</p>
        {Object.entries(CONTEXT_LABELS).map(([key, cfg]) => (
          <div key={key} className="space-y-2">
            <label className={labelStyle}>{cfg.label}</label>
            <div className="grid grid-cols-3 gap-2">
              {cfg.options.map((opt, i) => (
                <button key={i} onClick={() => setContext(prev => ({ ...prev, [key]: cfg.scores[i] }))}
                  className={"px-3 py-3 rounded-lg text-xs text-left border transition-all " + (context[key] === cfg.scores[i] ? "border-blue-500 bg-blue-50 text-blue-800 ring-1 ring-blue-300" : "border-gray-200 bg-white text-gray-700 hover:border-gray-400")}>{opt}</button>
              ))}
            </div>
          </div>
        ))}
        <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700"><span className="font-medium">Highest context factor: </span>{contextMax === 0 ? "Standard (no adjustment)" : contextMax === 1 ? "Elevated (+1 severity level)" : "High (+2 severity levels)"}</div>
      </div>);

      case 5: return (<div className="space-y-5">
        {!hasData && previewBanner}
        <p className="text-sm text-gray-600">Each comparison is classified on its own disparity and significance, with the shared context adjustment. The overall result takes the most severe.</p>
        <div className="rounded-2xl border-2 p-6 text-center space-y-2" style={{ borderColor: oc.border, backgroundColor: oc.bg }}>
          <div className="text-sm font-medium uppercase tracking-wider" style={{ color: oc.text }}>{hasData ? "Overall classification" : "Overall classification: not yet calculated"}</div>
          <div className="text-5xl font-bold" style={{ color: oc.text }}>{hasData ? overall : DASH}</div>
          {hasData && readyStats.length > 1 && <div className="text-xs" style={{ color: oc.text }}>Most severe of {readyStats.length} comparisons</div>}
        </div>
        {hasData && <ComparisonTable showAdj={true} />}
        {hasData && (
          <div className="rounded-xl border border-gray-200 p-4 text-sm space-y-2">
            <div className="flex items-center justify-between">
              <div className="font-medium text-gray-800">Plain-language summary</div>
              <button onClick={copyNarrative} className="px-3 py-1 rounded-md border border-gray-300 text-xs font-medium text-gray-700 hover:bg-gray-50 transition-colors">{copied ? "Copied" : "Copy"}</button>
            </div>
            <div className="space-y-2 text-gray-700">{narrative.map((p, i) => <p key={i}>{p}</p>)}</div>
          </div>
        )}
        {hasData && (
          <div className="rounded-xl border border-gray-200 p-4 text-sm text-gray-700">
            <div className="font-medium mb-1">Re-review frequency</div>
            <p><span className="font-medium">{rec.interval}</span> {DASH} target date {nextDue(testDate, rec.days)}.</p>
            <p className="mt-1">{rec.text}</p>
          </div>
        )}
        <div className="rounded-xl border border-gray-200 p-4 text-sm text-gray-700"><div className="font-medium mb-1">Recommended response:</div>
          {!hasData && <p>The recommended response depends on the overall classification, which appears once comparison data is entered.</p>}
          {hasData && overall === "Critical" && <p>Immediate suspension of AI system pending remediation. Governance body review required before reactivation. Leadership notification.</p>}
          {hasData && overall === "Moderate" && <p>Remediation plan due within 30 days. Add human review checkpoint for affected decisions. Notify governance body. System may continue with enhanced monitoring.</p>}
          {hasData && overall === "Low" && <p>Document finding and root cause analysis. Monitor at next scheduled test. No system interruption required. If Low persists across two consecutive test cycles, escalate to Moderate.</p>}
          {hasData && overall === "None" && <p>Document results. Continue standard monitoring schedule.</p>}
        </div>
      </div>);

      case 6: {
        if (!hasData) return (<div className="space-y-4">
          <p className="text-sm text-gray-600">The report is generated from your test data.</p>
          <div className="rounded-lg border border-amber-400 border-l-4 bg-amber-50 p-3 text-sm text-amber-900">
            <span className="block font-semibold mb-0.5">No report yet {DASH} comparison data is required.</span>
            A bias testing report cannot be produced without outcome counts. Enter at least one comparison on step 2, Comparisons, and the report will be generated here.
          </div>
          <div className="flex gap-3">
            <button onClick={() => setStep(1)} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors">Go to Comparisons</button>
            <button onClick={() => setStep(0)} className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors">Start new test</button>
          </div>
        </div>);

        const m = readyStats.length;
        return (<div className="space-y-4">
          <p className="text-sm text-gray-600">Review the summary below, then save it with your AI system&rsquo;s governance records.</p>
          <div className="rounded-xl border border-gray-200 p-5 space-y-4 text-sm bg-white" id="report-content">
            <div className="text-center space-y-1 border-b border-gray-200 pb-4">
              <div className="text-lg font-bold text-gray-900">AI Bias Testing Report</div>
              {agencyName && <div className="text-sm text-gray-600">{agencyName}</div>}
              <div className="text-xs text-gray-400">Generated {new Date().toLocaleDateString()}</div>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              <div><span className="text-gray-500">AI System:</span> <span className="font-medium">{sysName}</span></div>
              <div><span className="text-gray-500">System ID:</span> <span className="font-medium">{regId || "N/A"}</span></div>
              <div><span className="text-gray-500">Owner:</span> <span className="font-medium">{owner}</span></div>
              <div><span className="text-gray-500">Test Date:</span> <span className="font-medium">{testDate}</span></div>
              <div><span className="text-gray-500">Test Type:</span> <span className="font-medium">{testType}</span></div>
              <div><span className="text-gray-500">Comparisons tested:</span> <span className="font-medium">{m}</span></div>
            </div>
            <div className="border-t border-gray-200 pt-3">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Comparisons</div>
              <ComparisonTable showAdj={true} />
            </div>
            <div className="border-t border-gray-200 pt-3">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Contingency tables</div>
              <table className="w-full text-sm border border-gray-200">
                <thead><tr className="bg-gray-50">
                  <th className="px-2 py-2 text-left">Comparison</th><th className="px-2 py-2 text-left">Group</th>
                  <th className="px-2 py-2 text-center">Positive</th><th className="px-2 py-2 text-center">Negative</th>
                  <th className="px-2 py-2 text-center">Total</th><th className="px-2 py-2 text-center">Rate</th>
                </tr></thead>
                <tbody>
                  {stats.map((s, i) => !s.ready ? null : (
                    <React.Fragment key={i}>
                      <tr className="border-t"><td className="px-2 py-2" rowSpan={2}>{s.name}</td><td className="px-2 py-2">{s.aName}</td><td className="px-2 py-2 text-center font-mono">{s.a}</td><td className="px-2 py-2 text-center font-mono">{s.b}</td><td className="px-2 py-2 text-center font-mono">{s.n1}</td><td className="px-2 py-2 text-center font-mono">{pct(s.rateA)}</td></tr>
                      <tr><td className="px-2 py-2">{s.bName}</td><td className="px-2 py-2 text-center font-mono">{s.c}</td><td className="px-2 py-2 text-center font-mono">{s.d}</td><td className="px-2 py-2 text-center font-mono">{s.n2}</td><td className="px-2 py-2 text-center font-mono">{pct(s.rateB)}</td></tr>
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="border-t border-gray-200 pt-3">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Context assessment</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <div><span className="text-gray-500">Decision type:</span> <span className="font-medium">{["Informational", "Influences human decision", "Directly determines outcomes"][context.decisionType]}</span></div>
                <div><span className="text-gray-500">Affected population:</span> <span className="font-medium">{["Internal staff only", "General public", "Vulnerable populations"][context.affectedPop]}</span></div>
                <div><span className="text-gray-500">Scale of impact:</span> <span className="font-medium">{["< 100/year", "100 - 10,000/year", "> 10,000/year"][context.scale]}</span></div>
                <div><span className="text-gray-500">Reversibility:</span> <span className="font-medium">{["Easily reversed", "Reversible with effort", "Difficult to reverse"][context.reversibility]}</span></div>
                <div><span className="text-gray-500">Severity adjustment:</span> <span className="font-medium">{contextMax === 0 ? "None" : "+" + contextMax + " level" + (contextMax > 1 ? "s" : "")}</span></div>
              </div>
            </div>
            <div className="border-t border-gray-200 pt-3 flex items-center gap-3 flex-wrap">
              <span className="text-gray-500 text-xs">Overall Classification:</span>
              <span className="px-4 py-2 rounded-lg font-bold text-lg" style={{ backgroundColor: oc.bg, color: oc.text, border: "2px solid " + oc.border }}>{overall}</span>
              {m > 1 && <span className="text-gray-500 text-xs">most severe of {m} comparisons</span>}
            </div>
            <div className="border-t border-gray-200 pt-3">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Summary</div>
              <div className="text-gray-700 space-y-2">{narrative.map((p, i) => <p key={i}>{p}</p>)}</div>
            </div>
            <div className="border-t border-gray-200 pt-3">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Re-review frequency</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <div><span className="text-gray-500">Recommended interval:</span> <span className="font-medium">{rec.interval}</span></div>
                <div><span className="text-gray-500">Target next test date:</span> <span className="font-medium">{nextDue(testDate, rec.days)}</span></div>
              </div>
              <div className="text-gray-700 mt-1">{rec.text}</div>
            </div>
            <div className="border-t border-gray-200 pt-3 text-xs text-gray-500 space-y-1">
              <div>Methodology: Disparate Impact Ratio per Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4). Statistical significance via Fisher&rsquo;s Exact Test, Chi-Square Test of Independence, and Two-Proportion Z-Test, with the strongest applicable result used. Holm-Bonferroni adjustment reported for multiple comparisons and advisory only. Context assessment per NIST AI Risk Management Framework 1.0 (AI 100-1) Measure function.</div>
              <div>References: NIST AI RMF 1.0, NIST SP 1270, EEOC Technical Assistance on AI and Title VII (May 2023).</div>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={handleSavePdf} className="px-4 py-2 rounded-lg bg-gray-800 text-white text-sm font-medium hover:bg-gray-700 transition-colors">Save as PDF</button>
            <button onClick={() => setStep(0)} className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors">Start new test</button>
          </div>
          <p className="text-xs text-gray-500 text-center">Your browser&rsquo;s print dialog will open. Choose <span className="font-medium">Save as PDF</span> as the destination, and keep background graphics enabled so the classification colour is retained.</p>
        </div>);
      }

      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4" style={{ fontFamily: "'DM Sans',system-ui,sans-serif" }}>
      <style>{`
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
        input[type="number"] { -moz-appearance: textfield; appearance: textfield; }
        @media print {
          *,*::before,*::after { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        }
      `}</style>
      <div className="max-w-3xl mx-auto">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-bold text-gray-900">AI Bias Testing Tool</h1>
          <p className="text-sm text-gray-500 mt-1">Disparate impact analysis for AI systems</p>
          <p className="text-xs text-gray-400 mt-0.5">Based on NIST AI RMF 1.0, EEOC Uniform Guidelines (29 CFR 1607.4), and NIST SP 1270</p>
        </div>

        <div className="flex items-center justify-center gap-1 mb-4 flex-wrap">
          {STEPS.map((s, i) => (
            <div key={i} className="flex items-center">
              <button onClick={() => setStep(i)} className={"flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium transition-all cursor-pointer " + (i === step ? "bg-blue-600 text-white" : i < step ? "bg-blue-100 text-blue-700 hover:bg-blue-200" : "bg-gray-100 text-gray-400 hover:bg-gray-200 hover:text-gray-600")}>
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs" style={{ backgroundColor: i === step ? "rgba(255,255,255,0.2)" : "transparent" }}>{i < step ? "\u2713" : i + 1}</span>
                <span className="hidden sm:inline">{s}</span>
              </button>
              {i < STEPS.length - 1 && <div className={"w-4 h-px mx-0.5 " + (i < step ? "bg-blue-300" : "bg-gray-200")} />}
            </div>
          ))}
        </div>

        <div className="bg-white rounded-2xl border border-gray-200 mb-6 overflow-hidden">
          <button onClick={() => setAheadOpen(v => !v)} className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-blue-700 hover:bg-gray-50 transition-colors" aria-expanded={aheadOpen}>
            <span>What&rsquo;s ahead: the seven steps and what each one needs</span>
            <span>{aheadOpen ? "\u2212" : "+"}</span>
          </button>
          {aheadOpen && (
            <div className="px-5 pb-4 border-t border-gray-100">
              <p className="text-sm text-gray-600 my-3">You can open any step from the bar above to look around. Only Comparisons has to be filled in before the results mean anything.</p>
              <div>
                {STEP_INFO.map(([title, desc], i) => (
                  <div key={i} className="flex gap-3 py-2 text-sm border-b border-gray-50 last:border-b-0">
                    <span className="flex-none w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-xs font-semibold">{i + 1}</span>
                    <div><div className="font-semibold text-gray-900">{title}</div><div className="text-gray-500 leading-snug">{desc}</div></div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 mb-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">{STEPS[step]}</h2>
          {renderStep()}
        </div>

        {step < 6 && (
          <div className="flex justify-between">
            <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className={"px-5 py-2.5 rounded-lg text-sm font-medium transition-colors " + (step === 0 ? "text-gray-300 cursor-not-allowed" : "text-gray-700 border border-gray-300 hover:bg-gray-50")}>Back</button>
            <button onClick={() => setStep(step + 1)} disabled={!canProceed()} className={"px-5 py-2.5 rounded-lg text-sm font-medium transition-colors " + (canProceed() ? "bg-blue-600 text-white hover:bg-blue-700" : "bg-gray-200 text-gray-400 cursor-not-allowed")}>{step === 5 ? "Generate report" : "Continue"}</button>
          </div>
        )}

        <div className="mt-8 text-center text-xs text-gray-400 space-y-1">
          <p>Open source under MIT License. Free for use by anyone.</p>
          <p>Runs entirely in your browser. No data is transmitted anywhere.</p>
          <p>Methodology: NIST AI RMF 1.0, NIST SP 1270, EEOC Uniform Guidelines (29 CFR 1607.4), EEOC Technical Assistance on AI and Title VII.</p>
          <p>Questions, feedback, or contributions: <a href="https://github.com/jasonanwe/bias-testing-app" className="text-blue-500 hover:text-blue-600 underline">jasonanwe/bias-testing-app</a></p>
        </div>
      </div>
    </div>
  );
}
