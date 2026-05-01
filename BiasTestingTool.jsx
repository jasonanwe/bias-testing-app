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
 */

import { useState, useCallback, useMemo } from "react";

const STEPS = ["System Info", "Group Data", "Metrics", "Statistics", "Context", "Classification", "Report"];

function logFactorial(n) {
  let r = 0;
  for (let i = 2; i <= n; i++) r += Math.log(i);
  return r;
}

function fisherExact(a, b, c, d) {
  const n = a + b + c + d;
  const logDenom = logFactorial(a + b) + logFactorial(c + d) + logFactorial(a + c) + logFactorial(b + d) - logFactorial(n);
  const pCutoff = logDenom - (logFactorial(a) + logFactorial(b) + logFactorial(c) + logFactorial(d));
  let pValue = 0;
  const minAC = Math.min(a + b, a + c);
  for (let i = 0; i <= minAC; i++) {
    const j = (a + b) - i;
    const k = (a + c) - i;
    const l = (c + d) - k;
    if (j < 0 || k < 0 || l < 0) continue;
    const logP = logDenom - (logFactorial(i) + logFactorial(j) + logFactorial(k) + logFactorial(l));
    if (logP <= pCutoff + 1e-10) pValue += Math.exp(logP);
  }
  return Math.min(pValue, 1);
}

function chiSquare(a, b, c, d) {
  const n = a + b + c + d;
  const e1 = ((a + b) * (a + c)) / n;
  const e2 = ((a + b) * (b + d)) / n;
  const e3 = ((c + d) * (a + c)) / n;
  const e4 = ((c + d) * (b + d)) / n;
  if (e1 < 5 || e2 < 5 || e3 < 5 || e4 < 5) return { valid: false, stat: 0, p: 1 };
  const chi2 = ((a - e1) ** 2) / e1 + ((b - e2) ** 2) / e2 + ((c - e3) ** 2) / e3 + ((d - e4) ** 2) / e4;
  const p = 1 - chi2CDF(chi2);
  return { valid: true, stat: chi2, p };
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
  if (se === 0) return { valid: true, z: 0, p: 1 };
  const z = (p1 - p2) / se;
  const pVal = 2 * (1 - normalCDF(Math.abs(z)));
  return { valid: true, z, p: pVal };
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

const SEV_COLORS = {
  None: { bg: "#e8f5e9", text: "#1b5e20", border: "#4caf50" },
  Low: { bg: "#fff8e1", text: "#e65100", border: "#ffa000" },
  Moderate: { bg: "#fff3e0", text: "#bf360c", border: "#e65100" },
  Critical: { bg: "#ffebee", text: "#b71c1c", border: "#c62828" },
};

const CONTEXT_LABELS = {
  decisionType: { label: "Decision type", options: ["Informational (Standard)", "Influences human decision (Elevated, +1)", "Directly determines outcomes (High, +2)"], scores: [0, 1, 2] },
  affectedPop: { label: "Affected population", options: ["Internal staff only (Standard)", "General public (Elevated, +1)", "Vulnerable populations (High, +2)"], scores: [0, 1, 2] },
  scale: { label: "Scale of impact", options: ["< 100 decisions/year (Standard)", "100 - 10,000 decisions/year (Elevated, +1)", "> 10,000 decisions/year (High, +2)"], scores: [0, 1, 2] },
  reversibility: { label: "Reversibility", options: ["Easily reversed (Standard)", "Reversible with effort (Elevated, +1)", "Difficult to reverse (High, +2)"], scores: [0, 1, 2] },
};

export default function BiasTestingTool() {
  const [step, setStep] = useState(0);
  const [agencyName, setAgencyName] = useState("");
  const [sysName, setSysName] = useState("");
  const [regId, setRegId] = useState("");
  const [owner, setOwner] = useState("");
  const [testDate, setTestDate] = useState(new Date().toISOString().slice(0, 10));
  const [testType, setTestType] = useState("Pre-Deployment");
  const [groupAName, setGroupAName] = useState("Group A");
  const [groupBName, setGroupBName] = useState("Group B");
  const [groupAPos, setGroupAPos] = useState("");
  const [groupANeg, setGroupANeg] = useState("");
  const [groupBPos, setGroupBPos] = useState("");
  const [groupBNeg, setGroupBNeg] = useState("");
  const [context, setContext] = useState({ decisionType: 0, affectedPop: 0, scale: 0, reversibility: 0 });

  const a = parseInt(groupAPos) || 0, b = parseInt(groupANeg) || 0, c = parseInt(groupBPos) || 0, d = parseInt(groupBNeg) || 0;
  const n1 = a + b, n2 = c + d;
  const rateA = n1 > 0 ? a / n1 : 0, rateB = n2 > 0 ? c / n2 : 0;
  const diRatio = rateB > 0 ? (rateA < rateB ? rateA / rateB : rateB / rateA) : 0;
  const lowerGroup = rateA < rateB ? groupAName : groupBName;
  const higherGroup = rateA < rateB ? groupBName : groupAName;
  const lowerRate = Math.min(rateA, rateB), higherRate = Math.max(rateA, rateB);

  const fisherP = useMemo(() => (n1 > 0 && n2 > 0) ? fisherExact(a, b, c, d) : 1, [a, b, c, d]);
  const chi2Result = useMemo(() => (n1 > 0 && n2 > 0) ? chiSquare(a, b, c, d) : { valid: false, stat: 0, p: 1 }, [a, b, c, d]);
  const zResult = useMemo(() => (n1 > 0 && n2 > 0) ? zTest(rateA, n1, rateB, n2) : { valid: false, z: 0, p: 1 }, [rateA, rateB, n1, n2]);

  const quantClass = classifyDI(diRatio);
  const contextMax = Math.max(context.decisionType, context.affectedPop, context.scale, context.reversibility);
  const bestP = Math.min(fisherP, chi2Result.valid ? chi2Result.p : 1, zResult.valid ? zResult.p : 1);

  const getFinalClassification = useCallback(() => {
    if (quantClass === "None") return "None";
    const levels = ["None", "Low", "Moderate", "Critical"];
    let idx = levels.indexOf(quantClass);
    idx = Math.min(idx + contextMax, 3);
    if (bestP > 0.10) idx = Math.max(idx - 1, 0);
    else if (bestP <= 0.01) idx = Math.min(idx + 1, 3);
    return levels[idx];
  }, [quantClass, contextMax, bestP]);

  const finalClass = getFinalClassification();
  const fc = SEV_COLORS[finalClass];
  const canProceed = () => { if (step === 0) return sysName && owner; if (step === 1) return a + b > 0 && c + d > 0; return true; };

  const inputStyle = "w-full px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent";
  const selectStyle = "w-full px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400";
  const labelStyle = "block text-sm font-medium text-gray-700 mb-1";

  const renderStep = () => {
    switch (step) {
      case 0: return (<div className="space-y-4">
        <p className="text-sm text-gray-600">Identify the AI system under test and the organization conducting the assessment.</p>
        <div><label className={labelStyle}>Organization name</label><input className={inputStyle} value={agencyName} onChange={e => setAgencyName(e.target.value)} placeholder="e.g., Department of Health and Human Services" /></div>
        <div><label className={labelStyle}>AI system name *</label><input className={inputStyle} value={sysName} onChange={e => setSysName(e.target.value)} placeholder="e.g., Benefits Eligibility Review Tool" /></div>
        <div><label className={labelStyle}>System ID or register ID</label><input className={inputStyle} value={regId} onChange={e => setRegId(e.target.value)} placeholder="e.g., AI-2026-003" /></div>
        <div><label className={labelStyle}>AI system owner *</label><input className={inputStyle} value={owner} onChange={e => setOwner(e.target.value)} placeholder="Name and title" /></div>
        <div className="grid grid-cols-2 gap-4">
          <div><label className={labelStyle}>Test date</label><input type="date" className={inputStyle} value={testDate} onChange={e => setTestDate(e.target.value)} /></div>
          <div><label className={labelStyle}>Test type</label><select className={selectStyle} value={testType} onChange={e => setTestType(e.target.value)}><option>Pre-Deployment</option><option>Recurring</option><option>Incident-Triggered</option></select></div>
        </div>
      </div>);
      case 1: return (<div className="space-y-4">
        <p className="text-sm text-gray-600">Enter the two groups you are comparing and the outcome counts. Groups should be defined by protected class (e.g., race, ethnicity, sex, age, disability status) per applicable civil rights requirements.</p>
        <div className="grid grid-cols-2 gap-4">
          <div><label className={labelStyle}>Group A name</label><input className={inputStyle} value={groupAName} onChange={e => setGroupAName(e.target.value)} placeholder="e.g., Hispanic" /></div>
          <div><label className={labelStyle}>Group B name</label><input className={inputStyle} value={groupBName} onChange={e => setGroupBName(e.target.value)} placeholder="e.g., Non-Hispanic" /></div>
        </div>
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <table className="w-full text-sm"><thead><tr className="bg-gray-100"><th className="px-4 py-3 text-left font-medium text-gray-700"></th><th className="px-4 py-3 text-center font-medium text-gray-700">Positive outcome<br/><span className="font-normal text-gray-500 text-xs">(approved, selected, etc.)</span></th><th className="px-4 py-3 text-center font-medium text-gray-700">Negative outcome<br/><span className="font-normal text-gray-500 text-xs">(denied, not selected, etc.)</span></th><th className="px-4 py-3 text-center font-medium text-gray-700">Total</th></tr></thead>
          <tbody>
            <tr className="border-t border-gray-100"><td className="px-4 py-3 font-medium text-gray-700">{groupAName}</td><td className="px-4 py-3"><input type="number" min="0" className={inputStyle+" text-center"} value={groupAPos} onChange={e=>setGroupAPos(e.target.value)} placeholder="0"/></td><td className="px-4 py-3"><input type="number" min="0" className={inputStyle+" text-center"} value={groupANeg} onChange={e=>setGroupANeg(e.target.value)} placeholder="0"/></td><td className="px-4 py-3 text-center font-medium text-gray-700">{n1}</td></tr>
            <tr className="border-t border-gray-100"><td className="px-4 py-3 font-medium text-gray-700">{groupBName}</td><td className="px-4 py-3"><input type="number" min="0" className={inputStyle+" text-center"} value={groupBPos} onChange={e=>setGroupBPos(e.target.value)} placeholder="0"/></td><td className="px-4 py-3"><input type="number" min="0" className={inputStyle+" text-center"} value={groupBNeg} onChange={e=>setGroupBNeg(e.target.value)} placeholder="0"/></td><td className="px-4 py-3 text-center font-medium text-gray-700">{n2}</td></tr>
          </tbody></table>
        </div>
        {(n1>0&&n2>0)&&<p className="text-xs text-gray-500">Total observations: {n1+n2}. {(n1<30||n2<30)?"Note: One or both groups have fewer than 30 observations. Fisher's Exact Test will be the primary test.":""}</p>}
      </div>);
      case 2: return (<div className="space-y-5">
        <p className="text-sm text-gray-600">Calculated from your contingency table. The Disparate Impact Ratio uses the four-fifths rule from the Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4).</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl border border-gray-200 p-4 space-y-2"><div className="text-xs font-medium text-gray-500 uppercase tracking-wider">Selection rates</div><div className="flex justify-between text-sm"><span className="text-gray-700">{groupAName}:</span><span className="font-mono font-medium">{(rateA*100).toFixed(1)}%</span></div><div className="flex justify-between text-sm"><span className="text-gray-700">{groupBName}:</span><span className="font-mono font-medium">{(rateB*100).toFixed(1)}%</span></div></div>
          <div className="rounded-xl border-2 p-4 space-y-2" style={{borderColor:SEV_COLORS[quantClass].border,backgroundColor:SEV_COLORS[quantClass].bg}}><div className="text-xs font-medium uppercase tracking-wider" style={{color:SEV_COLORS[quantClass].text}}>Disparate impact ratio</div><div className="text-3xl font-mono font-bold" style={{color:SEV_COLORS[quantClass].text}}>{diRatio.toFixed(3)}</div><div className="text-sm font-medium" style={{color:SEV_COLORS[quantClass].text}}>Quantitative: {quantClass}</div></div>
        </div>
        <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700 space-y-2"><div className="font-medium">Interpretation</div><p>{lowerGroup} has a selection rate of {(lowerRate*100).toFixed(1)}%, compared to {higherGroup} at {(higherRate*100).toFixed(1)}%. The ratio is {diRatio.toFixed(3)}, which is {diRatio>=0.80?"at or above":"below"} the 0.80 threshold.</p>{diRatio<0.80&&<p className="font-medium" style={{color:SEV_COLORS[quantClass].text}}>This falls in the {quantClass} zone. Statistical significance testing on the next step will confirm whether this disparity is likely real or could be due to chance.</p>}</div>
        <div className="rounded-xl border border-gray-200 p-4 text-sm"><div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Threshold reference</div><table className="w-full text-sm"><thead><tr className="text-gray-500 text-xs"><th className="text-left py-1">Metric</th><th className="text-center py-1">Pass (&ge;0.90)</th><th className="text-center py-1">Low (0.80-0.89)</th><th className="text-center py-1">Moderate (0.65-0.79)</th><th className="text-center py-1">Critical (&lt;0.65)</th></tr></thead><tbody><tr><td className="py-1 text-gray-700">DI Ratio</td>{["None","Low","Moderate","Critical"].map(cl=><td key={cl} className="text-center py-1"><span className={`inline-block w-5 h-5 rounded-full ${quantClass===cl?"ring-2 ring-offset-1":"opacity-30"}`} style={{backgroundColor:SEV_COLORS[cl].border}}/></td>)}</tr></tbody></table></div>
      </div>);
      case 3: return (<div className="space-y-5">
        <p className="text-sm text-gray-600">All three standard statistical significance tests are run automatically. Each test determines whether the observed disparity could have occurred by chance.</p>
        {[{name:"Fisher's Exact Test",note:"Valid for all sample sizes. Recommended default.",p:fisherP,valid:true,stat:null,statLabel:null},{name:"Chi-Square Test",note:chi2Result.valid?"All expected cell counts are 5+. Valid.":"One or more expected cell counts below 5. Not valid for this data. Use Fisher's instead.",p:chi2Result.p,valid:chi2Result.valid,stat:chi2Result.stat,statLabel:"\u03C7\u00B2"},{name:"Two-Proportion Z-Test",note:zResult.valid?"Both groups have 30+ observations. Valid.":"One or both groups have fewer than 30 observations. Not valid for this data.",p:zResult.p,valid:zResult.valid,stat:zResult.z,statLabel:"Z"}].map(t=>(<div key={t.name} className={`rounded-xl border p-4 space-y-2 ${t.valid?"border-gray-200":"border-gray-100 opacity-60"}`}><div className="flex items-center justify-between"><div className="font-medium text-gray-800">{t.name}</div>{t.valid?(<span className={`px-3 py-1 rounded-full text-xs font-medium ${t.p<=0.05?"bg-red-100 text-red-800":t.p<=0.10?"bg-amber-100 text-amber-800":"bg-green-100 text-green-800"}`}>p = {t.p<0.001?"< 0.001":t.p.toFixed(4)} {t.p<=0.01?"(highly significant)":t.p<=0.05?"(significant)":t.p<=0.10?"(marginal)":"(not significant)"}</span>):<span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-500">Not applicable</span>}</div><div className="text-xs text-gray-500">{t.note}</div>{t.valid&&t.stat!==null&&<div className="text-xs text-gray-600 font-mono">{t.statLabel} = {t.stat.toFixed(4)}</div>}</div>))}
        <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700"><div className="font-medium mb-1">Best available p-value: {bestP<0.001?"< 0.001":bestP.toFixed(4)}</div><p>{bestP<=0.01?"Highly significant. The disparity is very unlikely due to chance. Consider escalating by one severity level.":bestP<=0.05?"Significant. The disparity is unlikely due to chance. The quantitative finding stands as classified.":bestP<=0.10?"Marginal. The result is borderline. The finding may be reduced by one level. Schedule a follow-up test with a larger sample.":"Not significant. The observed disparity could reasonably be due to chance. The finding is reduced by one severity level."}</p></div>
      </div>);
      case 4: return (<div className="space-y-5">
        <p className="text-sm text-gray-600">Assess the real-world context of the AI system. The highest-scoring factor determines the severity adjustment applied to the quantitative finding.</p>
        {Object.entries(CONTEXT_LABELS).map(([key,cfg])=>(<div key={key} className="space-y-2"><label className={labelStyle}>{cfg.label}</label><div className="grid grid-cols-3 gap-2">{cfg.options.map((opt,i)=>(<button key={i} onClick={()=>setContext(prev=>({...prev,[key]:cfg.scores[i]}))} className={`px-3 py-3 rounded-lg text-xs text-left border transition-all ${context[key]===cfg.scores[i]?"border-blue-500 bg-blue-50 text-blue-800 ring-1 ring-blue-300":"border-gray-200 bg-white text-gray-700 hover:border-gray-400"}`}>{opt}</button>))}</div></div>))}
        <div className="rounded-xl bg-gray-50 border border-gray-200 p-4 text-sm text-gray-700"><span className="font-medium">Highest context factor: </span>{contextMax===0?"Standard (no adjustment)":contextMax===1?"Elevated (+1 severity level)":"High (+2 severity levels)"}</div>
      </div>);
      case 5: return (<div className="space-y-5">
        <p className="text-sm text-gray-600">The final classification combines the quantitative disparity, statistical significance, and context assessment.</p>
        <div className="rounded-2xl border-2 p-6 text-center space-y-3" style={{borderColor:fc.border,backgroundColor:fc.bg}}><div className="text-sm font-medium uppercase tracking-wider" style={{color:fc.text}}>Final classification</div><div className="text-5xl font-bold" style={{color:fc.text}}>{finalClass}</div></div>
        <div className="rounded-xl border border-gray-200 p-4 text-sm space-y-3"><div className="font-medium text-gray-800">How this was determined:</div><div className="space-y-1 text-gray-700"><div className="flex justify-between"><span>Quantitative (DI Ratio: {diRatio.toFixed(3)})</span><span className="font-medium">{quantClass}</span></div><div className="flex justify-between"><span>Context adjustment</span><span className="font-medium">{contextMax===0?"None":contextMax===1?"+1 level":"+2 levels"}</span></div><div className="flex justify-between"><span>Statistical significance (p = {bestP<0.001?"< 0.001":bestP.toFixed(4)})</span><span className="font-medium">{bestP>0.10?"-1 level":bestP<=0.01?"+1 level":"No change"}</span></div><div className="border-t border-gray-200 pt-2 flex justify-between font-medium"><span>Final</span><span style={{color:fc.text}}>{finalClass}</span></div></div></div>
        <div className="rounded-xl border border-gray-200 p-4 text-sm text-gray-700"><div className="font-medium mb-1">Recommended response:</div>{finalClass==="Critical"&&<p>Immediate suspension of AI system pending remediation. Governance body review required before reactivation. Leadership notification.</p>}{finalClass==="Moderate"&&<p>Remediation plan due within 30 days. Add human review checkpoint for affected decisions. Notify governance body. System may continue with enhanced monitoring.</p>}{finalClass==="Low"&&<p>Document finding and root cause analysis. Monitor at next scheduled test. No system interruption required. If Low persists across two consecutive test cycles, escalate to Moderate.</p>}{finalClass==="None"&&<p>Document results. Continue standard monitoring schedule.</p>}</div>
      </div>);
      case 6: return (<div className="space-y-4">
        <p className="text-sm text-gray-600">Review the summary below. Print or save this report with your AI system's governance records.</p>
        <div className="rounded-xl border border-gray-200 p-5 space-y-4 text-sm bg-white" id="report-content">
          <div className="text-center space-y-1 border-b border-gray-200 pb-4"><div className="text-lg font-bold text-gray-900">AI Bias Testing Report</div>{agencyName&&<div className="text-sm text-gray-600">{agencyName}</div>}<div className="text-xs text-gray-400">Generated {new Date().toLocaleDateString()}</div></div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2">{agencyName&&<div><span className="text-gray-500">Organization:</span> <span className="font-medium">{agencyName}</span></div>}<div><span className="text-gray-500">AI System:</span> <span className="font-medium">{sysName}</span></div><div><span className="text-gray-500">System ID:</span> <span className="font-medium">{regId||"N/A"}</span></div><div><span className="text-gray-500">Owner:</span> <span className="font-medium">{owner}</span></div><div><span className="text-gray-500">Test Date:</span> <span className="font-medium">{testDate}</span></div><div><span className="text-gray-500">Test Type:</span> <span className="font-medium">{testType}</span></div></div>
          <div className="border-t border-gray-200 pt-3"><div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Contingency table</div><table className="w-full text-sm border border-gray-200 rounded"><thead><tr className="bg-gray-50"><th className="px-3 py-2 text-left"></th><th className="px-3 py-2 text-center">Positive</th><th className="px-3 py-2 text-center">Negative</th><th className="px-3 py-2 text-center">Total</th><th className="px-3 py-2 text-center">Rate</th></tr></thead><tbody><tr className="border-t"><td className="px-3 py-2 font-medium">{groupAName}</td><td className="px-3 py-2 text-center font-mono">{a}</td><td className="px-3 py-2 text-center font-mono">{b}</td><td className="px-3 py-2 text-center font-mono">{n1}</td><td className="px-3 py-2 text-center font-mono">{(rateA*100).toFixed(1)}%</td></tr><tr className="border-t"><td className="px-3 py-2 font-medium">{groupBName}</td><td className="px-3 py-2 text-center font-mono">{c}</td><td className="px-3 py-2 text-center font-mono">{d}</td><td className="px-3 py-2 text-center font-mono">{n2}</td><td className="px-3 py-2 text-center font-mono">{(rateB*100).toFixed(1)}%</td></tr></tbody></table></div>
          <div className="border-t border-gray-200 pt-3 grid grid-cols-3 gap-4"><div><span className="text-gray-500 text-xs block">DI Ratio</span><span className="font-mono font-bold text-lg">{diRatio.toFixed(3)}</span></div><div><span className="text-gray-500 text-xs block">Quantitative</span><span className="font-medium">{quantClass}</span></div><div><span className="text-gray-500 text-xs block">Context Adj.</span><span className="font-medium">{contextMax===0?"None":contextMax===1?"+1":"+2"}</span></div></div>
          <div className="border-t border-gray-200 pt-3"><div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Statistical tests</div><div className="grid grid-cols-3 gap-2 text-xs"><div className="rounded bg-gray-50 p-2"><div className="font-medium">Fisher's Exact</div><div className="font-mono">p = {fisherP<0.001?"< 0.001":fisherP.toFixed(4)}</div></div><div className="rounded bg-gray-50 p-2"><div className="font-medium">Chi-Square</div><div className="font-mono">{chi2Result.valid?`\u03C7\u00B2 = ${chi2Result.stat.toFixed(3)}, p = ${chi2Result.p<0.001?"< 0.001":chi2Result.p.toFixed(4)}`:"N/A"}</div></div><div className="rounded bg-gray-50 p-2"><div className="font-medium">Z-Test</div><div className="font-mono">{zResult.valid?`Z = ${zResult.z.toFixed(3)}, p = ${zResult.p<0.001?"< 0.001":zResult.p.toFixed(4)}`:"N/A"}</div></div></div></div>
          <div className="border-t border-gray-200 pt-3 flex items-center gap-3"><span className="text-gray-500 text-xs">Final Classification:</span><span className="px-4 py-2 rounded-lg font-bold text-lg" style={{backgroundColor:fc.bg,color:fc.text,border:`2px solid ${fc.border}`}}>{finalClass}</span></div>
          <div className="border-t border-gray-200 pt-3 text-xs text-gray-500 space-y-1"><div>Methodology: Disparate Impact Ratio per Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4). Statistical significance via Fisher's Exact Test, Chi-Square Test of Independence, and/or Two-Proportion Z-Test. Context assessment per NIST AI Risk Management Framework 1.0 (AI 100-1) Measure function.</div><div>References: NIST AI RMF 1.0, NIST SP 1270, EEOC Technical Assistance on AI and Title VII (May 2023).</div></div>
        </div>
        <div className="flex gap-3">
          <button onClick={()=>{const w=window.open('','_blank');w.document.write('<html><head><title>Bias Test Report - '+sysName+'</title><style>body{font-family:system-ui,sans-serif;max-width:700px;margin:2rem auto;color:#1a1a1a;font-size:13px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}.font-mono{font-family:monospace}.font-medium{font-weight:500}.font-bold{font-weight:700}.text-center{text-align:center}.text-xs{font-size:11px}.text-sm{font-size:13px}.text-lg{font-size:18px}.text-gray-500{color:#6b7280}.text-gray-600{color:#4b5563}.text-gray-700{color:#374151}.text-gray-800{color:#1f2937}.text-gray-900{color:#111827}.text-gray-400{color:#9ca3af}.bg-gray-50{background:#f9fafb}.uppercase{text-transform:uppercase}.tracking-wider{letter-spacing:0.05em}div{margin-bottom:4px}</style></head><body>'+document.getElementById('report-content').innerHTML+'</body></html>');w.document.close();w.print();}} className="px-4 py-2 rounded-lg bg-gray-800 text-white text-sm font-medium hover:bg-gray-700 transition-colors">Print report</button>
          <button onClick={()=>setStep(0)} className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors">Start new test</button>
        </div>
      </div>);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4" style={{fontFamily:"'DM Sans',system-ui,sans-serif"}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
      <style>{`
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
        input[type="number"] { -moz-appearance: textfield; appearance: textfield; }
      `}</style>
      <div className="max-w-2xl mx-auto">
        <div className="mb-8 text-center"><h1 className="text-2xl font-bold text-gray-900">AI Bias Testing Tool</h1><p className="text-sm text-gray-500 mt-1">Disparate impact analysis for AI systems</p><p className="text-xs text-gray-400 mt-0.5">Based on NIST AI RMF 1.0, EEOC Uniform Guidelines (29 CFR 1607.4), and NIST SP 1270</p></div>
        <div className="flex items-center justify-center gap-1 mb-8">{STEPS.map((s,i)=>(<div key={i} className="flex items-center"><button onClick={()=>{if(i<step)setStep(i);}} className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium transition-all ${i===step?"bg-blue-600 text-white":i<step?"bg-blue-100 text-blue-700 hover:bg-blue-200 cursor-pointer":"bg-gray-100 text-gray-400"}`}><span className="w-5 h-5 rounded-full flex items-center justify-center text-xs" style={{backgroundColor:i===step?"rgba(255,255,255,0.2)":"transparent"}}>{i<step?"\u2713":i+1}</span><span className="hidden sm:inline">{s}</span></button>{i<STEPS.length-1&&<div className={`w-4 h-px mx-0.5 ${i<step?"bg-blue-300":"bg-gray-200"}`}/>}</div>))}</div>
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 mb-6"><h2 className="text-lg font-bold text-gray-900 mb-4">{STEPS[step]}</h2>{renderStep()}</div>
        {step<6&&(<div className="flex justify-between"><button onClick={()=>setStep(Math.max(0,step-1))} disabled={step===0} className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors ${step===0?"text-gray-300 cursor-not-allowed":"text-gray-700 border border-gray-300 hover:bg-gray-50"}`}>Back</button><button onClick={()=>setStep(step+1)} disabled={!canProceed()} className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors ${canProceed()?"bg-blue-600 text-white hover:bg-blue-700":"bg-gray-200 text-gray-400 cursor-not-allowed"}`}>{step===5?"Generate report":"Continue"}</button></div>)}
        <div className="mt-8 text-center text-xs text-gray-400 space-y-1">
          <p>Open source under MIT License. Free for use by anyone.</p>
          <p>Methodology: NIST AI RMF 1.0, NIST SP 1270, EEOC Uniform Guidelines (29 CFR 1607.4), EEOC Technical Assistance on AI and Title VII.</p>
          <p>Questions, feedback, or contributions: <a href="https://github.com/jasonanwe/bias-testing-app" className="text-blue-500 hover:text-blue-600 underline">jasonanwe/bias-testing-app</a></p>
        </div>
      </div>
    </div>
  );
}
