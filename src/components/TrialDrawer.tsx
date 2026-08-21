import React, { useState } from "react";
import { X, ChevronDown, Sparkles, Copy, Download, Code, ArrowRight, ShieldCheck, Play, RefreshCw, CheckCircle2, AlertCircle, Activity } from "lucide-react";
import { LeaderboardEntry } from "../types";
import Markdown from "react-markdown";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";

interface TrialDrawerProps {
  trial: LeaderboardEntry;
  leaderboard: LeaderboardEntry[];
  rank: number;
  onClose: () => void;
}

export const TrialDrawer: React.FC<TrialDrawerProps> = ({ trial, leaderboard, rank, onClose }) => {
  const store = useAppState();
  const [activeTab, setActiveTab] = useState<"overview" | "comparison" | "generalization">("overview");
  const [compareTrialId, setCompareTrialId] = useState<string | null>(null);
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explanationPrompt, setExplanationPrompt] = useState<string | null>(null);
  const [width, setWidth] = useState(540);
  const [testSize, setTestSize] = useState(5);
  const [expandedSampleIdx, setExpandedSampleIdx] = useState<number | null>(null);

  const liveTrial = store.leaderboard.find(t => t.experiment_id === trial.experiment_id) || 
                    leaderboard.find(t => t.experiment_id === trial.experiment_id) || 
                    trial;

  const compareTrial = compareTrialId ? (store.leaderboard.find(t => t.experiment_id === compareTrialId) || leaderboard.find(t => t.experiment_id === compareTrialId)) : null;
  const compareRank = compareTrialId ? (leaderboard.findIndex(t => t.experiment_id === compareTrialId) + 1) : 0;

  const activeExportJob = store.jobs.find(
    (j) => (j.type === "export_rag" || j.payload?.action === "export_rag") && 
           (j.payload?.experiment_id === trial.experiment_id || j.payload?.trialId === trial.experiment_id)
  );

  const activeGenJob = store.jobs.find(
    (j) => (j.type === "run_generalization_test" || j.payload?.action === "run_generalization_test" || j.action === "run_generalization_test") &&
           (j.payload?.experiment_id === trial.experiment_id || 
            j.payload?.trialId === trial.experiment_id || 
            j.result?.experiment_id === trial.experiment_id ||
            (typeof j.title === "string" && trial.experiment_id && j.title.includes(trial.experiment_id.slice(0, 14))))
  );

  const handleExportRAG = () => {
    store.startJob("export_rag", `Export RAG Package: ${trial.experiment_id}`, "export_rag", {
      experiment_id: trial.experiment_id
    });
  };

  const handleRunGeneralization = () => {
    store.runGeneralizationTest(trial.experiment_id, testSize);
  };

  const genTest = liveTrial.generalization_test || 
                  trial.generalization_test || 
                  (activeGenJob?.status === "Completed" ? (activeGenJob.result?.generalization_result || activeGenJob.result?.result) : null) ||
                  (activeGenJob?.status === "Failed" ? { status: "failed", failure_reason: activeGenJob.error } : null);

  React.useEffect(() => {
    if (activeTab === "generalization") {
      store.fetchLeaderboard();
    }
  }, [activeTab, trial.experiment_id]);

  const handleDrag = (e: React.MouseEvent) => {
    const startX = e.clientX;
    const startWidth = width;
    
    const onMouseMove = (e: MouseEvent) => {
      const newWidth = startWidth - (e.clientX - startX);
      if (newWidth >= 440 && newWidth <= 860) {
        setWidth(newWidth);
      }
    };
    
    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  };

  const generatePrompt = (t1: LeaderboardEntry, t2: LeaderboardEntry) => {
    return `Trial A:
${JSON.stringify({
  config: t1.config,
  metrics: t1.result?.retrieval_metrics,
  composite: t1.composite_score,
  latency: t1.result?.avg_latency_ms
}, null, 2)}

Trial B:
${JSON.stringify({
  config: t2.config,
  metrics: t2.result?.retrieval_metrics,
  composite: t2.composite_score,
  latency: t2.result?.avg_latency_ms
}, null, 2)}

Task:
Explain:
1. Which parameters changed?
2. Which changes probably improved retrieval?
3. Which changes improved answer correctness?
4. Which changes likely hurt latency?
5. Why is Trial B better?
6. Would you recommend Trial B?
7. What should be tested next?

Be concise. Format as markdown.`;
  };

  const handleExplain = async () => {
    if (!compareTrial) return;
    
    setIsExplaining(true);
    setAiExplanation(null);
    console.log(`[AI Comparison]\nPreparing comparison prompt...`);
    
    const prompt = generatePrompt(trial, compareTrial);
    setExplanationPrompt(prompt);
    
    try {
      console.log(`Sending request...`);
      const conn = store.getActiveConnectionConfig();
      const response = await fetch("/api/llm/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, connection: conn })
      });
      console.log(`Receiving response...`);
      
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errText}`);
      }
      
      const data = await response.json();
      if (data.text) {
        setAiExplanation(data.text);
        console.log(`Completed.`);
      } else if (data.error) {
        throw new Error(data.error);
      }
    } catch (e) {
      console.error(e);
      setAiExplanation("Error generating explanation.");
    } finally {
      setIsExplaining(false);
    }
  };

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(trial, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trial_${rank.toString().padStart(3, '0')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportMarkdown = () => {
    if (!compareTrial) return;
    const md = `# Comparison: Trial #${rank} vs Trial #${compareRank}
Date: ${new Date().toISOString()}

## Trial #${rank} Config
${JSON.stringify(trial.config, null, 2)}

## Trial #${compareRank} Config
${JSON.stringify(compareTrial.config, null, 2)}

## Differences
... (Auto-generated from diff)

## AI Analysis
${aiExplanation || "Not generated."}
`;
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trial${rank}_vs_trial${compareRank}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderDifferences = () => {
    if (!compareTrial) return null;
    
    const diffs: React.ReactNode[] = [];
    
    const compareObj = (obj1: any, obj2: any, prefix = "") => {
      if (!obj1 || !obj2) return;
      Object.keys(obj1).forEach(key => {
        if (typeof obj1[key] === "object" && obj1[key] !== null) {
          compareObj(obj1[key], obj2[key], `${prefix}${key}.`);
        } else if (obj1[key] !== obj2[key]) {
          diffs.push(
            <div key={`${prefix}${key}`} className="flex items-center justify-between py-2 border-b border-slate-800/60 last:border-0">
              <span className="text-[11px] text-slate-400 capitalize">{prefix}${key}</span>
              <div className="flex items-center space-x-2 font-mono text-[10px]">
                <span className="text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">{String(obj1[key])}</span>
                <ArrowRight className="w-3 h-3 text-slate-500" />
                <span className="text-blue-300 bg-blue-900/30 px-1.5 py-0.5 rounded">{String(obj2[key])}</span>
              </div>
            </div>
          );
        }
      });
    };
    
    compareObj(trial.config, compareTrial.config, "config.");
    compareObj(trial.result?.retrieval_metrics, compareTrial.result?.retrieval_metrics, "metrics.");
    
    if (trial.composite_score !== compareTrial.composite_score) {
      diffs.push(
        <div key="composite" className="flex items-center justify-between py-2 border-b border-slate-800/60 last:border-0">
          <span className="text-[11px] text-slate-400 capitalize">Composite Score</span>
          <div className="flex items-center space-x-2 font-mono text-[10px]">
            <span className="text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">{trial.composite_score !== null ? (trial.composite_score! * 100).toFixed(1) + "%" : "N/A"}</span>
            <ArrowRight className="w-3 h-3 text-slate-500" />
            <span className="text-emerald-300 bg-emerald-900/30 px-1.5 py-0.5 rounded">{compareTrial.composite_score !== null ? (compareTrial.composite_score! * 100).toFixed(1) + "%" : "N/A"}</span>
          </div>
        </div>
      );
    }
    
    if (trial.result?.avg_latency_ms !== compareTrial.result?.avg_latency_ms) {
      diffs.push(
        <div key="latency" className="flex items-center justify-between py-2 border-b border-slate-800/60 last:border-0">
          <span className="text-[11px] text-slate-400 capitalize">Latency</span>
          <div className="flex items-center space-x-2 font-mono text-[10px]">
            <span className="text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">{trial.result?.avg_latency_ms?.toFixed(0)}ms</span>
            <ArrowRight className="w-3 h-3 text-slate-500" />
            <span className="text-amber-300 bg-amber-900/30 px-1.5 py-0.5 rounded">{compareTrial.result?.avg_latency_ms?.toFixed(0)}ms</span>
          </div>
        </div>
      );
    }

    if (diffs.length === 0) {
      return <div className="text-xs text-slate-500 py-4 text-center">No differences found between these trials.</div>;
    }

    return <div className="space-y-1">{diffs}</div>;
  };

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex shadow-2xl" style={{ width: `${width}px` }}>
      {/* Resizer handle */}
      <div 
        className="w-1.5 bg-slate-900 hover:bg-blue-500/50 cursor-col-resize transition-colors flex items-center justify-center border-l border-slate-800"
        onMouseDown={handleDrag}
      />
      
      <div className="flex-1 min-w-0 bg-slate-950 border-l border-slate-800 flex flex-col h-full">
        {/* Header (Sticky) */}
        <div className="sticky top-0 bg-slate-900 border-b border-slate-800 p-4 flex flex-col gap-3 z-10">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-slate-400">Trial</span>
              <span className="bg-slate-800 px-2 py-0.5 rounded text-blue-400">#{rank}</span>
            </h2>
            <div className="flex items-center gap-2">
              {activeExportJob?.status === "Completed" && activeExportJob.result?.download_url ? (
                <a href={activeExportJob.result.download_url} target="_blank" rel="noreferrer" className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[10px] font-bold uppercase tracking-wider transition flex items-center gap-1 shadow-sm">
                  <Download className="w-3.5 h-3.5" /> Download ZIP
                </a>
              ) : activeExportJob && (activeExportJob.status === "Running" || activeExportJob.status === "Queued") ? (
                <div className="px-3 py-1 bg-slate-800 text-blue-400 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 shadow-sm">
                  <div className="w-3.5 h-3.5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                  Building...
                </div>
              ) : (
                <button onClick={handleExportRAG} className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-[10px] font-bold uppercase tracking-wider transition flex items-center gap-1 shadow-sm">
                  <Code className="w-3.5 h-3.5" /> Export RAG
                </button>
              )}
              <button onClick={exportJSON} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition ml-2" title="Export JSON">
                <Download className="w-4 h-4" />
              </button>
              <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          {activeExportJob && (
            <div className="pt-2">
              <JobProgressCard job={activeExportJob} />
            </div>
          )}

          {activeGenJob && (
            <div className="pt-2">
              <JobProgressCard job={activeGenJob} />
            </div>
          )}

          <div className="flex gap-4 border-b border-slate-800 mt-2 overflow-x-auto overflow-y-hidden">
            <button 
              className={`pb-2 text-[11px] uppercase font-bold tracking-wider transition ${activeTab === "overview" ? "text-blue-400 border-b-2 border-blue-400" : "text-slate-500 hover:text-slate-300"}`}
              onClick={() => setActiveTab("overview")}
            >
              Overview
            </button>
            <button 
              className={`pb-2 text-[11px] uppercase font-bold tracking-wider transition flex items-center gap-1.5 ${activeTab === "generalization" ? "text-emerald-400 border-b-2 border-emerald-400" : "text-slate-500 hover:text-slate-300"}`}
              onClick={() => setActiveTab("generalization")}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              Generalization Test
              {genTest?.status === "completed" && (
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              )}
            </button>
            <button 
              className={`pb-2 text-[11px] uppercase font-bold tracking-wider transition ${activeTab === "comparison" ? "text-purple-400 border-b-2 border-purple-400" : "text-slate-500 hover:text-slate-300"}`}
              onClick={() => setActiveTab("comparison")}
            >
              Comparison
            </button>
          </div>
        </div>
        
        {/* Scrollable Content */}
        <div className="flex-1 min-w-0 overflow-y-auto overflow-x-hidden p-4 space-y-6">
          {activeTab === "overview" && (
            <>
              {/* Metrics */}
              <div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center justify-between">
                  <span>Optimization Metrics</span>
                  {trial.composite_score !== null && (
                    <span className="text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-900/50">
                      Composite: {(trial.composite_score * 100).toFixed(1)}%
                    </span>
                  )}
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded flex flex-col">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wide">Hit Rate</span>
                    <span className="text-lg font-mono font-bold text-white">
                      {trial.retrieval_hit_rate !== null ? `${(trial.retrieval_hit_rate! * 100).toFixed(1)}%` : "N/A"}
                    </span>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded flex flex-col">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wide">Answer Correctness</span>
                    <span className="text-lg font-mono font-bold text-white">
                      {(trial.answer_correctness ?? trial.answer_faithfulness) !== null && (trial.answer_correctness ?? trial.answer_faithfulness) !== undefined
                        ? `${((trial.answer_correctness ?? trial.answer_faithfulness)! * 100).toFixed(1)}%`
                        : "N/A"}
                    </span>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded flex flex-col">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wide">Context Precision</span>
                    <span className="text-lg font-mono font-bold text-white">
                      {trial.result?.retrieval_metrics?.precision !== undefined ? `${(trial.result.retrieval_metrics.precision * 100).toFixed(1)}%` : "N/A"}
                    </span>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded flex flex-col">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wide">Latency</span>
                    <span className="text-lg font-mono font-bold text-white">
                      {trial.result?.avg_latency_ms !== undefined ? `${trial.result.avg_latency_ms.toFixed(0)} ms` : "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Generalization Status Callout */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-bold text-white">Generalization Test</span>
                  </div>
                  {genTest?.status === "completed" ? (
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300">
                      Validated ({((genTest.generalization_composite_score || 0) * 100).toFixed(1)}%)
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                      Not Validated
                    </span>
                  )}
                </div>
                {genTest?.status === "completed" ? (
                  <div className="text-xs text-slate-300 space-y-1">
                    <p className="text-[11px] text-slate-400">
                      Evaluated against {genTest.test_size} holdout questions. Score delta:{" "}
                      <span className={`font-mono font-bold ${(genTest.score_delta || 0) >= 0 ? "text-emerald-400" : "text-amber-400"}`}>
                        {genTest.score_delta !== null ? `${genTest.score_delta >= 0 ? "+" : ""}${(genTest.score_delta * 100).toFixed(1)}%` : "N/A"}
                      </span>
                    </p>
                    <button
                      onClick={() => setActiveTab("generalization")}
                      className="text-[11px] text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1 mt-1.5"
                    >
                      View Holdout Results <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                ) : (
                  <div className="text-xs text-slate-400">
                    <p className="text-[11px]">Test if this configuration holds up on unseen, independent questions without overfitting.</p>
                    <button
                      onClick={() => setActiveTab("generalization")}
                      className="mt-2.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold flex items-center gap-1.5 transition shadow-sm"
                    >
                      <Play className="w-3 h-3 fill-current" /> Run Generalization Test
                    </button>
                  </div>
                )}
              </div>
              
              {/* Configuration */}
              <div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Configuration</h3>
                <div className="bg-slate-900 border border-slate-800 rounded divide-y divide-slate-800/60">
                  <div className="flex justify-between p-3">
                    <span className="text-[11px] text-slate-400">Retriever Strategy</span>
                    <span className="text-[11px] font-mono font-bold text-slate-300 uppercase">{trial.config?.retriever_config?.strategy || "N/A"}</span>
                  </div>
                  <div className="flex justify-between p-3">
                    <span className="text-[11px] text-slate-400">Chunk Size</span>
                    <span className="text-[11px] font-mono font-bold text-slate-300">{trial.config?.chunking_config?.chunk_size || "N/A"}</span>
                  </div>
                  <div className="flex justify-between p-3">
                    <span className="text-[11px] text-slate-400">Chunk Overlap</span>
                    <span className="text-[11px] font-mono font-bold text-slate-300">{trial.config?.chunking_config?.chunk_overlap || "N/A"}</span>
                  </div>
                  <div className="flex justify-between p-3">
                    <span className="text-[11px] text-slate-400">Top K</span>
                    <span className="text-[11px] font-mono font-bold text-slate-300">{trial.config?.retriever_config?.top_k || "N/A"}</span>
                  </div>
                  {trial.config?.retriever_config?.strategy === "hybrid" && (
                    <div className="flex justify-between p-3">
                      <span className="text-[11px] text-slate-400">Hybrid Alpha</span>
                      <span className="text-[11px] font-mono font-bold text-slate-300">{trial.config?.retriever_config?.hybrid_alpha ?? "0.5"}</span>
                    </div>
                  )}
                  <div className="flex justify-between p-3">
                    <span className="text-[11px] text-slate-400">Distance Metric</span>
                    <span className="text-[11px] font-mono font-bold text-slate-300 uppercase">{trial.config?.retriever_config?.distance_metric || "cosine"}</span>
                  </div>
                </div>
              </div>

              {/* Execution */}
              <div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Execution & Lineage</h3>
                <div className="grid grid-cols-2 gap-3 text-[11px]">
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded">
                    <span className="text-slate-500 block mb-1">Tokens Used</span>
                    <span className="font-mono text-slate-300">{trial.result?.total_tokens || 0}</span>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded">
                    <span className="text-slate-500 block mb-1">Experiment ID</span>
                    <span className="font-mono text-slate-300 truncate block w-full" title={trial.experiment_id}>{trial.experiment_id || "N/A"}</span>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === "generalization" && (
            <div className="space-y-6">
              {/* Feature Explanation Header */}
              <div className="bg-slate-900 border border-emerald-950 p-4 rounded-lg">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-md bg-emerald-950/60 border border-emerald-800/60 text-emerald-400 mt-0.5">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">Generalization (Holdout) Validation</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Evaluates Trial #{rank} using a newly synthesized holdout dataset strictly deduplicated against the optimization evaluation set.
                    </p>
                  </div>
                </div>

                {/* Trigger controls */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <label className="text-[11px] text-slate-400 uppercase font-semibold">Holdout Size:</label>
                    <input
                      type="number"
                      min={1}
                      max={1000}
                      value={testSize}
                      onChange={(e) => setTestSize(Number(e.target.value))}
                      className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-xs text-white font-mono outline-none w-20"
                    />
                    <span className="text-[11px] text-slate-500">Questions</span>
                  </div>

                  <button
                    onClick={handleRunGeneralization}
                    disabled={activeGenJob?.status === "Running" || activeGenJob?.status === "Queued"}
                    className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 transition shadow-sm"
                  >
                    {genTest?.status === "completed" ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5" /> Re-run Test
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 fill-current" /> Run Test
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Generalization Results */}
              {genTest && (genTest.status === "completed" || genTest.status === "success" || genTest.generalization_composite_score !== undefined) ? (
                <div className="space-y-4">
                  {/* Summary Metric Cards */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5">Holdout Comparison</h4>
                    <div className="grid grid-cols-2 gap-3">
                      {/* Optimization Score */}
                      <div className="bg-slate-900 border border-slate-800 p-3 rounded flex flex-col">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wide">Optimization Score</span>
                        <span className="text-xl font-mono font-bold text-blue-400">
                          {liveTrial.composite_score !== null ? `${(liveTrial.composite_score * 100).toFixed(1)}%` : "N/A"}
                        </span>
                        <span className="text-[10px] text-slate-500 mt-0.5">Training / Sweep Eval</span>
                      </div>

                      {/* Generalization Score */}
                      <div className="bg-slate-900 border border-emerald-900/40 p-3 rounded flex flex-col bg-gradient-to-br from-slate-900 to-emerald-950/20">
                        <span className="text-[10px] text-emerald-400 uppercase tracking-wide font-semibold">Generalization Score</span>
                        <span className="text-xl font-mono font-bold text-emerald-300">
                          {genTest.generalization_composite_score !== null ? `${(genTest.generalization_composite_score * 100).toFixed(1)}%` : "N/A"}
                        </span>
                        <span className="text-[10px] text-slate-400 mt-0.5">
                          Delta:{" "}
                          <strong className={(genTest.score_delta || 0) >= 0 ? "text-emerald-400" : "text-amber-400"}>
                            {genTest.score_delta !== null ? `${genTest.score_delta >= 0 ? "+" : ""}${(genTest.score_delta * 100).toFixed(1)}%` : "N/A"}
                          </strong>
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Secondary Metrics */}
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="bg-slate-900 border border-slate-800 p-2.5 rounded">
                      <span className="text-[10px] text-slate-400 block uppercase">Holdout Hit Rate</span>
                      <span className="text-sm font-mono font-bold text-white mt-1 block">
                        {genTest.retrieval_metrics?.hit_rate !== undefined ? `${(genTest.retrieval_metrics.hit_rate * 100).toFixed(1)}%` : "N/A"}
                      </span>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-2.5 rounded">
                      <span className="text-[10px] text-slate-400 block uppercase">Holdout Answer Correctness</span>
                      <span className="text-sm font-mono font-bold text-white mt-1 block">
                        {(genTest.answer_metrics?.answer_correctness ?? genTest.answer_metrics?.accuracy ?? genTest.answer_metrics?.faithfulness) !== undefined
                          ? `${(((genTest.answer_metrics?.answer_correctness ?? genTest.answer_metrics?.accuracy ?? genTest.answer_metrics?.faithfulness) || 0) * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-2.5 rounded">
                      <span className="text-[10px] text-slate-400 block uppercase">Holdout Latency</span>
                      <span className="text-sm font-mono font-bold text-white mt-1 block">
                        {genTest.avg_latency_ms !== undefined ? `${genTest.avg_latency_ms.toFixed(0)} ms` : "N/A"}
                      </span>
                    </div>
                  </div>

                  {/* Summary Text */}
                  {genTest.summary_text && (
                    <div className="bg-slate-900 border border-slate-800 p-3 rounded text-xs text-slate-300 leading-relaxed font-mono">
                      <span className="text-slate-500 uppercase text-[10px] block font-bold mb-1">Executive Summary</span>
                      {genTest.summary_text}
                    </div>
                  )}

                  {/* Sample Evaluations Inspector */}
                  {genTest.sample_evaluations && genTest.sample_evaluations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                        <span>Holdout Questions ({genTest.sample_evaluations.length})</span>
                      </h4>
                      <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
                        {genTest.sample_evaluations.map((sample, sIdx) => {
                          const isExpanded = expandedSampleIdx === sIdx;
                          return (
                            <div key={sIdx} className="bg-slate-900 border border-slate-800 rounded p-3 text-xs">
                              <div
                                className="flex items-start justify-between cursor-pointer gap-2"
                                onClick={() => setExpandedSampleIdx(isExpanded ? null : sIdx)}
                              >
                                <div className="flex-1 font-medium text-slate-200">
                                  <span className="text-slate-500 font-mono mr-1.5">Q{sIdx + 1}:</span>
                                  {sample.question}
                                </div>
                                <div className="flex items-center gap-2 shrink-0 font-mono text-[10px]">
                                  <span className={`px-1.5 py-0.5 rounded ${sample.hit_rate > 0.5 ? "bg-emerald-950 text-emerald-400" : "bg-red-950 text-red-400"}`}>
                                    Hit: {sample.hit_rate > 0.5 ? "Yes" : "No"}
                                  </span>
                                  <span className="px-1.5 py-0.5 rounded bg-blue-950 text-blue-400">
                                    Correctness: {(((sample.answer_correctness ?? sample.faithfulness ?? sample.accuracy) || 0) * 100).toFixed(0)}%
                                  </span>
                                  <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                                </div>
                              </div>

                              {isExpanded && (
                                <div className="mt-3 pt-2.5 border-t border-slate-800 space-y-2 text-[11px]">
                                  <div>
                                    <span className="text-slate-500 block font-semibold">Generated Answer:</span>
                                    <p className="text-slate-300 mt-0.5 bg-slate-950 p-2 rounded border border-slate-800">
                                      {sample.generated_answer || "No answer generated"}
                                    </p>
                                  </div>
                                  <div>
                                    <span className="text-slate-500 block font-semibold">Ground Truth:</span>
                                    <p className="text-slate-400 mt-0.5 italic">
                                      {sample.ground_truth}
                                    </p>
                                  </div>
                                  <div className="flex items-center gap-4 text-slate-400 pt-1">
                                    <span>Retrieved Chunks: <strong className="text-slate-200 font-mono">{sample.retrieved_chunks_count}</strong></span>
                                    <span>Precision: <strong className="text-slate-200 font-mono">{(sample.precision * 100).toFixed(0)}%</strong></span>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : genTest && genTest.status === "failed" ? (
                <div className="p-4 bg-red-950/40 border border-red-900 rounded-lg text-xs text-red-300 space-y-2">
                  <div className="flex items-center gap-2 font-bold">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    Generalization Test Failed
                  </div>
                  <p>{genTest.failure_reason || "Validation could not complete."}</p>
                </div>
              ) : activeGenJob && (activeGenJob.status === "Running" || activeGenJob.status === "Queued" || activeGenJob.status === "Starting") ? (
                <div className="p-8 text-center bg-slate-900/40 border border-dashed border-slate-800 rounded-lg">
                  <Activity className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-2" />
                  <h4 className="text-sm font-bold text-slate-300">Generalization Test in Progress</h4>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
                    The pipeline is currently evaluating on holdout data. See the job tracker above for details.
                  </p>
                </div>
              ) : (
                <div className="p-8 text-center bg-slate-900/40 border border-dashed border-slate-800 rounded-lg">
                  <ShieldCheck className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <h4 className="text-sm font-bold text-slate-300">No Generalization Test Run Yet</h4>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
                    Click "Run Test" above to generate a holdout validation set and verify the pipeline's generalization capability.
                  </p>

                  {genTest && (
                    <div className="mt-4 p-2 bg-slate-900 border border-slate-700 rounded text-left text-xs text-slate-400 overflow-x-auto">
                      <strong>Debug Info:</strong> Test found but status check failed.
                      <pre className="mt-1 text-[10px]">{JSON.stringify({ status: genTest.status, score: genTest.generalization_composite_score }, null, 2)}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === "comparison" && (
            <div className="space-y-6">
              {/* Target Selector */}
              <div>
                <label className="text-[10px] uppercase font-bold text-slate-500 block mb-2">Compare With...</label>
                <div className="relative">
                  <select 
                    className="w-full bg-slate-900 border border-slate-800 rounded p-2.5 text-xs text-white appearance-none outline-none focus:border-purple-500 font-mono"
                    value={compareTrialId || ""}
                    onChange={(e) => setCompareTrialId(e.target.value)}
                  >
                    <option value="" disabled>Select Trial to Compare</option>
                    {leaderboard.filter(t => t.experiment_id !== trial.experiment_id).map((t, idx) => (
                      <option key={t.experiment_id} value={t.experiment_id}>
                        Trial #{leaderboard.findIndex(x => x.experiment_id === t.experiment_id) + 1} (Score: {t.composite_score ? (t.composite_score * 100).toFixed(1) : 'N/A'}%)
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-2.5 w-4 h-4 text-slate-500 pointer-events-none" />
                </div>
              </div>

              {compareTrial && (
                <>
                  {/* Differences */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Parameter Diff</h3>
                      <button 
                        onClick={exportMarkdown}
                        className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded transition flex items-center gap-1"
                      >
                        <Download className="w-3 h-3" /> Export
                      </button>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 rounded p-3">
                      {renderDifferences()}
                    </div>
                  </div>

                  {/* AI Explanation Action */}
                  <div className="pt-4 border-t border-slate-800">
                    {!aiExplanation && !isExplaining && (
                      <button 
                        onClick={handleExplain}
                        className="w-full py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs uppercase tracking-wider rounded transition-colors flex items-center justify-center gap-2 shadow-lg shadow-purple-900/20"
                      >
                        <Sparkles className="w-4 h-4" /> Explain Differences (AI)
                      </button>
                    )}
                    
                    {isExplaining && (
                      <div className="w-full py-4 bg-slate-900 border border-slate-800 rounded flex flex-col items-center justify-center gap-3">
                        <Sparkles className="w-5 h-5 text-purple-400 animate-pulse" />
                        <span className="text-xs text-slate-400 font-mono">Analyzing parameters...</span>
                      </div>
                    )}
                    
                    {aiExplanation && (
                      <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
                        <div className="flex items-center justify-between">
                          <h3 className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-2">
                            <Sparkles className="w-3.5 h-3.5" /> AI Analysis
                          </h3>
                          {explanationPrompt && (
                            <button 
                              onClick={() => navigator.clipboard.writeText(explanationPrompt)}
                              className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1 transition"
                              title="Copy Analysis Prompt"
                            >
                              <Copy className="w-3 h-3" /> Prompt
                            </button>
                          )}
                        </div>
                        <div className="bg-slate-900 border border-purple-900/30 p-4 rounded text-[13px] text-slate-300 leading-relaxed shadow-inner markdown-body">
                           <Markdown>{aiExplanation}</Markdown>
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
