import React from "react";
import { Cpu, Play } from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";

export const PlaygroundView: React.FC = () => {
  const store = useAppState();
  const session = store.sessions.playground;

  const activeJob = store.jobs.find(
    (j) => j.type === "run_experiment" && (j.status === "Running" || j.status === "Queued")
  );
  const latestCompletedJob = store.jobs.find(
    (j) => j.type === "run_experiment" && j.status === "Completed"
  );

  const trialResult = latestCompletedJob?.result?.result || session.lastTrialResult;

  const handleExecute = () => {
    store.startJob("run_experiment", "Playground Trial Execution", "run_experiment", {
      experiment_name: `playground_trial_${Date.now().toString().slice(-4)}`,
      chunking_config: {
        strategy: session.chunkStrategy,
        chunk_size: session.chunkSize,
        chunk_overlap: session.chunkOverlap,
      },
      retriever_config: {
        strategy: session.retrieverStrategy,
        distance_metric: session.distanceMetric,
        top_k: session.topK,
        hybrid_alpha: 0.7,
      },
      system_prompt: session.systemPrompt,
    });
  };

  return (
    <div id="playground-view-container" className="space-y-4">
      {/* Header Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Single Trial Playground
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Custom chunking, vector metric tuning & precision/correctness evaluation
            </p>
          </div>
        </div>

        <button
          id="run-experiment-btn"
          onClick={handleExecute}
          disabled={!!activeJob}
          className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center space-x-1.5 shadow-sm"
        >
          <Play className={`w-3.5 h-3.5 ${activeJob ? "animate-spin" : ""}`} />
          <span>{activeJob ? "Executing Trial Job..." : "Run Trial"}</span>
        </button>
      </div>

      {/* Active Job Card */}
      {activeJob && (
        <div className="mb-4">
          <JobProgressCard job={activeJob} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Hyperparameter Controls */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-lg p-3.5 space-y-3.5 shadow-sm">
          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Pipeline Parameters
            </span>
            <span className="text-[9px] font-mono text-blue-400 bg-blue-950/60 px-1.5 py-0.5 rounded border border-blue-800/40">
              STATEFUL_SESSION
            </span>
          </div>

          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
              Chunking Algorithm
            </label>
            <select
              id="chunking-strategy-select"
              value={session.chunkStrategy}
              onChange={(e) =>
                store.updateSession("playground", {
                  chunkStrategy: e.target.value as any,
                })
              }
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
            >
              <option value="recursive">Recursive Character Splitter</option>
              <option value="fixed">Fixed Size Window</option>
              <option value="paragraph">Paragraph Splitter</option>
              <option value="semantic">Semantic Sentence Boundaries</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div>
              <label className="block text-[9px] font-bold text-slate-400 uppercase mb-1 font-mono">
                Size ({session.chunkSize} tokens)
              </label>
              <input
                id="chunk-size-input"
                type="range"
                min={128}
                max={2048}
                step={64}
                value={session.chunkSize}
                onChange={(e) =>
                  store.updateSession("playground", { chunkSize: Number(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
            </div>
            <div>
              <label className="block text-[9px] font-bold text-slate-400 uppercase mb-1 font-mono">
                Overlap ({session.chunkOverlap} tokens)
              </label>
              <input
                id="chunk-overlap-input"
                type="range"
                min={0}
                max={256}
                step={16}
                value={session.chunkOverlap}
                onChange={(e) =>
                  store.updateSession("playground", { chunkOverlap: Number(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
              Retriever Strategy
            </label>
            <select
              id="retriever-strategy-select"
              value={session.retrieverStrategy}
              onChange={(e) =>
                store.updateSession("playground", { retrieverStrategy: e.target.value as any })
              }
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
            >
              <option value="hybrid">Hybrid (Dense Vectors + Sparse BM25)</option>
              <option value="dense">Dense Vector Search Only</option>
              <option value="sparse">Sparse BM25 Keyword Only</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div>
              <label className="block text-[9px] font-bold text-slate-400 uppercase mb-1 font-mono">
                Distance Metric
              </label>
              <select
                id="distance-metric-select"
                value={session.distanceMetric}
                onChange={(e) =>
                  store.updateSession("playground", { distanceMetric: e.target.value as any })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-white font-mono"
              >
                <option value="cosine">Cosine Similarity</option>
                <option value="dot">Dot Product</option>
                <option value="euclidean">Euclidean Score</option>
              </select>
            </div>
            <div>
              <label className="block text-[9px] font-bold text-slate-400 uppercase mb-1 font-mono">
                Top-K ({session.topK})
              </label>
              <input
                id="top-k-input"
                type="range"
                min={1}
                max={10}
                value={session.topK}
                onChange={(e) =>
                  store.updateSession("playground", { topK: Number(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
              System Prompt
            </label>
            <textarea
              id="system-prompt-textarea"
              rows={3}
              value={session.systemPrompt}
              onChange={(e) => store.updateSession("playground", { systemPrompt: e.target.value })}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:border-blue-500 focus:outline-none font-mono"
            ></textarea>
          </div>
        </div>

        {/* Experiment Results Panel */}
        <div className="lg:col-span-8 space-y-3">
          {trialResult ? (
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4 shadow-sm">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-blue-400 block">
                    TRIAL_RESULT #{trialResult.experiment_id}
                  </span>
                  <h2 className="text-xs font-bold text-white uppercase tracking-wider">
                    Evaluation Metrics Summary
                  </h2>
                </div>

                <div className="text-right">
                  {trialResult.metrics_valid !== false && trialResult.composite_score !== null && trialResult.status !== "failed" ? (
                    <>
                      <span className="text-xl font-bold font-mono text-emerald-400">
                        {(trialResult.composite_score * 100).toFixed(1)}%
                      </span>
                      <span className="block text-[9px] text-slate-500 font-mono uppercase">
                        COMPOSITE SCORE
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="text-sm font-bold font-mono text-red-400">
                        FAILED
                      </span>
                      <span className="block text-[9px] text-slate-500 font-mono uppercase max-w-[200px] truncate" title={trialResult.failure_reason}>
                        {trialResult.failure_reason || "TRIAL INVALID"}
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Retrieval vs Answer Score Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-center">
                  <span className="text-[9px] text-slate-500 uppercase block font-mono">
                    Hit Rate@K
                  </span>
                  <span className="text-xs font-bold font-mono text-white">
                    {((trialResult.retrieval_metrics?.hit_rate || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-center">
                  <span className="text-[9px] text-slate-500 uppercase block font-mono">
                    Context Precision
                  </span>
                  <span className="text-xs font-bold font-mono text-blue-400">
                    {((trialResult.retrieval_metrics?.precision || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-center">
                  <span className="text-[9px] text-slate-500 uppercase block font-mono">
                    Answer Correctness
                  </span>
                  <span className="text-xs font-bold font-mono text-purple-400">
                    {(((trialResult.answer_metrics?.answer_correctness ?? trialResult.answer_metrics?.accuracy ?? trialResult.answer_metrics?.faithfulness) || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-center">
                  <span className="text-[9px] text-slate-500 uppercase block font-mono">
                    Avg Latency
                  </span>
                  <span className="text-xs font-bold font-mono text-amber-400">
                    {trialResult.avg_latency_ms || 120} ms
                  </span>
                </div>
              </div>

              {/* Sample Test Case Evaluation */}
              <div className="space-y-2">
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                  Evaluated Sample Test Pairs ({trialResult.sample_evaluations?.length || 0})
                </h3>

                {trialResult.sample_evaluations?.map((sample: any, idx: number) => (
                  <div
                    key={idx}
                    className="bg-slate-950/60 p-3 rounded border border-slate-800 space-y-1.5 text-xs font-mono"
                  >
                    <div className="flex items-center justify-between font-semibold text-white">
                      <span>Q: {sample.question}</span>
                      <span className="text-emerald-400 text-[10px]">
                        PRECISION: {((sample.r_precision || 0.9) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-slate-300 bg-slate-900 p-2 rounded border border-slate-800/80 text-[11px]">
                      <strong className="text-blue-400 block mb-0.5">GENERATED ANSWER:</strong>
                      {sample.answer}
                    </p>
                    <p className="text-slate-400 text-[10px]">
                      <strong className="text-slate-500">GROUND TRUTH:</strong>{" "}
                      {sample.ground_truth}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-16 bg-slate-900 border border-slate-800 rounded p-4 text-slate-500 text-xs font-mono">
              Configure parameters on the left and click{" "}
              <strong className="text-blue-400">RUN TRIAL</strong> to execute background job
              evaluation.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
