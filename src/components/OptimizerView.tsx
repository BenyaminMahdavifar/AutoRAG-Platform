import React, { useState } from "react";
import { Sliders, Sparkles, CheckCircle, Trophy, Trash, ShieldCheck } from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";
import { LeaderboardEntry } from "../types";
import { TrialDrawer } from "./TrialDrawer";

export const OptimizerView: React.FC = () => {
  const store = useAppState();
  const session = store.sessions.optimizer;
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);

  const activeJob = store.jobs.find(
    (j) => j.type === "run_optimizer" && (j.status === "Running" || j.status === "Queued")
  );
  const latestCompletedJob = store.jobs.find(
    (j) => j.type === "run_optimizer" && j.status === "Completed"
  );

  const sweepSummary = latestCompletedJob?.result?.summary || session.lastSweepSummary;
  const rawLeaderboard: LeaderboardEntry[] = (store.leaderboard && store.leaderboard.length > 0)
    ? store.leaderboard
    : (sweepSummary?.leaderboard || []);

  const leaderboard: LeaderboardEntry[] = rawLeaderboard.map((item) => {
    const liveMatch = store.leaderboard.find((l) => l.experiment_id === item.experiment_id);
    return liveMatch ? { ...item, ...liveMatch, generalization_test: liveMatch.generalization_test || item.generalization_test } : item;
  });

  const handleStartSweep = () => {
    store.startJob("run_optimizer", "Hyperparameter Sweep Execution", "run_optimizer", {
      strategy: session.strategy,
      max_trials: session.maxTrials,
      use_previous_history: session.usePreviousOptimizationHistory ?? true,
      usePreviousOptimizationHistory: session.usePreviousOptimizationHistory ?? true,
      learn_from_generalization_test: session.learnFromGeneralizationTest ?? false,
      learnFromGeneralizationTest: session.learnFromGeneralizationTest ?? false,
    });
  };

  const validLeaderboard = leaderboard.filter(item => item.result?.metrics_valid === true && item.status === "completed" && item.composite_score !== null);
  const bestScore = validLeaderboard.length > 0 ? validLeaderboard[0].composite_score : null;

  const selectedTrial = selectedTrialId ? leaderboard.find(t => t.experiment_id === selectedTrialId) : null;
  const selectedRank = selectedTrialId ? leaderboard.findIndex(t => t.experiment_id === selectedTrialId) + 1 : 0;


  return (
    <div id="optimizer-view-container" className="space-y-4">
      {/* Header Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Hyperparameter Optimization Engine
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Bayesian & LLM-guided hyperparameter parameter space exploration
            </p>
          </div>
        </div>

        <button
          id="run-sweep-btn"
          onClick={handleStartSweep}
          disabled={!!activeJob}
          className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center space-x-1.5 shadow-sm"
        >
          <Sparkles className={`w-3.5 h-3.5 ${activeJob ? "animate-spin" : ""}`} />
          <span>{activeJob ? "Running Sweep Job..." : "Start Sweep"}</span>
        </button>
      </div>

      {/* Active Job Progress Card */}
      {activeJob && (
        <div className="mb-4">
          <JobProgressCard job={activeJob} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Sweep Controls */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-lg p-3.5 space-y-3.5 shadow-sm">
          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Search Strategy
            </span>
            <span className="text-[9px] font-mono text-blue-400 bg-blue-950/60 px-1.5 py-0.5 rounded border border-blue-800/40">
              OPTIM_SPEC
            </span>
          </div>

          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
              Search Strategy
            </label>
            <select
              id="search-strategy-select"
              value={session.strategy}
              onChange={(e) =>
                store.updateSession("optimizer", { strategy: e.target.value as any })
              }
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
            >
              <option value="llm_guided">LLM-Guided Strategy (Adaptive Feedback)</option>
              <option value="grid">Grid Search (Exhaustive Combinations)</option>
              <option value="random">Random Sampling Search</option>
            </select>
          </div>

          <div>
            <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
              Max Optimization Trials:{" "}
              <span className="text-blue-400 font-mono">{session.maxTrials}</span>
            </label>
            <input
              id="max-trials-slider"
              type="range"
              min={2}
              max={10}
              value={session.maxTrials}
              onChange={(e) =>
                store.updateSession("optimizer", { maxTrials: Number(e.target.value) })
              }
              className="w-full accent-blue-500"
            />
          </div>

          {/* Historical Trials Learning Toggle */}
          <div className="bg-slate-950 p-2.5 rounded border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <label htmlFor="toggle-previous-history" className="text-[10px] font-bold text-slate-300 uppercase tracking-wider font-mono cursor-pointer">
                Learn from Previous Trials
              </label>
              <input
                id="toggle-previous-history"
                type="checkbox"
                checked={session.usePreviousOptimizationHistory ?? true}
                onChange={(e) =>
                  store.updateSession("optimizer", { usePreviousOptimizationHistory: e.target.checked })
                }
                className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-blue-500 cursor-pointer accent-blue-500"
              />
            </div>
            <p className="text-[9px] text-slate-400 font-mono leading-tight">
              {(session.usePreviousOptimizationHistory ?? true)
                ? "Adaptive optimizer uses retained valid trials from past runs of this KB to guide search and avoid duplicate exploration."
                : "Strict experiment isolation active. Optimizer starts fresh with zero memory of previous runs."}
            </p>
          </div>

          {/* Learn from Generalization Tests Toggle */}
          <div className="bg-slate-950 p-2.5 rounded border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <label htmlFor="toggle-generalization-learning" className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider font-mono cursor-pointer flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" />
                Learn from Generalization Tests
              </label>
              <input
                id="toggle-generalization-learning"
                type="checkbox"
                checked={session.learnFromGeneralizationTest ?? false}
                onChange={(e) =>
                  store.updateSession("optimizer", { learnFromGeneralizationTest: e.target.checked })
                }
                className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-emerald-600 focus:ring-emerald-500 cursor-pointer accent-emerald-500"
              />
            </div>
            <p className="text-[9px] text-slate-400 font-mono leading-tight">
              {(session.learnFromGeneralizationTest ?? false)
                ? "LLM optimizer incorporates summarized holdout validation results to penalize fragile pipelines and favor robust architectures."
                : "Holdout validation memory disabled for this sweep."}
            </p>
          </div>

          <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-[10px] text-slate-400 font-mono space-y-1.5">
            <span className="font-bold text-slate-300 block uppercase">
              Optimizer Bounds Check:
            </span>
            <div className="flex items-center space-x-1.5 text-emerald-400">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Bounds & Duplicate Filtering</span>
            </div>
            <div className="flex items-center space-x-1.5 text-emerald-400">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Deterministic Artifact Reuse</span>
            </div>
            <div className="flex items-center space-x-1.5 text-emerald-400">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Holdout Dataset Isolation</span>
            </div>
          </div>
        </div>

        {/* Sweep Results & Leaderboard */}
        <div className="lg:col-span-8 space-y-3">
          {leaderboard.length > 0 ? (
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4 shadow-sm">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <Trophy className="w-5 h-5 text-amber-400" />
                  <div>
                    <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-emerald-400">
                      SWEEP_LEADERBOARD
                    </span>
                    <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                      Evaluated {leaderboard.length} Optimization Trials
                    </h2>
                  </div>
                </div>

                <div className="flex items-center space-x-4 text-right">
                  <div>
                    <span className="text-xl font-bold font-mono text-blue-400">
                      {bestScore !== null ? `${(bestScore * 100).toFixed(1)}%` : "N/A"}
                    </span>
                    <span className="block text-[9px] text-slate-500 font-mono uppercase">
                      BEST COMPOSITE
                    </span>
                  </div>
                  <button
                    onClick={() => store.clearLeaderboard()}
                    className="p-2 bg-slate-800 hover:bg-red-900/50 text-slate-400 hover:text-red-400 rounded transition-colors"
                    title="Clear Leaderboard"
                  >
                    <Trash className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Leaderboard Table */}
              <div>
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Trial Leaderboard
                </h3>
                <div className="overflow-x-auto border border-slate-800 rounded">
                  <table className="w-full text-left text-[11px] text-slate-300 font-mono">
                    <thead className="bg-slate-800/70 text-slate-400 uppercase text-[9px]">
                      <tr>
                        <th className="px-3 py-2">Rank</th>
                        <th className="px-3 py-2">Optimization Score</th>
                        <th className="px-3 py-2">Generalization</th>
                        <th className="px-3 py-2">Hit Rate</th>
                        <th className="px-3 py-2">Answer Correctness</th>
                        <th className="px-3 py-2">Chunk Size</th>
                        <th className="px-3 py-2">Retriever</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {leaderboard.map((item, idx) => {
                        const isSelected = selectedTrialId === item.experiment_id;
                        const genTest = item.generalization_test;
                        const correctness = item.answer_correctness ?? item.answer_faithfulness;
                        return (
                          <tr 
                            key={item.experiment_id || idx} 
                            onClick={() => setSelectedTrialId(item.experiment_id)}
                            className={`cursor-pointer transition-colors ${isSelected ? "bg-slate-800/80 border-l-2 border-l-blue-500" : "hover:bg-slate-800/40 border-l-2 border-l-transparent"}`}
                          >
                            <td className="px-3 py-2 font-bold text-slate-400">#{idx + 1}</td>
                            <td className="px-3 py-2 font-bold text-blue-400">
                              {item.status === 'completed' && item.result?.metrics_valid && item.composite_score !== null ? (
                                `${(item.composite_score * 100).toFixed(1)}%`
                              ) : (
                                <span className="text-red-400 font-bold" title={item.result?.failure_reason || "FAILED"}>FAILED</span>
                              )}
                            </td>
                            <td className="px-3 py-2">
                              {genTest && genTest.status === "completed" && genTest.generalization_composite_score !== null ? (
                                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/60 text-emerald-300 font-semibold text-[10px]">
                                  <ShieldCheck className="w-3 h-3 text-emerald-400" />
                                  {`${(genTest.generalization_composite_score * 100).toFixed(1)}%`}
                                  {genTest.score_delta !== null && (
                                    <span className={`text-[9px] ${genTest.score_delta >= 0 ? "text-emerald-400" : "text-amber-400"}`}>
                                      ({genTest.score_delta >= 0 ? "+" : ""}{(genTest.score_delta * 100).toFixed(1)}%)
                                    </span>
                                  )}
                                </span>
                              ) : (
                                <span className="text-slate-600 text-[10px]">—</span>
                              )}
                            </td>
                            <td className="px-3 py-2">
                              {item.retrieval_hit_rate !== null && item.retrieval_hit_rate !== undefined ? `${(item.retrieval_hit_rate * 100).toFixed(1)}%` : "N/A"}
                            </td>
                            <td className="px-3 py-2">
                              {correctness !== null && correctness !== undefined ? `${(correctness * 100).toFixed(1)}%` : "N/A"}
                            </td>
                            <td className="px-3 py-2">
                              {item.config?.chunking_config?.chunk_size || 512} tokens
                            </td>
                            <td className="px-3 py-2 uppercase">
                              {item.config?.retriever_config?.strategy || "hybrid"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-16 bg-slate-900 border border-slate-800 rounded p-4 text-slate-500 text-xs font-mono">
              Select search strategy and click{" "}
              <strong className="text-blue-400">START SWEEP</strong> to launch automated background
              job.
            </div>
          )}
        </div>
      </div>

      {selectedTrial && (
        <TrialDrawer 
          trial={selectedTrial} 
          leaderboard={leaderboard} 
          rank={selectedRank} 
          onClose={() => setSelectedTrialId(null)} 
        />
      )}
    </div>
  );
};
