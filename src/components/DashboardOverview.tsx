import React from "react";
import { Database, Layers, Sliders, ArrowUpRight, Zap, Activity, HardDrive, Cpu, Trophy, Clock } from "lucide-react";
import { useAppState } from "../state/appState";

export const DashboardOverview: React.FC = () => {
  const store = useAppState();
  const manifest = store.manifest;
  const leaderboard = store.leaderboard;
  const cacheMeta = store.cacheMetadata;
  const activeProfile =
    store.connectionProfiles.find(
      (p) => p.profile_id === store.workspaceState.activeConnectionProfileId
    ) || store.connectionProfiles[0];

  const validLeaderboard = leaderboard.filter(
    (item) => item.status === "completed" && item.result?.metrics_valid !== false && item.composite_score !== null
  );
  const topScore = validLeaderboard.length > 0 ? validLeaderboard[0].composite_score : null;
  const bestChunk = validLeaderboard.length > 0 ? validLeaderboard[0].config?.chunking_config?.chunk_size || 512 : 512;
  const bestStrategy = validLeaderboard.length > 0 ? validLeaderboard[0].config?.retriever_config?.strategy || "hybrid" : "hybrid";

  const runningJobs = store.jobs.filter((j) => j.status === "Running" || j.status === "Queued");

  return (
    <div id="dashboard-container" className="space-y-4">
      {/* Top Header Control Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg px-4 py-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            <span className="text-xs font-bold text-white tracking-wide uppercase font-sans">
              WORKSPACE: STATEFUL_DESKTOP_SESSION
            </span>
          </div>
          <span className="h-3 w-px bg-slate-800"></span>
          <span className="text-[10px] font-mono text-slate-400 uppercase hidden sm:inline-block">
            PROFILE: {activeProfile.name}
          </span>
          {runningJobs.length > 0 && (
            <span className="px-2 py-0.5 text-[9px] font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30 rounded font-bold animate-pulse">
              {runningJobs.length} JOB(S) IN PROGRESS
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2 font-mono text-xs">
          <button
            id="inspect-kb-btn"
            onClick={() => store.setActiveTab("kb")}
            className="px-3 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-200 transition-all flex items-center space-x-1.5"
          >
            <Database className="w-3.5 h-3.5 text-slate-400" />
            <span>KNOWLEDGE_BASE</span>
          </button>
          <button
            id="start-sweep-btn"
            onClick={() => store.setActiveTab("optimizer")}
            className="px-3.5 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center space-x-1.5 shadow-sm"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Run Sweep</span>
          </button>
        </div>
      </div>

      {/* Top 4 Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div id="metric-docs-card" className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 shadow-sm">
          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 font-mono">
            Knowledge Base Cache
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {manifest?.total_docs || cacheMeta.documentsCount || 0}{" "}
            <span className="text-xs text-slate-400 font-normal">DOCS</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1 font-mono">
            TOTAL SIZE: {((manifest?.total_size_bytes || 0) / 1024).toFixed(1)} KB
          </div>
        </div>

        <div id="metric-trials-card" className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 shadow-sm">
          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 font-mono">
            Evaluated Experiments
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {leaderboard.length || cacheMeta.experimentsCount || 0}{" "}
            <span className="text-xs text-slate-400 font-normal">TRIALS</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1 font-mono">PERSISTED IN WORKSPACE</div>
        </div>

        <div id="metric-topscore-card" className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 shadow-sm">
          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 font-mono">
            Top Composite Score
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono">
            {topScore !== null ? `${(topScore * 100).toFixed(1)}%` : "N/A"}
          </div>
          <div className="text-[10px] text-slate-500 mt-1 font-mono">WEIGHTED HARMONIC MEAN</div>
        </div>

        <div id="metric-recommendation-card" className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 shadow-sm">
          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 font-mono">
            Optimal Config
          </div>
          <div className="text-2xl font-bold text-white font-mono uppercase tracking-tight">
            {validLeaderboard.length > 0 ? `${bestChunk}T / ${bestStrategy}` : "N/A"}
          </div>
          <div className="text-[10px] text-slate-500 mt-1 font-mono">CHUNK SIZE & RETRIEVAL STRATEGY</div>
        </div>
      </div>

      {/* Main Grid Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left System Trace & Quick Overview */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-lg flex flex-col overflow-hidden shadow-sm">
          <div className="bg-slate-800/50 px-3.5 py-2.5 border-b border-slate-800 flex justify-between items-center">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-200 font-sans">
              System Telemetry & Cache
            </span>
            <span className="text-[10px] font-mono text-blue-400">ONLINE</span>
          </div>
          <div className="p-3.5 space-y-3.5">
            <div>
              <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1 font-mono">
                Active Provider Profile
              </label>
              <div className="bg-slate-950 p-2.5 border border-slate-800/80 font-mono text-[11px] text-blue-300 rounded space-y-0.5">
                <div><strong className="text-slate-400">NAME:</strong> {activeProfile.name}</div>
                <div><strong className="text-slate-400">PROVIDER:</strong> {activeProfile.provider}</div>
                <div><strong className="text-slate-400">MODEL:</strong> {activeProfile.default_model}</div>
              </div>
            </div>

            <div>
              <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1 font-mono">
                Workspace Cache Stats
              </label>
              <div className="space-y-1.5 font-mono text-[11px]">
                <div className="flex justify-between items-center bg-slate-950/60 p-2 border border-slate-800/60 rounded">
                  <span className="text-slate-400">Scanned Documents</span>
                  <span className="text-white font-bold">{cacheMeta.documentsCount} Files</span>
                </div>
                <div className="flex justify-between items-center bg-slate-950/60 p-2 border border-slate-800/60 rounded">
                  <span className="text-slate-400">Synthetic Test Pairs</span>
                  <span className="text-emerald-400 font-bold">{cacheMeta.datasetItemsCount} Pairs</span>
                </div>
                <div className="flex justify-between items-center bg-slate-950/60 p-2 border border-slate-800/60 rounded">
                  <span className="text-slate-400">Exported Artifacts</span>
                  <span className="text-blue-400 font-bold">{cacheMeta.reportsCount} Reports</span>
                </div>
              </div>
            </div>

            <div className="p-2.5 bg-blue-950/40 border border-blue-800/60 rounded text-[11px] font-mono text-slate-300 space-y-1">
              <div className="text-[10px] text-blue-400 uppercase font-bold">State Engine Guarantees</div>
              <p className="leading-relaxed text-slate-300 text-[10px]">
                • Changing tabs never loses form inputs or session data.
              </p>
              <p className="leading-relaxed text-slate-300 text-[10px]">
                • Long-running operations execute as background jobs with real-time logs.
              </p>
            </div>
          </div>
        </div>

        {/* Right Leaderboard Table */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-white">
                Optimization Leaderboard
              </h2>
              <p className="text-[10px] text-slate-400 font-mono">Ranked by Composite Evaluation Score</p>
            </div>
            <button
              id="view-all-reports-btn"
              onClick={() => store.setActiveTab("reports")}
              className="text-[11px] text-blue-400 hover:text-blue-300 font-mono flex items-center space-x-1"
            >
              <span>REPORT_ENGINE</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {leaderboard.length === 0 ? (
            <div className="text-center py-10 bg-slate-950/50 rounded border border-slate-800/80 flex-1 flex flex-col items-center justify-center">
              <Sliders className="w-7 h-7 text-slate-600 mb-2" />
              <p className="text-xs font-medium text-slate-300 font-mono">No experiment trials executed</p>
              <p className="text-[11px] text-slate-500 max-w-sm my-2">
                Launch a hyperparameter sweep to evaluate chunk sizes, vector search strategies, and answer correctness.
              </p>
              <button
                id="run-first-sweep-btn"
                onClick={() => store.setActiveTab("optimizer")}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded uppercase tracking-wider transition-all"
              >
                Run First Sweep
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto border border-slate-800 rounded">
              <table className="w-full text-left text-[11px] text-slate-300">
                <thead className="bg-slate-800/70 text-slate-400 uppercase tracking-wider text-[9px] font-mono border-b border-slate-800">
                  <tr>
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Trial ID</th>
                    <th className="px-3 py-2">Composite</th>
                    <th className="px-3 py-2">Hit Rate</th>
                    <th className="px-3 py-2">Answer Correctness</th>
                    <th className="px-3 py-2">Chunk Size</th>
                    <th className="px-3 py-2">Strategy</th>
                    <th className="px-3 py-2">Top K</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {leaderboard.map((item, idx) => (
                    <tr key={item.experiment_id || idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-3 py-2 font-bold text-slate-400">#{idx + 1}</td>
                      <td className="px-3 py-2 text-blue-400 font-semibold">{item.experiment_id}</td>
                      <td className="px-3 py-2 font-bold text-emerald-400">
                        {item.status === "completed" && item.result?.metrics_valid !== false && item.composite_score !== null ? (
                          `${(item.composite_score * 100).toFixed(1)}%`
                        ) : (
                          <span className="text-red-400 font-bold" title={item.result?.failure_reason || "Trial Failed"}>FAILED</span>
                        )}
                      </td>
                      <td className="px-3 py-2">{((item.retrieval_hit_rate || 0) * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{(((item.answer_correctness ?? item.answer_faithfulness) || 0) * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{item.config?.chunking_config?.chunk_size || 512} tokens</td>
                      <td className="px-3 py-2">
                        <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-300 text-[10px] uppercase border border-blue-500/20">
                          {item.config?.retriever_config?.strategy || "hybrid"}
                        </span>
                      </td>
                      <td className="px-3 py-2">{item.config?.retriever_config?.top_k || 4}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
