import React from "react";
import { FileText, Download, FileSpreadsheet, Code2, Globe, RefreshCw } from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";

export const ReportsView: React.FC = () => {
  const store = useAppState();
  const reports = store.reports;

  const activeJob = store.jobs.find(
    (j) => j.type === "export_reports" && (j.status === "Running" || j.status === "Queued")
  );

  const handleExportAll = () => {
    store.startJob("export_reports", "Export Markdown, CSV & HTML Reports", "export_reports", {});
  };

  return (
    <div id="reports-view-container" className="space-y-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Reports & Artifact Exporter
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Export evaluation summaries, leaderboards & metrics in JSON, Markdown, CSV, and HTML formats
            </p>
          </div>
        </div>

        <button
          id="export-all-reports-btn"
          onClick={handleExportAll}
          disabled={!!activeJob}
          className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center space-x-1.5 shadow-sm"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${activeJob ? "animate-spin" : ""}`} />
          <span>{activeJob ? "Generating Export Job..." : "Generate All Formats"}</span>
        </button>
      </div>

      {/* Active Job Progress Card */}
      {activeJob && (
        <div className="mb-4">
          <JobProgressCard job={activeJob} />
        </div>
      )}

      {/* Report Summary Leaderboard */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm overflow-hidden">
        <h2 className="text-xs font-bold text-white uppercase tracking-wider mb-3 font-mono">
          Detailed Pipeline Configuration Summary
        </h2>
        {store.leaderboard.length === 0 ? (
          <div className="text-center py-8 bg-slate-950/50 rounded border border-slate-800 text-xs text-slate-500 font-mono">
            No pipeline trials available to display.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px] text-slate-300 font-mono whitespace-nowrap">
              <thead className="bg-slate-800/70 text-slate-400 uppercase text-[9px]">
                <tr>
                  <th className="px-3 py-2">Rank</th>
                  <th className="px-3 py-2">Score</th>
                  <th className="px-3 py-2">LLM Provider</th>
                  <th className="px-3 py-2">Embedding</th>
                  <th className="px-3 py-2">Chunking</th>
                  <th className="px-3 py-2">Retriever</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {store.leaderboard.filter(item => item.status === "completed" && item.result?.metrics_valid === true && item.composite_score !== null).map((item, idx) => {
                  const chunkCfg = item.config?.chunking_config;
                  const retCfg = item.config?.retriever_config;
                  const llmCfg = item.config?.llm_config;
                  const embCfg = item.config?.embedding_config;

                  return (
                    <tr key={item.experiment_id || idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-3 py-2 font-bold text-slate-400">#{idx + 1}</td>
                      <td className="px-3 py-2 font-bold text-emerald-400">
                        {item.composite_score !== null ? `${(item.composite_score * 100).toFixed(1)}%` : "N/A"}
                      </td>
                      <td className="px-3 py-2">
                        {llmCfg?.provider} <span className="text-slate-500">/</span> {llmCfg?.model_name}
                      </td>
                      <td className="px-3 py-2 text-slate-400">
                        {embCfg?.model_name}
                      </td>
                      <td className="px-3 py-2">
                        <span className="text-blue-400">{chunkCfg?.strategy}</span>
                        <div className="text-[9px] text-slate-500 mt-0.5">
                          Size: {chunkCfg?.chunk_size} | Overlap: {chunkCfg?.chunk_overlap}
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-400 rounded text-[9px] uppercase border border-blue-500/20">
                          {retCfg?.strategy || "hybrid"}
                        </span>
                        <div className="text-[9px] text-slate-500 mt-1">
                          Dist: {retCfg?.distance_metric} | K: {retCfg?.top_k}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Export Format Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center space-x-2.5 shadow-sm">
          <div className="p-2 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Code2 className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-mono uppercase">JSON Format</h3>
            <p className="text-[10px] text-slate-500 font-mono">Structured trial specs</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center space-x-2.5 shadow-sm">
          <div className="p-2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-mono uppercase">Markdown Spec</h3>
            <p className="text-[10px] text-slate-500 font-mono">Formatted tables</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center space-x-2.5 shadow-sm">
          <div className="p-2 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <FileSpreadsheet className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-mono uppercase">CSV Export</h3>
            <p className="text-[10px] text-slate-500 font-mono">Tabular metrics</p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex items-center space-x-2.5 shadow-sm">
          <div className="p-2 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Globe className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-mono uppercase">HTML Dashboard</h3>
            <p className="text-[10px] text-slate-500 font-mono">Standalone report page</p>
          </div>
        </div>
      </div>

      {/* Generated Reports Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
        <h2 className="text-xs font-bold text-white uppercase tracking-wider mb-3 font-mono">
          Exported Report Artifacts
        </h2>

        {reports.length === 0 ? (
          <div className="text-center py-12 bg-slate-950/50 rounded border border-slate-800 text-xs text-slate-500 font-mono">
            Click <strong className="text-blue-400">Generate All Formats</strong> above to compile and write artifacts to disk.
          </div>
        ) : (
          <div className="space-y-1.5">
            {reports.map((report: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2.5 bg-slate-950/60 rounded border border-slate-800 text-xs text-slate-300 font-mono"
              >
                <div className="flex items-center space-x-2.5">
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span className="font-bold text-white">{report.name}</span>
                  <span className="text-[10px] text-slate-500">({report.size_bytes} bytes)</span>
                </div>

                <a
                  href={`/api/reports/file/${report.name}`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-blue-300 rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
