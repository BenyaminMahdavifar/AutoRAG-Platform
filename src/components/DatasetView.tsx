import React from "react";
import { Layers, RefreshCw, HelpCircle } from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";

export const DatasetView: React.FC = () => {
  const store = useAppState();
  const session = store.sessions.dataset;
  const dataset = store.dataset;

  const activeJob = store.jobs.find(
    (j) => j.type === "build_dataset" && (j.status === "Running" || j.status === "Queued")
  );

  const filterText = session.filterText;

  const filteredItems =
    dataset?.items.filter(
      (item) =>
        item.question.toLowerCase().includes(filterText.toLowerCase()) ||
        item.ground_truth.toLowerCase().includes(filterText.toLowerCase())
    ) || [];

  const handleGenerate = () => {
    store.startJob("build_dataset", "Synthetic Dataset Generation", "build_dataset", {});
  };

  return (
    <div id="dataset-view-container" className="space-y-4">
      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Dataset Builder & Version Engine
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Synthetic ground-truth Q&A pair extraction, version tracking & dataset audit
            </p>
          </div>
        </div>

        <button
          id="build-dataset-btn"
          onClick={handleGenerate}
          disabled={!!activeJob}
          className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center space-x-1.5 shadow-sm"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${activeJob ? "animate-spin" : ""}`} />
          <span>{activeJob ? "Building Dataset Job..." : "Generate Dataset"}</span>
        </button>
      </div>

      {/* Active Job Card */}
      {activeJob && (
        <div className="mb-4">
          <JobProgressCard job={activeJob} />
        </div>
      )}

      {/* Dataset Summary Meta */}
      {dataset && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Dataset ID</span>
            <span className="text-xs font-mono font-bold text-blue-400">{dataset.dataset_id}</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Version</span>
            <span className="text-xs font-mono font-bold text-emerald-400">v{dataset.version}</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Test Cases</span>
            <span className="text-xs font-mono font-bold text-white">{dataset.items.length} PAIRS</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Created At</span>
            <span className="text-[11px] font-mono text-slate-400">
              {new Date(dataset.created_at).toISOString().slice(0, 19).replace("T", " ")}
            </span>
          </div>
        </div>
      )}

      {/* Filter and Item List */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider">
            Evaluation Test Cases ({filteredItems.length})
          </h2>
          <input
            id="dataset-search-input"
            type="text"
            placeholder="Search test cases..."
            value={filterText}
            onChange={(e) => store.updateSession("dataset", { filterText: e.target.value })}
            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-white focus:border-blue-500 focus:outline-none font-mono w-60"
          />
        </div>

        {!dataset || dataset.items.length === 0 ? (
          <div className="text-center py-10 bg-slate-950/50 rounded border border-slate-800">
            <HelpCircle className="w-6 h-6 text-slate-600 mx-auto mb-2" />
            <p className="text-xs font-medium text-slate-300 font-mono">No Dataset Version Initialized</p>
            <p className="text-[11px] text-slate-500 max-w-sm mx-auto mt-1 mb-3">
              Trigger dataset generation to extract ground-truth question/answer test pairs from your knowledge base.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredItems.map((item, idx) => (
              <div
                key={item.item_id || idx}
                className="bg-slate-950/60 border border-slate-800/80 rounded p-3 transition-all hover:border-slate-700"
              >
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-mono text-[10px] font-bold border border-blue-500/20">
                      CASE_{String(idx + 1).padStart(3, "0")}
                    </span>
                    <h3 className="text-xs font-semibold text-white">{item.question}</h3>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono whitespace-nowrap">
                    DOC: {item.metadata?.filename || "KB_FILE"}
                  </span>
                </div>

                <div className="bg-slate-900 p-2.5 rounded border border-slate-800/60 text-xs">
                  <span className="text-[9px] font-bold text-emerald-400 uppercase tracking-wider block mb-1 font-mono">
                    Ground Truth Fact:
                  </span>
                  <p className="text-slate-300 leading-relaxed text-[11px] font-mono">{item.ground_truth}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
