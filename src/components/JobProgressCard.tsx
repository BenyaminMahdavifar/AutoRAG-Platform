import React, { useState, useEffect } from "react";
import {
  Play,
  XCircle,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Terminal,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Loader2,
  Sparkles
} from "lucide-react";
import { Job } from "../types";
import { appState } from "../state/appState";

interface JobProgressCardProps {
  job: Job;
  compact?: boolean;
}

export const JobProgressCard: React.FC<JobProgressCardProps> = ({ job, compact = false }) => {
  const [showLogs, setShowLogs] = useState(!compact);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const startMs = new Date(job.started_at).getTime();
    const updateElapsed = () => {
      const nowMs = job.status === "Running" ? Date.now() : new Date(job.updated_at).getTime();
      setElapsedSeconds(Math.max(0, Math.floor((nowMs - startMs) / 1000)));
    };
    updateElapsed();

    if (job.status === "Running") {
      const interval = setInterval(updateElapsed, 1000);
      return () => clearInterval(interval);
    }
  }, [job.started_at, job.updated_at, job.status]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const getStageTimeline = () => {
    const stagesForType: Record<string, string[]> = {
      scan_kb: ["Scanning Documents Directory", "Verifying Document Checksums", "Manifest Saved"],
      upload_doc: ["Writing Document File", "Scanning Updated Directory", "Directory Manifest Updated"],
      build_dataset: ["Scan Knowledge Base", "Chunking Documents", "Generating Vector Embeddings", "Synthesizing Ground Truth Q&A", "Dataset Version Saved"],
      run_experiment: ["Scan Knowledge Base", "Building Vector Index", "Loading Evaluation Dataset", "Executing Retrieval & Generation Benchmark", "Trial Completed"],
      run_optimizer: ["Initializing Search Space", "Building Index & Dataset", "Running Optimization Trials", "Optimization Sweep Complete"],
      export_rag: ["Preparing export", "Building Knowledge Base", "Chunking documents", "Generating embeddings", "Building vector database", "Writing configuration", "Generating source code", "Validating package", "Compressing ZIP"],
      export_reports: ["Gathering Experiment Trials", "Generating Markdown & CSV Reports", "Rendering HTML Dashboard", "All Reports Exported"],
    };

    const defaultStages = stagesForType[job.type] || [
      "Initialization",
      "Processing Payload",
      "Executing Backend Subsystem",
      "Finalizing Artifacts"
    ];

    return defaultStages.map((stageName) => {
      const isCompleted =
        job.completed_stages?.includes(stageName) ||
        job.status === "Completed" ||
        defaultStages.indexOf(stageName) < defaultStages.indexOf(job.current_stage);
      const isCurrent = job.current_stage === stageName && job.status === "Running";

      return {
        name: stageName,
        status: isCompleted ? "completed" : isCurrent ? "current" : "pending",
      };
    });
  };

  const stageTimeline = getStageTimeline();

  return (
    <div
      id={`job-card-${job.job_id}`}
      className={`bg-slate-900 border rounded-lg overflow-hidden transition-all shadow-md ${
        job.status === "Failed"
          ? "border-rose-800/80 bg-rose-950/10"
          : job.status === "Completed"
          ? "border-emerald-800/60"
          : "border-blue-800/80"
      }`}
    >
      {/* Top Header Status Bar */}
      <div className="bg-slate-950 px-3.5 py-2 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2.5">
          {job.status === "Running" && <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />}
          {job.status === "Completed" && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
          {job.status === "Failed" && <AlertTriangle className="w-4 h-4 text-rose-400" />}
          {job.status === "Cancelled" && <XCircle className="w-4 h-4 text-amber-400" />}

          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-white uppercase font-sans tracking-wide">
                {job.title}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase">
                {job.type}
              </span>
            </div>
            <span className="text-[10px] font-mono text-slate-400 block mt-0.5">
              ID: {job.job_id} | STARTED: {new Date(job.started_at).toLocaleTimeString()}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="flex items-center space-x-1 text-slate-400 text-[11px]">
            <Clock className="w-3 h-3 text-slate-500" />
            <span>{formatTime(elapsedSeconds)}</span>
          </div>

          <span
            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
              job.status === "Running"
                ? "bg-blue-500/20 text-blue-400 border border-blue-500/30 animate-pulse"
                : job.status === "Completed"
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : job.status === "Failed"
                ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            }`}
          >
            {job.status}
          </span>

          {job.status === "Running" && (
            <button
              onClick={() => appState.cancelJob(job.job_id)}
              className="px-2 py-1 bg-rose-900/40 hover:bg-rose-800/80 text-rose-300 border border-rose-800/60 rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1"
            >
              <XCircle className="w-3 h-3" />
              <span>Cancel</span>
            </button>
          )}

          {job.status === "Failed" && (
            <button
              onClick={() => appState.retryJob(job.job_id)}
              className="px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Progress Body */}
      <div className="p-3.5 space-y-3">
        {/* Progress Bar & Stage Status */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-slate-300 font-semibold flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" />
              <span>{job.current_stage || "Processing Stage..."}</span>
            </span>
            <span className="font-bold text-blue-400">{job.progress}%</span>
          </div>

          <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800/80">
            <div
              className={`h-full transition-all duration-300 ${
                job.status === "Failed"
                  ? "bg-rose-500"
                  : job.status === "Completed"
                  ? "bg-emerald-500"
                  : "bg-gradient-to-r from-blue-600 to-indigo-500"
              }`}
              style={{ width: `${Math.min(100, Math.max(0, job.progress))}%` }}
            ></div>
          </div>
        </div>

        {/* Stage Timeline */}
        <div className="pt-1">
          <div className="text-[10px] font-mono text-slate-500 uppercase font-bold mb-1.5">
            Stage Timeline
          </div>
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
            {stageTimeline.map((stg, i) => (
              <div
                key={i}
                className={`px-2 py-1 rounded border flex items-center space-x-1 transition-all ${
                  stg.status === "completed"
                    ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-300"
                    : stg.status === "current"
                    ? "bg-blue-950/60 border-blue-500 text-blue-300 font-bold animate-pulse"
                    : "bg-slate-950/40 border-slate-800/60 text-slate-500 opacity-60"
                }`}
              >
                <span>
                  {stg.status === "completed" ? "✓" : stg.status === "current" ? "⟳" : "□"}
                </span>
                <span>{stg.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Error Recovery Diagnostic Panel if Failed */}
        {job.status === "Failed" && (
          <div className="bg-rose-950/30 border border-rose-800/80 rounded p-3 text-xs space-y-1.5 font-mono">
            <div className="text-rose-400 font-bold uppercase flex items-center space-x-1.5">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Diagnostic Failure Report</span>
            </div>
            <p className="text-slate-300">
              <strong className="text-rose-400">Failed Stage:</strong> {job.current_stage}
            </p>
            <p className="text-slate-300 break-words whitespace-pre-wrap">
              <strong className="text-rose-400">Reason:</strong> {job.error || "Execution interrupted"}
            </p>
            {job.suggested_fix && (
              <p className="text-amber-300 bg-amber-950/40 p-2 rounded border border-amber-800/60 text-[11px] mt-1 break-words whitespace-pre-wrap">
                <strong className="text-amber-400 block mb-0.5">SUGGESTED FIX:</strong>
                {job.suggested_fix}
              </p>
            )}
          </div>
        )}

        {/* Collapsible Live Logs Stream */}
        <div>
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="text-[10px] font-mono text-slate-400 hover:text-slate-200 flex items-center space-x-1 py-1"
          >
            <Terminal className="w-3 h-3 text-blue-400" />
            <span>Live Logs Stream ({job.logs?.length || 0} lines)</span>
            {showLogs ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>

          {showLogs && (
            <div className="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-slate-300 max-h-40 overflow-y-auto space-y-1 mt-1">
              {job.logs && job.logs.length > 0 ? (
                job.logs.map((logLine, idx) => (
                  <div key={idx} className="leading-relaxed whitespace-pre-wrap break-all font-mono">
                    {logLine}
                  </div>
                ))
              ) : (
                <span className="text-slate-600 italic">No logs generated yet...</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
