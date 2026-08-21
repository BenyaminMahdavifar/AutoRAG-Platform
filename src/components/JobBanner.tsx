import React, { useState } from "react";
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
  Trash2,
  Sparkles,
  Layers
} from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";

export const JobBanner: React.FC = () => {
  const store = useAppState();
  const [isOpen, setIsOpen] = useState(false);

  const activeJobs = store.jobs.filter((j) => j.status === "Running" || j.status === "Queued");
  const pastJobs = store.jobs.filter((j) => j.status !== "Running" && j.status !== "Queued");

  if (store.jobs.length === 0) return null;

  return (
    <div id="job-banner-container" className="bg-[#0b1120] border-b border-slate-800 text-xs font-mono">
      {/* Top Bar Summary Strip */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 flex items-center justify-between cursor-pointer hover:bg-slate-900/60 transition-colors"
      >
        <div className="flex items-center space-x-3">
          {activeJobs.length > 0 ? (
            <div className="flex items-center space-x-2 text-blue-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span className="font-bold uppercase tracking-wider text-[11px]">
                {activeJobs.length} ACTIVE BACKGROUND JOB{activeJobs.length > 1 ? "S" : ""} RUNNING
              </span>
            </div>
          ) : (
            <div className="flex items-center space-x-2 text-slate-400">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[11px] uppercase tracking-wider text-slate-300">
                JOB MANAGER ({store.jobs.length} TOTAL LOGGED)
              </span>
            </div>
          )}

          {activeJobs.length > 0 && (
            <span className="hidden sm:inline-block px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20 text-[10px]">
              {activeJobs[0].title} ({activeJobs[0].progress}%)
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3 text-[11px]">
          {activeJobs.length > 0 && (
            <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden md:block border border-slate-700">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${activeJobs[0].progress}%` }}
              ></div>
            </div>
          )}

          <div className="flex items-center space-x-1 text-slate-400 font-bold uppercase">
            <span>{isOpen ? "HIDE JOBS" : "VIEW JOBS DRAWER"}</span>
            {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </div>
        </div>
      </div>

      {/* Expanded Drawer Body */}
      {isOpen && (
        <div className="border-t border-slate-800/80 bg-slate-950 p-4 max-w-7xl mx-auto space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-blue-400" />
              <h2 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
                Background Job Queue & Operational Status
              </h2>
            </div>

            <button
              onClick={() => store.clearJobHistory()}
              className="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded text-[10px] text-slate-400 hover:text-slate-200 transition-all flex items-center space-x-1"
            >
              <Trash2 className="w-3 h-3 text-rose-400" />
              <span>CLEAR FINISHED JOBS</span>
            </button>
          </div>

          {/* Active Jobs */}
          {activeJobs.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">
                Active Operations ({activeJobs.length})
              </h3>
              <div className="grid grid-cols-1 gap-3">
                {activeJobs.map((job) => (
                  <JobProgressCard key={job.job_id} job={job} />
                ))}
              </div>
            </div>
          )}

          {/* Past Jobs */}
          {pastJobs.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                Completed & Past Jobs ({pastJobs.length})
              </h3>
              <div className="grid grid-cols-1 gap-3 max-h-96 overflow-y-auto pr-1">
                {pastJobs.map((job) => (
                  <JobProgressCard key={job.job_id} job={job} compact={true} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
