import React from "react";
import { useAppState } from "../state/appState";
import { Activity, Clock, Database, Layers, CheckCircle, XCircle } from "lucide-react";

export const DeveloperDebugView: React.FC = () => {
  const store = useAppState();

  // Extract orchestrator metrics from completed jobs
  const metricsData: any[] = [];
  
  store.jobs.forEach(job => {
    if (job.status === "Completed" && job.result) {
      if (job.type === "build_dataset" && job.result.dataset?.execution_metadata?.orchestrator_metrics) {
        metricsData.push({
          jobId: job.job_id,
          action: "Dataset Generation",
          timestamp: job.updated_at,
          metrics: job.result.dataset.execution_metadata.orchestrator_metrics
        });
      }
      
      if (job.type === "run_optimizer" && job.result.summary) {
        // Try to extract from best experiment if available
        const bestExp = job.result.summary.leaderboard?.[0];
        if (bestExp?.results?.orchestrator_metrics) {
           metricsData.push({
              jobId: job.job_id,
              action: "Optimizer Sweep",
              timestamp: job.updated_at,
              metrics: bestExp.results.orchestrator_metrics
           });
        }
      }
      
      if (job.type === "run_experiment" && job.result.result?.orchestrator_metrics) {
        metricsData.push({
          jobId: job.job_id,
          action: "Experiment Trial",
          timestamp: job.updated_at,
          metrics: job.result.result.orchestrator_metrics
        });
      }

      if (job.type === "run_generalization_test" && job.result?.generalization_test?.orchestrator_metrics) {
        metricsData.push({
          jobId: job.job_id,
          action: "Generalization Test",
          timestamp: job.updated_at,
          metrics: job.result.generalization_test.orchestrator_metrics
        });
      }
    }
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Activity className="w-6 h-6 text-indigo-400" />
          AI Orchestrator Debug Panel
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
         <div className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-lg flex flex-col items-center">
            <Layers className="w-8 h-8 text-blue-400 mb-2" />
            <span className="text-2xl font-bold text-white">
              {metricsData.reduce((acc, d) => acc + (d.metrics.optimized_requests || 0), 0)}
            </span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Batches Executed</span>
         </div>
         <div className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-lg flex flex-col items-center">
            <Database className="w-8 h-8 text-emerald-400 mb-2" />
            <span className="text-2xl font-bold text-white">
              {metricsData.reduce((acc, d) => acc + (d.metrics.total_tokens_used || 0), 0)}
            </span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Total Tokens</span>
         </div>
         <div className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-lg flex flex-col items-center">
            <CheckCircle className="w-8 h-8 text-amber-400 mb-2" />
            <span className="text-2xl font-bold text-white">
              {metricsData.reduce((acc, d) => acc + (d.metrics.estimated_tokens_saved || 0), 0)}
            </span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Est. Tokens Saved</span>
         </div>
         <div className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-lg flex flex-col items-center">
            <Clock className="w-8 h-8 text-purple-400 mb-2" />
            <span className="text-2xl font-bold text-white">
              {metricsData.reduce((acc, d) => acc + (d.metrics.total_execution_time || 0), 0).toFixed(1)}s
            </span>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Total Exec Time</span>
         </div>
      </div>

      <div className="bg-[#111827] border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800 bg-[#0f172a] flex justify-between items-center">
          <h3 className="font-semibold text-slate-200">Execution History</h3>
        </div>
        
        {metricsData.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            No orchestrator metrics available yet. Run a dataset generation or optimizer sweep.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 bg-slate-900/50 uppercase">
                <tr>
                  <th className="px-4 py-3">Action</th>
                  <th className="px-4 py-3">Original Reqs</th>
                  <th className="px-4 py-3">Optimized (Batches)</th>
                  <th className="px-4 py-3">Tokens Used</th>
                  <th className="px-4 py-3">Tokens Saved</th>
                  <th className="px-4 py-3">Cache Hits</th>
                  <th className="px-4 py-3">Exec Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {metricsData.map((data, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/20 transition-colors">
                    <td className="px-4 py-3 font-medium text-slate-200">
                      {data.action}
                      <div className="text-[10px] text-slate-500">{new Date(data.timestamp).toLocaleString()}</div>
                    </td>
                    <td className="px-4 py-3 text-slate-300">{data.metrics.original_requests}</td>
                    <td className="px-4 py-3 text-blue-400 font-semibold">{data.metrics.optimized_requests}</td>
                    <td className="px-4 py-3 text-emerald-400">{data.metrics.total_tokens_used}</td>
                    <td className="px-4 py-3 text-amber-400">{data.metrics.estimated_tokens_saved}</td>
                    <td className="px-4 py-3 text-slate-300">{data.metrics.cache_hits}</td>
                    <td className="px-4 py-3 text-slate-300">{data.metrics.total_execution_time.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
