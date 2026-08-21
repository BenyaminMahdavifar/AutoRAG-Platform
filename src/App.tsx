import React, { useEffect } from "react";
import { Header } from "./components/Header";
import { JobBanner } from "./components/JobBanner";
import { DashboardOverview } from "./components/DashboardOverview";
import { KnowledgeBaseView } from "./components/KnowledgeBaseView";
import { DatasetView } from "./components/DatasetView";
import { PlaygroundView } from "./components/PlaygroundView";
import { OptimizerView } from "./components/OptimizerView";
import { ReportsView } from "./components/ReportsView";
import { SettingsView } from "./components/SettingsView";
import { DeveloperDebugView } from "./components/DeveloperDebugView";
import { useAppState } from "./state/appState";

export default function App() {
  const store = useAppState();
  const activeTab = store.workspaceState.activeTab;

  useEffect(() => {
    store.syncAll();
  }, []);

  return (
    <div
      id="main-app-container"
      className="min-h-screen bg-[#0f172a] text-slate-200 font-sans flex flex-col justify-between selection:bg-blue-600 selection:text-white"
    >
      <div>
        <Header />
        <JobBanner />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          {activeTab === "dashboard" && <DashboardOverview />}
          {activeTab === "kb" && <KnowledgeBaseView />}
          {activeTab === "dataset" && <DatasetView />}
          {activeTab === "playground" && <PlaygroundView />}
          {activeTab === "optimizer" && <OptimizerView />}
          {activeTab === "reports" && <ReportsView />}
          {activeTab === "settings" && <SettingsView />}
          {activeTab === "debug" && <DeveloperDebugView />}
        </main>
      </div>

      <footer className="border-t border-slate-800 bg-[#0b1120] px-4 py-2.5 text-[10px] font-mono text-slate-500 mt-8 flex flex-wrap justify-between items-center gap-2">
        <div className="flex items-center space-x-4">
          <span>OS: Linux 6.2.0-generic</span>
          <span>PYTHON: 3.11.4</span>
          <span>ENGINE: AutoRAG_v1.2 (Stateful Desktop Architecture)</span>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-emerald-500 flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>ALL_SYSTEMS_OPERATIONAL</span>
          </span>
          <span className="text-slate-400">
            {new Date().toISOString().slice(0, 10)} UTC
          </span>
        </div>
      </footer>
    </div>
  );
}
