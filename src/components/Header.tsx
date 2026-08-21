import React from "react";
import { Cpu, Database, Zap, Layers, Sliders, FileText, Settings, Activity, Server, HardDrive } from "lucide-react";
import { useAppState } from "../state/appState";

export const Header: React.FC = () => {
  const store = useAppState();
  const activeTab = store.workspaceState.activeTab;
  const activeProfile =
    store.connectionProfiles.find(
      (p) => p.profile_id === store.workspaceState.activeConnectionProfileId
    ) || store.connectionProfiles[0];

  const meta = store.cacheMetadata;

  const tabs = [
    { id: "dashboard", label: "Workspace", icon: Zap },
    { id: "kb", label: "Knowledge Base", icon: Database },
    { id: "dataset", label: "Dataset Builder", icon: Layers },
    { id: "playground", label: "Playground", icon: Cpu },
    { id: "optimizer", label: "Optimizer Sweep", icon: Sliders },
    { id: "reports", label: "Report Engine", icon: FileText },
    { id: "settings", label: "Connections", icon: Settings },
    { id: "debug", label: "Dev Debug", icon: Activity },
  ];

  return (
    <header id="header-container" className="bg-[#0b1120] border-b border-slate-800 sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14 border-b border-slate-800/60">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center font-bold text-xs text-white shadow-sm">
              R
            </div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm tracking-tight text-white uppercase font-sans">
                AutoRAG <span className="text-blue-500 font-mono text-[10px] lowercase">v2.0</span>
              </span>
              <span className="h-3 w-px bg-slate-800"></span>
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest hidden sm:inline-block">
                Desktop-Class State Engine
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            {/* Cache Metadata Quick Badges */}
            <div className="hidden xl:flex items-center space-x-2 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-[10px] font-mono text-slate-400">
              <HardDrive className="w-3 h-3 text-blue-400" />
              <span>DOCS: <strong className="text-slate-200">{meta.documentsCount}</strong></span>
              <span className="text-slate-700">|</span>
              <span>TRIALS: <strong className="text-slate-200">{meta.experimentsCount}</strong></span>
              <span className="text-slate-700">|</span>
              <span>TESTS: <strong className="text-slate-200">{meta.datasetItemsCount}</strong></span>
            </div>

            {/* Active Connection Profile Badge */}
            <button
              onClick={() => store.setActiveTab("settings")}
              className="flex items-center space-x-2 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] font-mono text-slate-300 transition-colors"
            >
              <Server className="w-3 h-3 text-emerald-400" />
              <span className="text-slate-400 uppercase hidden md:inline">PROFILE:</span>
              <span className="text-blue-400 font-semibold">{activeProfile.name}</span>
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 overflow-x-auto py-1.5 scrollbar-none">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                id={`tab-btn-${tab.id}`}
                onClick={() => store.setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? "bg-blue-600 text-white font-semibold shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/80"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span className="uppercase text-[11px] tracking-wider">{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
