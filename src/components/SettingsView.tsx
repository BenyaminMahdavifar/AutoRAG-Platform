import React, { useState, useRef } from "react";
import { Settings, RefreshCw, Plus, Trash2, Download, Upload, CheckCircle2, Server, Database, Save, CheckCircle, AlertTriangle, Cpu } from "lucide-react";
import { useAppState } from "../state/appState";
import { ConnectionProfile } from "../types";
import { JobProgressCard } from "./JobProgressCard";

export const SettingsView: React.FC = () => {
  const store = useAppState();
  const activeProfileId = store.workspaceState.activeConnectionProfileId;
  const profiles = store.connectionProfiles;

  const [selectedProfileId, setSelectedProfileId] = useState<string>(activeProfileId);
  const selectedProfile = profiles.find((p) => p.profile_id === selectedProfileId) || profiles[0];

  const [name, setName] = useState(selectedProfile?.name || "");
  const [provider, setProvider] = useState(selectedProfile?.provider || "openai");
  const [baseUrl, setBaseUrl] = useState(selectedProfile?.base_url || "");
  const [defaultModel, setDefaultModel] = useState(selectedProfile?.default_model || "");
  const [timeoutSec, setTimeoutSec] = useState<number>(selectedProfile?.timeout_sec || 30);
  const [temperature, setTemperature] = useState<number>(selectedProfile?.temperature ?? 0.2);
  const [topP, setTopP] = useState<number>(selectedProfile?.top_p ?? 0.95);
  const [maxTokens, setMaxTokens] = useState<number>(selectedProfile?.max_tokens ?? 1024);
  const [apiKeySecret, setApiKeySecret] = useState(
    store.apiKeySecrets[selectedProfile?.api_key_reference || ""] || ""
  );

  // New embedding fields
  const [hfToken, setHfToken] = useState(
    store.apiKeySecrets[selectedProfile?.hf_token_reference || ""] || ""
  );
  const [embeddingModel, setEmbeddingModel] = useState(selectedProfile?.embedding_model || "sentence-transformers/all-MiniLM-L6-v2");
  const [embeddingDevice, setEmbeddingDevice] = useState<"auto" | "cpu" | "cuda" | "mps">(selectedProfile?.embedding_device || "auto");
  const [autoLogin, setAutoLogin] = useState(selectedProfile?.auto_login ?? true);

  
  
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const setupJob = store.jobs.find(j => j.type === "setup_environment");
  const isValidating = setupJob && (setupJob.status === "Running" || setupJob.status === "Starting" || setupJob.status === "Queued");
  const validationResult = (setupJob && setupJob.status === "Completed" && setupJob.result && setupJob.result.validation) ? setupJob.result.validation : (setupJob && setupJob.status === "Failed" ? { error: setupJob.error } : null);


  const profilesImportInputRef = useRef<HTMLInputElement>(null);
  const workspaceImportInputRef = useRef<HTMLInputElement>(null);

  const handleSelectProfile = (pId: string) => {
    setSelectedProfileId(pId);
    const target = profiles.find((p) => p.profile_id === pId);
    if (target) {
      setName(target.name);
      setProvider(target.provider);
      setBaseUrl(target.base_url);
      setDefaultModel(target.default_model);
      setTimeoutSec(target.timeout_sec || 30);
      setTemperature(target.temperature ?? 0.2);
      setTopP(target.top_p ?? 0.95);
      setMaxTokens(target.max_tokens ?? 1024);
      setApiKeySecret(store.apiKeySecrets[target.api_key_reference] || "");
      setHfToken(store.apiKeySecrets[target.hf_token_reference || ""] || "");
      setEmbeddingModel(target.embedding_model || "sentence-transformers/all-MiniLM-L6-v2");
      setEmbeddingDevice(target.embedding_device || "auto");
      setAutoLogin(target.auto_login ?? true);
      
    }
  };

  const handleSaveProfile = () => {
    if (!selectedProfile) return;
    const updated: ConnectionProfile = {
      ...selectedProfile,
      name,
      provider: provider as any,
      base_url: baseUrl,
      default_model: defaultModel,
      timeout_sec: timeoutSec,
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      embedding_model: embeddingModel,
      embedding_device: embeddingDevice,
      auto_login: autoLogin,
    };
    if (!updated.hf_token_reference) {
      updated.hf_token_reference = `hf_ref_${updated.profile_id}`;
    }
    store.updateConnectionProfile(updated, apiKeySecret, hfToken);
    setStatusMsg(`Connection profile "${name}" updated successfully!`);
    
    if (autoLogin) {
      handleValidateEnvironment();
    }
  };

  const handleCreateProfile = () => {
    const newId = `prof_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const keyRef = `key_ref_${newId}`;
    const hfRef = `hf_ref_${newId}`;
    const newProf: ConnectionProfile = {
      profile_id: newId,
      name: "New Connection Profile",
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      default_model: "gpt-4o-mini",
      api_key_reference: keyRef,
      hf_token_reference: hfRef,
      timeout_sec: 30,
      temperature: 0.2,
      top_p: 0.95,
      max_tokens: 1024,
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
      embedding_device: "auto",
      auto_login: true,
    };
    store.addConnectionProfile(newProf, "", "");
    handleSelectProfile(newId);
    setStatusMsg("New profile created and set as active!");
  };

  const handleDeleteProfile = (pId: string) => {
    if (profiles.length <= 1) {
      setStatusMsg("Cannot delete the only remaining profile!");
      return;
    }
    store.deleteConnectionProfile(pId);
    const nextProf = store.connectionProfiles[0];
    if (nextProf) {
      handleSelectProfile(nextProf.profile_id);
    }
    setStatusMsg("Profile deleted.");
  };

  const handleSetAsActive = () => {
    store.setActiveConnectionProfile(selectedProfileId);
    setStatusMsg(`Profile "${selectedProfile.name}" activated for workspace operations!`);
  };

  const handleValidateEnvironment = () => {
    store.startJob("setup_environment", "Environment Setup", "setup_environment", {
      provider,
      base_url: baseUrl,
      api_key: apiKeySecret,
      model_name: defaultModel,
      timeout_sec: timeoutSec,
      hf_token: hfToken,
      embedding_model: embeddingModel,
    });
  };

  // Import / Export Profile Handlers
  const handleExportProfiles = () => {
    const jsonStr = store.exportConnectionProfiles();
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autorag_connection_profiles_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportProfilesFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (text && store.importConnectionProfiles(text)) {
        setStatusMsg("Connection profiles imported successfully!");
      } else {
        setStatusMsg("Failed to import connection profiles. Check JSON format.");
      }
    };
    reader.readAsText(file);
  };

  // Import / Export Workspace Settings Handlers
  const handleExportWorkspace = () => {
    const jsonStr = store.exportWorkspaceSettings();
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autorag_workspace_settings_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportWorkspaceFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (text && store.importWorkspaceSettings(text)) {
        setStatusMsg("Workspace settings imported successfully!");
      } else {
        setStatusMsg("Failed to import workspace settings. Check JSON format.");
      }
    };
    reader.readAsText(file);
  };

  return (
    <div id="settings-view-container" className="space-y-4 max-w-6xl mx-auto pb-10">
      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <Settings className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Connection Profiles & Workspace Persistence
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Manage connection profiles, embedding models, and workspace state for a self-configuring environment.
            </p>
          </div>
        </div>
      </div>

      {statusMsg && (
        <div className="p-2.5 bg-blue-950/60 border border-blue-800 rounded text-xs font-mono text-blue-300 flex items-center justify-between">
          <span>{statusMsg}</span>
          <button onClick={() => setStatusMsg(null)} className="text-blue-400 font-bold">×</button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Column 1: Connection Profiles List */}
        <div className="lg:col-span-4 space-y-3">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono">
                Connection Profiles ({profiles.length})
              </span>
              <button
                onClick={handleCreateProfile}
                className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1"
              >
                <Plus className="w-3 h-3" />
                <span>New</span>
              </button>
            </div>

            <div className="space-y-1.5 max-h-[320px] overflow-y-auto pr-1">
              {profiles.map((prof) => {
                const isSelected = prof.profile_id === selectedProfileId;
                const isActive = prof.profile_id === activeProfileId;

                return (
                  <div
                    key={prof.profile_id}
                    onClick={() => handleSelectProfile(prof.profile_id)}
                    className={`p-2.5 rounded border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? "bg-blue-600/10 border-blue-500 text-white font-semibold"
                        : "bg-slate-950/60 border-slate-800/80 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold truncate max-w-[180px]">{prof.name}</span>
                      {isActive && (
                        <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded text-[9px] font-mono font-bold uppercase">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mt-1">
                      <span className="uppercase text-blue-400">{prof.provider}</span>
                      <span>{prof.default_model}</span>
                    </div>
                  </div>
                );
              })}
            </div>
            
            <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-[10px] font-mono">
              <button
                onClick={handleExportProfiles}
                className="text-slate-400 hover:text-white flex items-center space-x-1"
              >
                <Download className="w-3 h-3 text-blue-400" />
                <span>Export Profiles</span>
              </button>
              <button
                onClick={() => profilesImportInputRef.current?.click()}
                className="text-slate-400 hover:text-white flex items-center space-x-1"
              >
                <Upload className="w-3 h-3 text-emerald-400" />
                <span>Import Profiles</span>
              </button>
              <input
                type="file"
                ref={profilesImportInputRef}
                onChange={handleImportProfilesFile}
                accept=".json"
                className="hidden"
              />
            </div>
          </div>
        </div>

        {/* Column 2: Profile Details Editor */}
        <div className="lg:col-span-8 space-y-4">
          {selectedProfile ? (
            <>
              {/* Common Header for selected profile */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex items-center justify-between shadow-sm">
                 <div className="flex items-center space-x-3">
                   <Server className="w-5 h-5 text-blue-400" />
                   <div>
                     <input
                       type="text"
                       value={name}
                       onChange={(e) => setName(e.target.value)}
                       className="bg-transparent border-b border-transparent hover:border-slate-700 focus:border-blue-500 text-sm font-bold text-white uppercase font-mono px-1 py-0.5 focus:outline-none w-64 transition-colors"
                     />
                     <div className="text-[10px] text-slate-500 font-mono px-1">ID: {selectedProfile.profile_id}</div>
                   </div>
                 </div>
                 <div className="flex items-center space-x-2 font-mono">
                   {selectedProfileId !== activeProfileId && (
                     <button
                       onClick={handleSetAsActive}
                       className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1 shadow-sm"
                     >
                       <CheckCircle2 className="w-3 h-3" />
                       <span>Activate Profile</span>
                     </button>
                   )}
                   <button
                     onClick={handleSaveProfile}
                     className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1 shadow-sm"
                   >
                     <Save className="w-3 h-3" />
                     <span>Save Profile</span>
                   </button>
                   <button
                     onClick={() => handleDeleteProfile(selectedProfileId)}
                     className="p-1.5 bg-slate-800 hover:bg-rose-900/60 text-slate-400 hover:text-rose-300 border border-slate-700 rounded transition-all"
                     title="Delete Profile"
                   >
                     <Trash2 className="w-3.5 h-3.5" />
                   </button>
                 </div>
              </div>

              {/* Section 1: LLM Connection */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono mb-3 flex items-center">
                  <span className="bg-slate-800 text-white w-4 h-4 flex items-center justify-center rounded-full mr-2 text-[9px]">1</span>
                  LLM Connection
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Provider
                    </label>
                    <select
                      value={provider}
                      onChange={(e) => setProvider(e.target.value as any)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
                    >
                      <option value="openai">OpenAI Official</option>
                      <option value="gemini">Google Gemini API</option>
                      <option value="ollama">Ollama Local Server</option>
                      <option value="lmstudio">LM Studio Local Server</option>
                      <option value="openrouter">OpenRouter API</option>
                      <option value="custom">Custom OpenAI-Compatible</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Base URL
                    </label>
                    <input
                      type="text"
                      value={baseUrl}
                      onChange={(e) => setBaseUrl(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Default Chat Model
                    </label>
                    <input
                      type="text"
                      value={defaultModel}
                      onChange={(e) => setDefaultModel(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      API Key
                    </label>
                    <input
                      type="password"
                      value={apiKeySecret}
                      onChange={(e) => setApiKeySecret(e.target.value)}
                      placeholder="sk-..."
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div className="sm:col-span-2 grid grid-cols-4 gap-4">
                     <div>
                      <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                        Temperature ({temperature})
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.01"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full accent-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                        Top P ({topP})
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={topP}
                        onChange={(e) => setTopP(parseFloat(e.target.value))}
                        className="w-full accent-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                        Max Tokens
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(parseInt(e.target.value) || 1024)}
                        className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                        Timeout (s)
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={timeoutSec}
                        onChange={(e) => setTimeoutSec(parseInt(e.target.value) || 30)}
                        className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 2: Embedding Configuration */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono mb-3 flex items-center">
                  <span className="bg-slate-800 text-white w-4 h-4 flex items-center justify-center rounded-full mr-2 text-[9px]">2</span>
                  Embedding Configuration
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="sm:col-span-2">
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Hugging Face Access Token
                    </label>
                    <input
                      type="password"
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      placeholder="hf_..."
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                    <p className="text-[9px] text-slate-500 mt-1 font-mono">Required for gated models. Saved securely per profile.</p>
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Embedding Model
                    </label>
                    <input
                      type="text"
                      value={embeddingModel}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      placeholder="sentence-transformers/all-MiniLM-L6-v2"
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                      Device Compute
                    </label>
                    <select
                      value={embeddingDevice}
                      onChange={(e) => setEmbeddingDevice(e.target.value as any)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    >
                      <option value="auto">Auto (Best Available)</option>
                      <option value="cuda">CUDA (GPU)</option>
                      <option value="mps">MPS (Apple Silicon)</option>
                      <option value="cpu">CPU Only</option>
                    </select>
                  </div>
                  <div className="sm:col-span-2 flex items-center space-x-2 mt-1">
                    <input
                      type="checkbox"
                      id="autoLogin"
                      checked={autoLogin}
                      onChange={(e) => setAutoLogin(e.target.checked)}
                      className="accent-blue-500 w-3.5 h-3.5 rounded border-slate-700 bg-slate-950"
                    />
                    <label htmlFor="autoLogin" className="text-xs text-slate-300 font-mono select-none cursor-pointer">
                      Automatically Authenticate & Load Model on Startup
                    </label>
                  </div>
                </div>
              </div>

              {/* Section 3: Workspace Persistence */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono mb-3 flex items-center">
                  <span className="bg-slate-800 text-white w-4 h-4 flex items-center justify-center rounded-full mr-2 text-[9px]">3</span>
                  Workspace Persistence
                </h3>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-slate-950 p-3 rounded border border-slate-800">
                   <div>
                     <h4 className="text-xs font-bold text-white mb-2 font-mono flex items-center"><Database className="w-3 h-3 mr-1.5 text-blue-400"/> Auto-Restore States</h4>
                     <ul className="text-[10px] text-slate-400 font-mono space-y-1.5 ml-1">
                       <li className="flex items-center"><CheckCircle2 className="w-3 h-3 text-emerald-500 mr-2"/> Last Active Connection Profile</li>
                       <li className="flex items-center"><CheckCircle2 className="w-3 h-3 text-emerald-500 mr-2"/> Cached Embedding Model</li>
                       <li className="flex items-center"><CheckCircle2 className="w-3 h-3 text-emerald-500 mr-2"/> Last Selected Dataset & Pipeline</li>
                       <li className="flex items-center"><CheckCircle2 className="w-3 h-3 text-emerald-500 mr-2"/> UI Tab Preferences</li>
                     </ul>
                   </div>
                   <div className="flex flex-col justify-between">
                     <div>
                       <p className="text-[10px] text-slate-400 font-mono leading-relaxed mb-3">
                         The application automatically preserves your configuration across sessions to minimize setup friction. You can export or import the entire workspace state manually.
                       </p>
                     </div>
                     <div className="flex items-center space-x-2 font-mono text-xs">
                        <button
                          onClick={handleExportWorkspace}
                          className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1 w-full justify-center"
                        >
                          <Download className="w-3 h-3 text-blue-400" />
                          <span>Backup State</span>
                        </button>
                        <button
                          onClick={() => workspaceImportInputRef.current?.click()}
                          className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1 w-full justify-center"
                        >
                          <Upload className="w-3 h-3 text-emerald-400" />
                          <span>Restore State</span>
                        </button>
                        <input
                          type="file"
                          ref={workspaceImportInputRef}
                          onChange={handleImportWorkspaceFile}
                          accept=".json"
                          className="hidden"
                        />
                     </div>
                   </div>
                </div>
              </div>

              {/* Section 4: Diagnostics & Validation */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono flex items-center">
                    <span className="bg-slate-800 text-white w-4 h-4 flex items-center justify-center rounded-full mr-2 text-[9px]">4</span>
                    Environment Lifecycle
                  </h3>
                  <button
                    onClick={handleValidateEnvironment}
                    disabled={isValidating}
                    className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[10px] font-bold uppercase transition-all flex items-center space-x-1 shadow-sm"
                  >
                    <RefreshCw className={`w-3 h-3 ${isValidating ? "animate-spin" : ""}`} />
                    <span>Prepare Environment</span>
                  </button>
                </div>
                
                <div className="bg-slate-950 border border-slate-800 rounded p-3 font-mono text-xs">
                  {!validationResult && !isValidating && (
                    <div className="text-slate-500 text-center py-4">
                      Click "Prepare Environment" to automatically resolve, install dependencies, and validate the runtime.
                    </div>
                  )}
                  {setupJob && (isValidating || setupJob.status === "Failed") && (
                    <div className="mb-4">
                      <JobProgressCard job={setupJob} />
                    </div>
                  )}
                  {validationResult && !isValidating && setupJob?.status === "Completed" && (
                    <div className="space-y-4">
                      {validationResult.error ? (
                        <div className="text-rose-400 flex items-center font-bold">
                          <AlertTriangle className="w-4 h-4 mr-2" /> Error: {validationResult.error}
                        </div>
                      ) : (
                        <>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pb-3 border-b border-slate-800">
                            <div className="space-y-1">
                              <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">Python:</span> {validationResult.python_runtime}</div>
                              <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">CUDA Available:</span> {validationResult.cuda_available ? <span className="text-emerald-400">Yes ({validationResult.gpu_name})</span> : <span className="text-amber-400">No (CPU Fallback)</span>}</div>
                              <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">GPU Memory:</span> {validationResult.memory_available}</div>
                            </div>
                            <div className="space-y-1">
                               <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">LLM Connection:</span> {validationResult.llm_connection ? <span className="text-emerald-400">✓ Verified</span> : <span className="text-rose-400">✗ Failed</span>}</div>
                               <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">HF Auth:</span> {validationResult.hf_authenticated ? <span className="text-emerald-400">✓ Valid</span> : <span className="text-amber-400">✗ Unauthenticated</span>}</div>
                               <div className="flex items-center text-slate-300"><span className="w-32 text-slate-500">Model Download:</span> {validationResult.model_available ? <span className="text-emerald-400">✓ Ready</span> : <span className="text-rose-400">✗ Not Found</span>}</div>
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-slate-500 uppercase font-bold mb-2 tracking-wider">Raw Debug Logs</div>
                            <div className="bg-black/50 p-2 rounded border border-slate-800 text-[10px] text-slate-300 space-y-1 max-h-48 overflow-y-auto font-mono">
                              {validationResult.details?.map((d: string, i: number) => (
                                <div key={i} className={`${d.toLowerCase().includes('failed') || d.toLowerCase().includes('error') ? 'text-rose-400' : 'text-slate-300'}`}>
                                  <span className="text-slate-600 mr-2">[{new Date().toISOString().slice(11, 19)}]</span>
                                  {d}
                                </div>
                              ))}
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>

            </>
          ) : (
            <div className="text-center py-12 bg-slate-900 border border-slate-800 rounded p-4 text-slate-500 text-xs font-mono">
              Select a connection profile to view details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
