import React from "react";
import { Database, Upload, RefreshCw, Hash, Code } from "lucide-react";
import { useAppState } from "../state/appState";
import { JobProgressCard } from "./JobProgressCard";
import { DocumentMeta } from "../types";

export const KnowledgeBaseView: React.FC = () => {
  const store = useAppState();
  const session = store.sessions.kb;
  const documents = store.documents;
  const [importMode, setImportMode] = React.useState<"single" | "folder" | "zip">("single");
  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const zipInputRef = React.useRef<HTMLInputElement>(null);

  const activeJob = store.jobs.find(
    (j) => (j.type === "scan_kb" || j.type === "upload_doc" || j.type === "import_kb" || j.type === "clear_kb") && (j.status === "Running" || j.status === "Queued")
  );

  const selectedDoc = documents.find((d) => d.doc_id === session.selectedDocId) || documents[0] || null;

  const handleRescan = () => {
    store.startJob("scan_kb", "Knowledge Base Directory Rescan", "scan_kb", {});
  };

  const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    const fileArray = Array.from(files) as File[];
    const payloadFiles = [];
    
    for (const file of fileArray) {
      const reader = new FileReader();
      const b64 = await new Promise<string>((resolve) => {
        reader.onload = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(file);
      });
      
      payloadFiles.push({
        relative_path: file.webkitRelativePath || file.name,
        content_b64: b64,
        size: file.size
      });
    }
    
    store.startJob("import_kb", `Import Folder (${files.length} files)`, "import_kb", {
      import_type: "folder",
      batch_id: Date.now().toString(),
      files: payloadFiles
    });
    
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const handleZipSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    const b64 = await new Promise<string>((resolve) => {
      reader.onload = () => resolve((reader.result as string).split(',')[1]);
      reader.readAsDataURL(file);
    });
    
    store.startJob("import_kb", `Import ZIP (${file.name})`, "import_kb", {
      import_type: "zip",
      batch_id: Date.now().toString(),
      zip_b64: b64
    });
    
    if (zipInputRef.current) zipInputRef.current.value = "";
  };

  const handleUpload = (e: React.FormEvent) => {
    e.preventDefault();
    if (!session.uploadName || !session.uploadContent) return;

    store.startJob("upload_doc", `Upload Document (${session.uploadName})`, "upload_doc", {
      filename: session.uploadName,
      content: session.uploadContent,
    });

    store.updateSession("kb", { uploadName: "", uploadContent: "" });
  };

  const previewChunks = (text: string) => {
    if (!text) return [];
    if (session.chunkStrategy === "fixed") {
      const chunks = [];
      for (let i = 0; i < text.length; i += session.chunkSize) {
        chunks.push(text.slice(i, i + session.chunkSize));
      }
      return chunks;
    } else if (session.chunkStrategy === "paragraph") {
      return text.split("\n\n").filter((p) => p.trim());
    } else {
      return text.split(/(?<=[.!?])\s+/).filter((s) => s.trim());
    }
  };

  const activeDocContent = selectedDoc?.content_preview || "";
  const liveChunks = previewChunks(activeDocContent);

  return (
    <div id="kb-view-container" className="space-y-4">
      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-blue-500" />
          <div>
            <h1 className="text-xs font-bold text-white uppercase tracking-wider font-sans">
              Knowledge Base Subsystem
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">
              Directory document scanning, SHA256 verification & live chunk preview
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => store.startJob("clear_kb", "Clear Knowledge Base", "clear_kb", {})}
            disabled={!!activeJob || documents.length === 0}
            className="px-3.5 py-1.5 bg-red-900/40 hover:bg-red-900/60 disabled:opacity-50 text-red-200 border border-red-800/50 rounded text-xs font-mono transition-all flex items-center space-x-1.5 shadow-sm"
          >
            <span>CLEAR_KB</span>
          </button>
          <button
            id="scan-kb-btn"
            onClick={handleRescan}
            disabled={!!activeJob}
            className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 border border-slate-700 rounded text-xs font-mono transition-all flex items-center space-x-1.5 shadow-sm"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${activeJob ? "animate-spin" : ""}`} />
            <span>{activeJob ? "Scanning Job Running..." : "RESCAN_DIRECTORY"}</span>
          </button>
        </div>
      </div>

      {/* Active Job Card */}
      {activeJob && (
        <div className="mb-4">
          <JobProgressCard job={activeJob} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Column 1: Document List & Upload */}
        <div className="lg:col-span-4 space-y-3">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Scanned Files ({documents.length})
              </h2>
              <span className="text-[10px] font-mono text-blue-400">FS_SYNC</span>
            </div>
            <div className="space-y-1.5 max-h-[260px] overflow-y-auto pr-1">
              {documents.map((doc) => {
                const isSelected = selectedDoc?.doc_id === doc.doc_id;
                return (
                  <div
                    key={doc.doc_id}
                    id={`doc-item-${doc.doc_id}`}
                    onClick={() => store.updateSession("kb", { selectedDocId: doc.doc_id })}
                    className={`p-2.5 rounded border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? "bg-blue-600/10 border-blue-500 text-white font-semibold"
                        : "bg-slate-950/60 border-slate-800/80 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono truncate max-w-[170px]">{doc.filename}</span>
                      <span className="text-[10px] font-mono text-slate-500">{doc.size_bytes} B</span>
                    </div>
                    <div className="flex items-center space-x-1.5 text-[10px] text-slate-400 mt-1 font-mono">
                      <Hash className="w-3 h-3 text-slate-500" />
                      <span>{doc.checksum.slice(0, 12)}...</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Upload Form */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 shadow-sm">
            <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <div className="flex items-center space-x-1.5">
                <Upload className="w-3.5 h-3.5 text-blue-400" />
                <span>Import Source</span>
              </div>
            </h2>
            <div className="flex space-x-1 border border-slate-800 rounded bg-slate-950 p-1 mb-3">
              <button 
                onClick={() => setImportMode("single")}
                className={`flex-1 text-[10px] uppercase font-bold py-1 rounded transition-all ${importMode === "single" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >Single</button>
              <button 
                onClick={() => setImportMode("folder")}
                className={`flex-1 text-[10px] uppercase font-bold py-1 rounded transition-all ${importMode === "folder" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >Folder</button>
              <button 
                onClick={() => setImportMode("zip")}
                className={`flex-1 text-[10px] uppercase font-bold py-1 rounded transition-all ${importMode === "zip" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >ZIP</button>
            </div>

            {importMode === "single" && (
              <form onSubmit={handleUpload} className="space-y-2">
                <div>
                  <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                    Filename
                  </label>
                  <input
                    id="upload-filename-input"
                    type="text"
                    placeholder="guidelines.md"
                    value={session.uploadName}
                    onChange={(e) => store.updateSession("kb", { uploadName: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] uppercase font-bold text-slate-400 mb-1 font-mono">
                    Content
                  </label>
                  <textarea
                    id="upload-content-textarea"
                    rows={3}
                    placeholder="Paste document text or markdown..."
                    value={session.uploadContent}
                    onChange={(e) => store.updateSession("kb", { uploadContent: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
                  ></textarea>
                </div>
                <button
                  id="submit-upload-btn"
                  type="submit"
                  disabled={!!activeJob || !session.uploadName || !session.uploadContent}
                  className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center justify-center space-x-1.5 shadow-sm"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload File</span>
                </button>
              </form>
            )}

            {importMode === "folder" && (
              <div className="space-y-2">
                <p className="text-[10px] text-slate-400 font-mono mb-2">
                  Select a folder from your disk. Nested files (txt, md, json, pdf) will be imported recursively.
                </p>
                <input
                  type="file"
                  ref={folderInputRef}
                  onChange={handleFolderSelect}
                  // @ts-ignore - webkitdirectory is non-standard but supported by most modern browsers
                  webkitdirectory=""
                  directory=""
                  multiple
                  className="hidden"
                />
                <button
                  onClick={() => folderInputRef.current?.click()}
                  disabled={!!activeJob}
                  className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center justify-center space-x-1.5 shadow-sm"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Select Folder</span>
                </button>
              </div>
            )}

            {importMode === "zip" && (
              <div className="space-y-2">
                <p className="text-[10px] text-slate-400 font-mono mb-2">
                  Upload a ZIP archive. Its contents will be extracted and imported.
                </p>
                <input
                  type="file"
                  accept=".zip"
                  ref={zipInputRef}
                  onChange={handleZipSelect}
                  className="hidden"
                />
                <button
                  onClick={() => zipInputRef.current?.click()}
                  disabled={!!activeJob}
                  className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs font-semibold uppercase tracking-wider transition-all flex items-center justify-center space-x-1.5 shadow-sm"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Select ZIP Archive</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Column 2 & 3: Selected Document & Live Chunk Inspector */}
        <div className="lg:col-span-8 space-y-3">
          {selectedDoc ? (
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-sm space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div>
                  <h2 className="text-sm font-bold text-white font-mono">{selectedDoc.filename}</h2>
                  <p className="text-[10px] text-slate-400 font-mono">{selectedDoc.filepath}</p>
                </div>
                <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-mono uppercase">
                  .{selectedDoc.file_type}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                  <span className="text-[9px] text-slate-500 uppercase font-bold block font-mono">
                    Checksum
                  </span>
                  <span className="text-[11px] font-mono text-slate-300">
                    {selectedDoc.checksum.slice(0, 14)}...
                  </span>
                </div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                  <span className="text-[9px] text-slate-500 uppercase font-bold block font-mono">
                    Characters
                  </span>
                  <span className="text-[11px] font-mono font-bold text-white">
                    {selectedDoc.metadata?.char_count || activeDocContent.length} chars
                  </span>
                </div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                  <span className="text-[9px] text-slate-500 uppercase font-bold block font-mono">
                    Est. Tokens
                  </span>
                  <span className="text-[11px] font-mono font-bold text-blue-400">
                    {Math.max(1, Math.floor(activeDocContent.length / 4))} tokens
                  </span>
                </div>
              </div>

              {/* Live Chunking Inspector */}
              <div className="bg-slate-950/80 rounded border border-slate-800 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                  <div className="flex items-center space-x-1.5">
                    <Code className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                      Chunk Inspector
                    </span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <select
                      id="chunk-strategy-select"
                      value={session.chunkStrategy}
                      onChange={(e) =>
                        store.updateSession("kb", { chunkStrategy: e.target.value as any })
                      }
                      className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                    >
                      <option value="recursive">Recursive Character</option>
                      <option value="fixed">Fixed Size</option>
                      <option value="paragraph">Paragraph Split</option>
                      <option value="semantic">Semantic Sentence</option>
                    </select>

                    {session.chunkStrategy === "fixed" && (
                      <input
                        id="chunk-size-slider"
                        type="number"
                        min={64}
                        max={1024}
                        step={64}
                        value={session.chunkSize}
                        onChange={(e) =>
                          store.updateSession("kb", { chunkSize: Number(e.target.value) })
                        }
                        className="w-16 bg-slate-900 border border-slate-700 rounded px-1.5 py-1 text-xs text-white font-mono"
                      />
                    )}
                  </div>
                </div>

                <p className="text-[10px] text-slate-400 font-mono mb-2">
                  Previewing <span className="font-bold text-blue-400">{liveChunks.length}</span>{" "}
                  chunks generated via{" "}
                  <span className="text-white uppercase font-bold">{session.chunkStrategy}</span>{" "}
                  strategy:
                </p>

                <div className="space-y-1.5 max-h-[300px] overflow-y-auto pr-1">
                  {liveChunks.slice(0, 8).map((chunkText, i) => (
                    <div
                      key={i}
                      className="p-2.5 bg-slate-900 rounded border border-slate-800 text-[11px] text-slate-300 font-mono"
                    >
                      <div className="flex items-center justify-between text-[10px] text-blue-400 font-semibold mb-1">
                        <span>CHUNK_{String(i + 1).padStart(3, "0")}</span>
                        <span className="text-slate-500">{chunkText.length} CHARS</span>
                      </div>
                      <p className="leading-relaxed line-clamp-3">{chunkText}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 bg-slate-900 border border-slate-800 rounded p-4 text-slate-500 text-xs font-mono">
              Select a file from the Knowledge Base list to inspect contents.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
