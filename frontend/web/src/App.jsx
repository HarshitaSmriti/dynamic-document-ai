import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, Upload, CheckCircle, AlertCircle, RefreshCw, 
  Settings, Download, Copy, Table, List, Eye, Sparkles, Layers, ArrowRight, ExternalLink
} from 'lucide-react';

export default function App() {
  const [backendUrl, setBackendUrl] = useState(
    import.meta.env.VITE_API_BASE_URL || 
    (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8000' : '')
  );
  
  const [showSettings, setShowSettings] = useState(false);
  const [healthStatus, setHealthStatus] = useState({ online: false, model: 'Connecting...', backend: 'hosted_api' });
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isPdf, setIsPdf] = useState(false);
  const [extractionMode, setExtractionMode] = useState('discovery'); // 'discovery' | 'schema'
  const [customInstructions, setCustomInstructions] = useState('');
  const [customSchemaJson, setCustomSchemaJson] = useState('{\n  "schema_name": "Custom Document",\n  "fields": [\n    {"name": "document_number", "type": "string"},\n    {"name": "date", "type": "date"}\n  ]\n}');
  
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('fields'); // 'fields' | 'tables' | 'lists' | 'json' | 'warnings'
  const [copied, setCopied] = useState(false);
  
  const fileInputRef = useRef(null);

  // Check health on mount or when backendUrl changes
  useEffect(() => {
    checkHealth();
  }, [backendUrl]);

  const checkHealth = async () => {
    try {
      const base = backendUrl.replace(/\/$/, '');
      const res = await fetch(`${base}/api/v1/health`, { method: 'GET' }).catch(() => fetch(`${base}/health`, { method: 'GET' }));
      if (res.ok) {
        let model = 'qwen/qwen2.5-vl-72b-instruct';
        let backend = 'hosted_api';
        try {
          const data = await res.json();
          if (data && typeof data === 'object') {
            model = data.provider?.model_name || data.model || model;
            backend = data.backend || backend;
          }
        } catch (_) {}
        setHealthStatus({
          online: true,
          model: model,
          backend: backend
        });
      } else {
        setHealthStatus({ online: false, model: 'Offline', backend: 'unknown' });
      }
    } catch (e) {
      setHealthStatus({ online: false, model: 'Offline (CORS / Unreachable)', backend: 'unknown' });
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setError(null);
      setResult(null);
      const isPdfFile = selected.type === 'application/pdf' || selected.name.endsWith('.pdf');
      setIsPdf(isPdfFile);
      if (!isPdfFile) {
        setFilePreview(URL.createObjectURL(selected));
      } else {
        setFilePreview(null);
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
      setError(null);
      setResult(null);
      const isPdfFile = dropped.type === 'application/pdf' || dropped.name.endsWith('.pdf');
      setIsPdf(isPdfFile);
      if (!isPdfFile) {
        setFilePreview(URL.createObjectURL(dropped));
      } else {
        setFilePreview(null);
      }
    }
  };

  const executeExtraction = async () => {
    if (!file) {
      setError('Please upload a document file (PDF or Image) first.');
      return;
    }

    setLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const base = backendUrl.replace(/\/$/, '');
      const formData = new FormData();
      formData.append('file', file);
      
      if (customInstructions.trim()) {
        formData.append('custom_instructions', customInstructions);
      }
      if (extractionMode === 'schema' && customSchemaJson.trim()) {
        formData.append('schema_json', customSchemaJson);
      }

      const response = await fetch(`${base}/api/v1/extract/upload`, {
        method: 'POST',
        body: formData,
      });

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      setLatency(elapsed);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || `Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.success) {
        setResult(data.data);
        if (data.data?.tables?.length > 0) setActiveTab('fields');
      } else {
        setResult(data.data);
        setError(data.error_message || 'Extraction failed to parse data.');
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to extraction backend API.');
    } finally {
      setLoading(false);
    }
  };

  const copyJson = () => {
    if (result) {
      navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const downloadJson = () => {
    if (result) {
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${file?.name?.replace(/\.[^/.]+$/, '') || 'document'}_extracted.json`;
      a.click();
    }
  };

  const downloadTableCsv = (table) => {
    if (!table || !table.headers || !table.rows) return;
    const csvContent = [
      table.headers.map(h => `"${String(h).replace(/"/g, '""')}"`).join(','),
      ...table.rows.map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${table.table_name || 'table'}.csv`;
    a.click();
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#090d16' }}>
      {/* Top Navbar */}
      <header style={{ borderBottom: '1px solid #1f293d', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#0d1322' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'linear-gradient(135deg, #2563eb, #7c3aed)', padding: '8px', borderRadius: '8px', display: 'flex' }}>
            <Sparkles size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '-0.02em', color: '#f3f4f6' }}>Enterprise Document AI</h1>
            <p style={{ fontSize: '12px', color: '#9ca3af' }}>Dynamic Schema-Agnostic Extraction Engine</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Health Status Indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#111827', padding: '6px 12px', borderRadius: '20px', border: '1px solid #1f293d', fontSize: '12px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: healthStatus.online ? '#10b981' : '#ef4444' }}></span>
            <span style={{ color: '#d1d5db', fontWeight: '500' }}>{healthStatus.model}</span>
          </div>

          <button 
            onClick={() => setShowSettings(!showSettings)}
            style={{ background: '#1f293d', border: '1px solid #374151', padding: '8px', borderRadius: '8px', cursor: 'pointer', color: '#9ca3af' }}
            title="Backend Settings"
          >
            <Settings size={18} />
          </button>
        </div>
      </header>

      {/* Settings Drawer / Modal */}
      {showSettings && (
        <div style={{ backgroundColor: '#111827', borderBottom: '1px solid #374151', padding: '16px 24px', display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '12px', color: '#9ca3af', display: 'block', marginBottom: '4px' }}>Backend API Base URL (Render / Local):</label>
            <input 
              type="text" 
              value={backendUrl} 
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="e.g. https://your-backend.onrender.com or http://localhost:8000"
              style={{ width: '100%', padding: '8px 12px', background: '#090d16', border: '1px solid #374151', borderRadius: '6px', color: '#f3f4f6', fontSize: '13px' }}
            />
          </div>
          <button 
            onClick={checkHealth}
            style={{ padding: '8px 16px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', alignSelf: 'flex-end' }}
          >
            <RefreshCw size={14} /> Test Connection
          </button>
        </div>
      )}

      {/* Main Content Layout */}
      <main style={{ flex: 1, padding: '24px', display: 'grid', gridTemplateColumns: '420px 1fr', gap: '24px', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        
        {/* Left Column: Upload & Options */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Upload Area */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <h2 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '14px', color: '#f3f4f6', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Upload size={18} color="#3b82f6" /> 1. Upload Document
            </h2>

            <div 
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{ 
                border: '2px dashed #374151', borderRadius: '8px', padding: '28px 16px', textAlign: 'center', cursor: 'pointer',
                backgroundColor: file ? '#131b2e' : '#0d1322', transition: 'all 0.2s ease', borderColor: file ? '#3b82f6' : '#374151'
              }}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="application/pdf,image/png,image/jpeg,image/jpg,image/webp" 
                style={{ display: 'none' }} 
              />
              
              {file ? (
                <div>
                  <FileText size={36} color="#3b82f6" style={{ margin: '0 auto 8px' }} />
                  <p style={{ fontSize: '14px', fontWeight: '600', color: '#f3f4f6', wordBreak: 'break-all' }}>{file.name}</p>
                  <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '4px' }}>{(file.size / 1024).toFixed(1)} KB • {isPdf ? 'Multi-page PDF' : 'Image File'}</p>
                </div>
              ) : (
                <div>
                  <Upload size={32} color="#6b7280" style={{ margin: '0 auto 8px' }} />
                  <p style={{ fontSize: '13px', fontWeight: '500', color: '#d1d5db' }}>Drag & Drop document or <span style={{ color: '#3b82f6' }}>browse</span></p>
                  <p style={{ fontSize: '11px', color: '#6b7280', marginTop: '6px' }}>Supports Multi-Page PDF, PNG, JPG, JPEG</p>
                </div>
              )}
            </div>

            {/* Visual Preview if Image */}
            {filePreview && (
              <div style={{ marginTop: '14px', maxHeight: '180px', overflow: 'hidden', borderRadius: '6px', border: '1px solid #1f293d' }}>
                <img src={filePreview} alt="Preview" style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }} />
              </div>
            )}
          </div>

          {/* Extraction Mode Config */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <h2 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '14px', color: '#f3f4f6', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="#8b5cf6" /> 2. Extraction Configuration
            </h2>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
              <button 
                onClick={() => setExtractionMode('discovery')}
                style={{ 
                  flex: 1, padding: '8px 12px', borderRadius: '6px', fontSize: '12px', fontWeight: '600', cursor: 'pointer',
                  border: extractionMode === 'discovery' ? '1px solid #3b82f6' : '1px solid #374151',
                  backgroundColor: extractionMode === 'discovery' ? '#1e3a8a' : '#111827',
                  color: extractionMode === 'discovery' ? '#ffffff' : '#9ca3af'
                }}
              >
                ✨ Dynamic Discovery
              </button>
              <button 
                onClick={() => setExtractionMode('schema')}
                style={{ 
                  flex: 1, padding: '8px 12px', borderRadius: '6px', fontSize: '12px', fontWeight: '600', cursor: 'pointer',
                  border: extractionMode === 'schema' ? '1px solid #8b5cf6' : '1px solid #374151',
                  backgroundColor: extractionMode === 'schema' ? '#4c1d95' : '#111827',
                  color: extractionMode === 'schema' ? '#ffffff' : '#9ca3af'
                }}
              >
                📐 Custom Schema
              </button>
            </div>

            {extractionMode === 'discovery' ? (
              <div style={{ fontSize: '12px', color: '#9ca3af', lineHeight: '1.6', backgroundColor: '#0d1322', padding: '12px', borderRadius: '6px', border: '1px solid #1f293d' }}>
                <p><strong>Universal Extraction:</strong> The model dynamically inspects and extracts all entities, key-values, line-item tables, and lists without forcing any predefined schema.</p>
              </div>
            ) : (
              <div>
                <label style={{ fontSize: '12px', color: '#9ca3af', display: 'block', marginBottom: '4px' }}>Dynamic Schema JSON Definition:</label>
                <textarea 
                  value={customSchemaJson}
                  onChange={(e) => setCustomSchemaJson(e.target.value)}
                  rows={6}
                  style={{ width: '100%', padding: '8px', background: '#090d16', border: '1px solid #374151', borderRadius: '6px', color: '#f3f4f6', fontFamily: 'monospace', fontSize: '11px' }}
                />
              </div>
            )}

            <div style={{ marginTop: '14px' }}>
              <label style={{ fontSize: '12px', color: '#9ca3af', display: 'block', marginBottom: '4px' }}>Custom Guidance / Prompts (Optional):</label>
              <input 
                type="text" 
                value={customInstructions}
                onChange={(e) => setCustomInstructions(e.target.value)}
                placeholder="e.g. Focus on serial numbers and payment terms"
                style={{ width: '100%', padding: '8px 12px', background: '#090d16', border: '1px solid #374151', borderRadius: '6px', color: '#f3f4f6', fontSize: '12px' }}
              />
            </div>

            {/* Run Button */}
            <button 
              onClick={executeExtraction}
              disabled={loading || !file}
              style={{ 
                marginTop: '18px', width: '100%', padding: '12px', borderRadius: '8px', fontWeight: '600', fontSize: '14px',
                backgroundColor: loading || !file ? '#1f293d' : '#2563eb', color: loading || !file ? '#6b7280' : '#ffffff',
                border: 'none', cursor: loading || !file ? 'not-allowed' : 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px'
              }}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="glow-effect" style={{ animation: 'spin 1s linear infinite' }} />
                  Extracting with Qwen-VL-72B...
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Run Dynamic Extraction
                </>
              )}
            </button>
          </div>
        </section>

        {/* Right Column: Interactive Results */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {error && (
            <div style={{ padding: '14px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5', display: 'flex', gap: '10px', alignItems: 'center' }}>
              <AlertCircle size={20} color="#ef4444" />
              <span style={{ fontSize: '13px' }}>{error}</span>
            </div>
          )}

          {result ? (
            <div className="glass-card" style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column' }}>
              
              {/* Header Stats Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1f293d', paddingBottom: '16px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ backgroundColor: '#1e3a8a', color: '#93c5fd', fontSize: '12px', fontWeight: '700', padding: '4px 10px', borderRadius: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {result.document_type || 'Document'}
                  </span>
                  <span style={{ fontSize: '13px', color: '#9ca3af' }}>
                    Processed in <strong style={{ color: '#f3f4f6' }}>{latency}s</strong>
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button 
                    onClick={copyJson}
                    style={{ background: '#1f293d', border: '1px solid #374151', color: '#d1d5db', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    {copied ? <CheckCircle size={14} color="#10b981" /> : <Copy size={14} />}
                    {copied ? 'Copied!' : 'Copy JSON'}
                  </button>
                  <button 
                    onClick={downloadJson}
                    style={{ background: '#2563eb', border: 'none', color: '#ffffff', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <Download size={14} /> Download JSON
                  </button>
                </div>
              </div>

              {/* Navigation Tabs */}
              <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid #1f293d', marginBottom: '16px' }}>
                {[
                  { id: 'fields', label: `Dynamic Fields (${Object.keys(result.document || {}).length})`, icon: FileText },
                  { id: 'tables', label: `Tables (${result.tables?.length || 0})`, icon: Table },
                  { id: 'lists', label: `Lists (${result.lists?.length || 0})`, icon: List },
                  { id: 'json', label: 'Full JSON', icon: Eye },
                  { id: 'warnings', label: `Warnings (${result.warnings?.length || 0})`, icon: AlertCircle }
                ].map(tab => {
                  const Icon = tab.icon;
                  const active = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      style={{
                        padding: '10px 16px', background: 'none', border: 'none', borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
                        color: active ? '#3b82f6' : '#9ca3af', fontWeight: active ? '600' : '400', fontSize: '13px', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', gap: '6px'
                      }}
                    >
                      <Icon size={15} />
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Tab Contents */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                
                {/* TAB 1: DYNAMIC KEY-VALUES */}
                {activeTab === 'fields' && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '12px' }}>
                    {Object.entries(result.document || {}).map(([key, val]) => (
                      <div key={key} style={{ backgroundColor: '#0d1322', padding: '12px 16px', borderRadius: '8px', border: '1px solid #1f293d' }}>
                        <div style={{ fontSize: '11px', color: '#60a5fa', textTransform: 'uppercase', fontWeight: '600', letterSpacing: '0.04em', marginBottom: '4px' }}>
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div style={{ fontSize: '13px', color: '#f3f4f6', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
                          {Array.isArray(val) ? val.join(' • ') : (typeof val === 'object' && val !== null ? JSON.stringify(val) : String(val))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* TAB 2: CONTINUATION TABLES */}
                {activeTab === 'tables' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {result.tables?.map((tbl, tIdx) => (
                      <div key={tIdx} style={{ backgroundColor: '#0d1322', borderRadius: '8px', border: '1px solid #1f293d', overflow: 'hidden' }}>
                        <div style={{ padding: '12px 16px', backgroundColor: '#131b2e', borderBottom: '1px solid #1f293d', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#f3f4f6' }}>{tbl.table_name || `Table ${tIdx + 1}`}</h3>
                            <p style={{ fontSize: '11px', color: '#9ca3af' }}>{tbl.rows?.length || 0} rows • Spanning pages: {tbl.pages?.join(', ') || '1'}</p>
                          </div>
                          <button 
                            onClick={() => downloadTableCsv(tbl)}
                            style={{ background: '#1f293d', border: '1px solid #374151', color: '#93c5fd', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                          >
                            <Download size={12} /> Export CSV
                          </button>
                        </div>
                        <div style={{ overflowX: 'auto', maxHeight: '350px' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                            <thead>
                              <tr style={{ backgroundColor: '#111827', borderBottom: '1px solid #1f293d' }}>
                                {tbl.headers?.map((h, hIdx) => (
                                  <th key={hIdx} style={{ padding: '10px 14px', color: '#9ca3af', fontWeight: '600', whiteSpace: 'nowrap' }}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {tbl.rows?.map((row, rIdx) => (
                                <tr key={rIdx} style={{ borderBottom: '1px solid #172033', backgroundColor: rIdx % 2 === 0 ? 'transparent' : '#0a0f1d' }}>
                                  {row.map((cell, cIdx) => (
                                    <td key={cIdx} style={{ padding: '8px 14px', color: '#e5e7eb', whiteSpace: 'nowrap' }}>{cell}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* TAB 3: LISTS & CLAUSES */}
                {activeTab === 'lists' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {result.lists?.map((lst, lIdx) => (
                      <div key={lIdx} style={{ backgroundColor: '#0d1322', padding: '16px', borderRadius: '8px', border: '1px solid #1f293d' }}>
                        <h3 style={{ fontSize: '13px', fontWeight: '600', color: '#a78bfa', marginBottom: '10px' }}>{lst.list_name}</h3>
                        <ul style={{ paddingLeft: '20px', fontSize: '12px', color: '#d1d5db', lineHeight: '1.7' }}>
                          {lst.items?.map((item, iIdx) => (
                            <li key={iIdx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                )}

                {/* TAB 4: RAW JSON EXPLORER */}
                {activeTab === 'json' && (
                  <pre style={{ backgroundColor: '#070a10', padding: '16px', borderRadius: '8px', border: '1px solid #1f293d', fontSize: '12px', color: '#38bdf8', overflowX: 'auto', maxHeight: '500px' }}>
                    {JSON.stringify(result, null, 2)}
                  </pre>
                )}

                {/* TAB 5: WARNINGS */}
                {activeTab === 'warnings' && (
                  <div>
                    {result.warnings?.length > 0 ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {result.warnings.map((w, wIdx) => (
                          <div key={wIdx} style={{ padding: '10px 14px', backgroundColor: '#1c1917', border: '1px solid #44403c', borderRadius: '6px', fontSize: '12px', color: '#fbbf24' }}>
                            ⚠️ {w}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p style={{ fontSize: '13px', color: '#10b981', padding: '12px' }}>✅ No warnings or extraction anomalies detected.</p>
                    )}
                  </div>
                )}

              </div>
            </div>
          ) : (
            <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', textAlign: 'center', border: '1px dashed #1f293d' }}>
              <Sparkles size={48} color="#374151" style={{ marginBottom: '16px' }} />
              <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#d1d5db' }}>Ready for Intelligent Dynamic Extraction</h3>
              <p style={{ fontSize: '13px', color: '#6b7280', maxWidth: '420px', marginTop: '6px' }}>
                Upload any document (Invoice, Receipt, PO, Agreement, Medical form, etc.) and click &quot;Run Dynamic Extraction&quot; to inspect structured results.
              </p>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
