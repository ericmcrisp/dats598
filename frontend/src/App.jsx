import { useState } from 'react';

function App() {
  const [textInput, setTextInput] = useState('');
  const [factCheckResult, setFactCheckResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const [config, setConfig] = useState({
    embedding_model_common_name: 'mini-L6',
    claim_confidence_threshold: 0.6,
    evidence_top_k: 5,
    evidence_min_similarity: 0.3,
    supports_threshold: 0.5,
    mode: 'rules'
  });

  // Effect to clear success message after a delay
  const clearSuccessMessage = () => {
    setTimeout(() => setSuccessMessage(null), 3000); // Clear after 3 seconds
  };

  const handleConfigChange = (e) => {
    const { name, value } = e.target;
    setConfig((prev) => {
      const numericFields = [
        'claim_confidence_threshold',
        'evidence_top_k',
        'evidence_min_similarity',
        'supports_threshold'
      ];
      return {
        ...prev,
        [name]: numericFields.includes(name) ? parseFloat(value) : value,
      };
    });
  };

  const saveConfig = async () => {
    setError(null);
    setSuccessMessage(null);

    try {
      const res = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!res.ok) {
        const errorText = await res.text();
        setError(`Config error: ${errorText}`);
        return;
      }

      await res.json();
      // Replacing alert() with state-driven UI message
      setSuccessMessage('Configuration saved successfully!');
      clearSuccessMessage();

    } catch (err) {
      setError(`Config error: ${err.message}`);
    }
  };

  const handleFactCheckSubmit = async () => {
    setLoading(true);
    setFactCheckResult(null);
    setError(null);
    setSuccessMessage(null);  

    try {
      const res = await fetch('/api/factcheck', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textInput }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        setError(`Error ${res.status}: ${errorText}`);
        return;
      }

      const data = await res.json();
      console.log('Response:', data);
      setFactCheckResult(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (verdict) => {
    if (verdict === 'SUPPORTS') return 'text-green-600 bg-green-50 border-green-200';
    if (verdict === 'REFUTES') return 'text-red-600 bg-red-50 border-red-200';
    return 'text-orange-600 bg-orange-50 border-orange-200';
  };

  const getVerdictIcon = (verdict) => {
    if (verdict === 'SUPPORTS') return '✓';
    if (verdict === 'REFUTES') return '✗';
    return '?';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-blue-50 to-indigo-200 backdrop-blur-sm flex items-center justify-center py-8">
      <div className="max-w-5xl mx-full px-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-800 mb-2">FactCheck AI</h1>
          <p className="text-gray-600">Verify claims with AI-powered fact checking</p>
        </div>

        {/* Success Display */}
        {successMessage && (
          <div className="p-4 bg-green-50 border-l-4 border-green-500 rounded-lg mb-6 shadow">
            <p className="text-green-700 font-medium">✅ {successMessage}</p>
          </div>
        )}

        {/* Config Toggle */}
        <div className="text-center mb-6">
          <button 
            onClick={() => setShowConfig(!showConfig)}
            className="bg-white text-gray-700 px-6 py-2 rounded-lg shadow hover:shadow-md transition-shadow border border-gray-200"
          >
            {showConfig ? '⚙️ Hide Settings' : '⚙️ Show Settings'}
          </button>
        </div>

        {/* Config Panel */}
        {showConfig && (
          <div className="mb-6 p-6 bg-white rounded-lg shadow-lg border border-gray-200">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">Configuration</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block font-medium mb-2 text-gray-700">Embedding Model</label>
                <select
                  name="embedding_model_name"
                  value={config.embedding_model_name}
                  onChange={handleConfigChange}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="mini_L6">all-MiniLM-L6-v2</option>
                  <option value="mini_L12">all-MiniLM-L12-v2</option>
                  <option value="paraphase_L6">paraphrase-MiniLM-L6-v2</option>
                  <option value="Gemma3">tencent/KaLM-Embedding-Gemma3-12B-2511</option>
                  <option value="e5small">intfloat/e5-small-v2</option>

                </select>
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">
                  Claim Confidence: {config.claim_confidence_threshold}
                </label>
                <input
                  type="range"
                  name="claim_confidence_threshold"
                  value={config.claim_confidence_threshold}
                  step="0.01"
                  min="0"
                  max="1"
                  onChange={handleConfigChange}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">
                  Evidence Top K: {config.evidence_top_k}
                </label>
                <input
                  type="range"
                  name="evidence_top_k"
                  value={config.evidence_top_k}
                  min="1"
                  max="10"
                  onChange={handleConfigChange}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">
                  Min Similarity: {config.evidence_min_similarity}
                </label>
                <input
                  type="range"
                  name="evidence_min_similarity"
                  value={config.evidence_min_similarity}
                  step="0.05"
                  min="0"
                  max="1"
                  onChange={handleConfigChange}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">
                  Supports Threshold: {config.supports_threshold}
                </label>
                <input
                  type="range"
                  name="supports_threshold"
                  value={config.supports_threshold}
                  step="0.05"
                  min="0"
                  max="1"
                  onChange={handleConfigChange}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">Pipeline Mode</label>
                <select
                  name="mode"
                  value={config.mode}
                  onChange={handleConfigChange}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="rules">Rules-Based</option>
                  <option value="llm">LLM-Based (GPT)</option>
                </select>
              </div>

              <div>
                <label className="block font-medium mb-2 text-gray-700">Pipeline Mode</label>
                <select
                  name="claim_mode"
                  value={config.claim_mode}
                  onChange={handleConfigChange}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="simple">Simplistic Claim Detection</option>
                  <option value="advanced">More Advanced Claim Detection</option>
                </select>
              </div>

            </div>

            <button 
              onClick={saveConfig}
              className="mt-4 bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors shadow"
            >
              💾 Save Configuration
            </button>
          </div>
        )}

        {/* Input Section */}
        <div className="mb-6 p-6 bg-white rounded-lg shadow-lg border border-gray-200">
          <label className="block font-semibold mb-3 text-gray-800">Enter text to fact-check:</label>
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Enter claims to verify... e.g., 'The Eiffel Tower was built in 1889.'"
            rows={6}
            className="w-full border border-gray-300 rounded-lg p-4 mb-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button 
            onClick={handleFactCheckSubmit}
            disabled={loading || !textInput.trim()}
            className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg hover:from-blue-600 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all shadow-md font-semibold text-lg"
          >
            {loading ? '🔍 Checking...' : '🔍 Fact-Check'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded-lg mb-6 shadow">
            <p className="text-red-700 font-medium">⚠️ {error}</p>
          </div>
        )}

        {/* Results Display */}
        {factCheckResult && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-800">Results</h2>
              <button 
                onClick={() => setShowRaw(!showRaw)}
                className="text-sm bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors"
              >
                {showRaw ? '📊 Show Formatted' : '🔧 Show Raw JSON'}
              </button>
            </div>

            {showRaw ? (
              <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
                <pre className="overflow-auto text-sm text-gray-800">
                  {JSON.stringify(factCheckResult, null, 2)}
                </pre>
              </div>
            ) : (
              <>
                {/* Claims */}
                {factCheckResult.claims && factCheckResult.claims.length > 0 ? (
                  <div className="space-y-4">
                    {factCheckResult.claims.map((verification, idx) => {
                      const claim = verification.claim || {};
                      const verdict = verification.verdict || 'UNKNOWN';
                      
                      return (
                        <div 
                          key={idx} 
                          className="bg-white rounded-lg shadow-lg border border-gray-200 p-6 hover:shadow-xl transition-shadow"
                        >
                          {/* Claim Header */}
                          <div className="flex items-start gap-4 mb-4">
                            <div className={`text-3xl font-bold px-3 py-1 rounded-lg border-2 ${getVerdictColor(verdict)}`}>
                              {getVerdictIcon(verdict)}
                            </div>
                            <div className="flex-1">
                              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                                {claim.text || 'No claim text'}
                              </h3>
                              <div className="flex flex-wrap gap-2">
                                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getVerdictColor(verdict)}`}>
                                  {verdict}
                                </span>
                                {verification.confidence !== undefined && (
                                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700 border border-gray-300">
                                    Confidence: {(verification.confidence * 100).toFixed(0)}%
                                  </span>
                                )}
                                {claim.type && (
                                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-700 border border-blue-300">
                                    {claim.type}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>

                          {/* Explanation */}
                          {verification.explanation && (
                            <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                              <h4 className="font-semibold text-gray-800 mb-2">📝 Explanation</h4>
                              <p className="text-gray-700">{verification.explanation}</p>
                            </div>
                          )}

                          {/* Best Evidence */}
                          {verification.best_evidence && (
                            <div className="mb-4 p-4 bg-green-50 rounded-lg border border-green-200">
                              <h4 className="font-semibold text-green-800 mb-2">⭐ Best Evidence</h4>
                              <p className="text-gray-700 mb-2">{verification.best_evidence.text}</p>
                              <p className="text-sm text-green-600">
                                Source: {verification.best_evidence.source || 'Unknown'}
                              </p>
                            </div>
                          )}

                          {/* All Evidence */}
                          {verification.all_evidence && verification.all_evidence.length > 0 && (
                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                              <h4 className="font-semibold text-blue-800 mb-3">
                                📚 All Evidence ({verification.all_evidence.length})
                              </h4>
                              <div className="space-y-2">
                                {verification.all_evidence.map((ev, i) => (
                                  <div key={i} className="p-3 bg-white rounded border border-blue-100">
                                    <p className="text-gray-700 text-sm mb-1">{ev.text}</p>
                                    <p className="text-xs text-blue-600">
                                      {ev.source || 'Unknown source'}
                                      {ev.similarity !== undefined && ` • Similarity: ${(ev.similarity * 100).toFixed(0)}%`}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Stats */}
                          {(verification.evidence_count || verification.max_similarity || verification.avg_similarity) && (
                            <div className="mt-4 flex flex-wrap gap-4 text-sm text-gray-600">
                              {verification.evidence_count !== undefined && (
                                <span>📊 Evidence: {verification.evidence_count}</span>
                              )}
                              {verification.max_similarity !== undefined && (
                                <span>🎯 Max Similarity: {(verification.max_similarity * 100).toFixed(1)}%</span>
                              )}
                              {verification.avg_similarity !== undefined && (
                                <span>📈 Avg Similarity: {(verification.avg_similarity * 100).toFixed(1)}%</span>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-8 text-center">
                    <p className="text-gray-600">No claims found in the response</p>
                  </div>
                )}

                {/* Summary */}
                {factCheckResult.summary && (
                  <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold mb-4 text-gray-800">📊 Summary</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(factCheckResult.summary).map(([key, value]) => (
                        <div key={key} className="text-center p-4 bg-gray-50 rounded-lg border border-gray-200">
                          <div className="text-2xl font-bold text-gray-800">
                            {typeof value === 'number' ? value.toFixed(2) : value}
                          </div>
                          <div className="text-sm text-gray-600 mt-1">
                            {key.replace(/_/g, ' ').toUpperCase()}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;