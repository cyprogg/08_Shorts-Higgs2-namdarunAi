interface ReferenceAnalysisPanelProps {
  readonly analysisMarkdown: string | null;
  readonly isBusy: boolean;
  readonly onAnalyze: () => void;
  readonly selectedLlm?: string;
}

export function ReferenceAnalysisPanel({
  analysisMarkdown,
  isBusy,
  onAnalyze,
  selectedLlm
}: ReferenceAnalysisPanelProps) {
  return (
    <section className="panel analysis-panel" aria-labelledby="analysis-title">
      <div className="section-header">
        <div>
          <p className="eyebrow">Analysis</p>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <h2 id="analysis-title">Reference report</h2>
            {selectedLlm && (
              <span
                style={{
                  fontSize: "0.75rem",
                  background: "#f0fdf4",
                  color: "#166534",
                  border: "1px solid #bbf7d0",
                  padding: "2px 8px",
                  borderRadius: "10px",
                  fontWeight: 600
                }}
              >
                🤖 {selectedLlm}
              </span>
            )}
          </div>
        </div>
        <button className="secondary-button" disabled={isBusy} onClick={onAnalyze} type="button">
          {isBusy ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <span className="spinner-icon">⏳</span> 4. 레퍼런스 분석 진행 중...
            </span>
          ) : (
            "4. Analyze Reference"
          )}
        </button>
      </div>
      <pre className="script-preview">
        {analysisMarkdown ?? "Analysis report will appear here after script analysis."}
      </pre>
    </section>
  );
}
