import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api, { apiErrorMessage, type SearchRecord } from '../lib/api';

export const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<SearchRecord[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.trackerSearch()
      .then(setHistory)
      .catch(err => setErrorMsg(apiErrorMessage(err)));
  }, []);

  /** REVISIT: mark the record active on the backend, then open the Ops
   *  Dashboard for that sector so the full detect → forecast → mission chain
   *  runs against its centroid. */
  const handleRevisit = async (record: SearchRecord) => {
    try {
      await api.trackerRevisit(record.id);
    } catch (err) {
      // Non-fatal — we still want to open the dashboard even if the revisit
      // POST fails (frontend shouldn't block on telemetry).
      console.error('tracker/revisit:', apiErrorMessage(err));
    }
    const [lon, lat] = record.center;
    const aoiId = `custom_${lon.toFixed(4)}_${lat.toFixed(4)}`;
    navigate(`/drift/aoi/${aoiId}`, { state: { highlightedId: record.id } });
  };

  return (
    <div style={{
      padding: '40px', background: '#1e2229', minHeight: '100vh',
      color: '#e2e8f0', boxSizing: 'border-box', fontFamily: 'Inter, sans-serif',
    }}>
      <h2 style={{ color: '#f59e0b', marginBottom: '1.5rem', fontWeight: 'bold' }}>
        Sector Deployment History
      </h2>

      <button
        onClick={() => navigate('/drift')}
        style={{
          background: '#272c35', color: '#e2e8f0', border: '1px solid #38404d',
          padding: '10px 20px', borderRadius: '4px', cursor: 'pointer',
          marginBottom: '30px', fontWeight: 'bold',
          boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
        }}
      >
        &larr; Return to Dashboard
      </button>

      {errorMsg && (
        <div style={{
          maxWidth: '800px', marginBottom: '1.5rem', padding: '0.75rem 1rem',
          background: 'rgba(220, 38, 38, 0.15)', borderLeft: '4px solid #dc2626',
          color: '#fecaca', borderRadius: '4px',
        }}>
          {errorMsg}
        </div>
      )}

      <div style={{ display: 'grid', gap: '15px', maxWidth: '800px' }}>
        {history.slice().reverse().map(item => (
          <div key={item.id} style={{
            border: '1px solid #38404d', padding: '20px', borderRadius: '8px',
            background: '#272c35', display: 'flex',
            justifyContent: 'space-between', alignItems: 'center',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)',
          }}>
            <div>
              <h3 style={{ margin: '0 0 10px 0', color: '#e2e8f0', fontWeight: 'bold' }}>
                {item.id}
              </h3>
              <div style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.5' }}>
                <strong style={{ color: '#cbd5e1' }}>Logged:</strong> {item.date} <br />
                <strong style={{ color: '#cbd5e1' }}>Density Profile:</strong>{' '}
                {(item.density * 100).toFixed(1)}%<br />
                <strong style={{ color: '#cbd5e1' }}>Coordinates:</strong>{' '}
                {item.center[1].toFixed(4)}&deg;N, {item.center[0].toFixed(4)}&deg;E
              </div>
            </div>
            <button
              onClick={() => handleRevisit(item)}
              style={{
                padding: '12px 24px', background: '#10b981', color: '#1e2229',
                border: 'none', borderRadius: '4px', cursor: 'pointer',
                fontWeight: 'bold',
                boxShadow: '0 2px 4px rgba(16, 185, 129, 0.2)',
              }}
            >
              REVISIT ON MAP
            </button>
          </div>
        ))}
        {history.length === 0 && !errorMsg && (
          <div style={{ color: '#94a3b8' }}>No tracking sectors deployed yet.</div>
        )}
      </div>
    </div>
  );
};
