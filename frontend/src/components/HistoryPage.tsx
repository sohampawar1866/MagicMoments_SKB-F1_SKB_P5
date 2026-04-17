import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export const HistoryPage = () => {
    const [history, setHistory] = useState<any[]>([]);
    const navigate = useNavigate();

    useEffect(() => {
        axios.get('http://localhost:8000/api/v1/tracker/search')
            .then(res => setHistory(res.data))
            .catch(console.error);
    }, []);

    const handleRevisit = async (id: string) => {
        await axios.post(`http://localhost:8000/api/v1/tracker/revisit/${id}`);
        navigate('/drift', { state: { highlightedId: id } });
    };

    return (
        <div style={{ padding: '40px', background: '#1e2229', minHeight: '100vh', color: '#e2e8f0', boxSizing: 'border-box', fontFamily: 'Inter, sans-serif' }}>
            <h2 style={{ color: '#f59e0b', marginBottom: '1.5rem', fontWeight: 'bold' }}>Sector Deployment History</h2>
            <button
                onClick={() => navigate('/drift')}
                style={{ background: '#272c35', color: '#e2e8f0', border: '1px solid #38404d', padding: '10px 20px', borderRadius: '4px', cursor: 'pointer', marginBottom: '30px', fontWeight: 'bold', boxShadow: '0 1px 2px rgba(0,0,0,0.2)' }}
            >
                &larr; Return to Dashboard
            </button>

            <div style={{ display: 'grid', gap: '15px', maxWidth: '800px' }}>
                {history.slice().reverse().map((item: any) => (
                    <div key={item.id} style={{ border: '1px solid #38404d', padding: '20px', borderRadius: '8px', background: '#272c35', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)' }}>
                        <div>
                            <h3 style={{ margin: '0 0 10px 0', color: '#e2e8f0', fontWeight: 'bold' }}>{item.id}</h3>
                            <div style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.5' }}>
                                <strong style={{ color: '#cbd5e1' }}>Logged:</strong> {item.date} <br />
                                <strong style={{ color: '#cbd5e1' }}>Density Profile:</strong> {(item.density * 100).toFixed(1)}%<br />
                                <strong style={{ color: '#cbd5e1' }}>Coordinates:</strong> {item.center[1].toFixed(4)}&deg;N, {item.center[0].toFixed(4)}&deg;E
                            </div>
                        </div>
                        <button
                            onClick={() => handleRevisit(item.id)}
                            style={{ padding: '12px 24px', background: '#10b981', color: '#1e2229', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', boxShadow: '0 2px 4px rgba(16, 185, 129, 0.2)' }}
                        >
                            REVISIT ON MAP
                        </button>
                    </div>
                ))}
                {history.length === 0 && (
                    <div style={{ color: '#94a3b8' }}>No tracking sectors deployed yet.</div>
                )}
            </div>
        </div>
    );
};
