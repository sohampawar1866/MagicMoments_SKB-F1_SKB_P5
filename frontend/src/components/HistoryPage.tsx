import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, Trash2 } from 'lucide-react';
import gsap from 'gsap';
import api, { apiErrorMessage, type SearchRecord } from '../lib/api';

export const HistoryPage = () => {
    const [history, setHistory] = useState<SearchRecord[]>([]);
    const [isClearing, setIsClearing] = useState(false);
    const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 900);
    const navigate = useNavigate();
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const onResize = () => setIsMobile(window.innerWidth <= 900);
        window.addEventListener('resize', onResize);
        return () => window.removeEventListener('resize', onResize);
    }, []);

    useEffect(() => {
        if (history.length > 0 && containerRef.current) {
            gsap.fromTo(
                containerRef.current.children,
                { opacity: 0, scale: 0.95 },
                { opacity: 1, scale: 1, duration: 0.4, stagger: 0.05, ease: 'back.out(1.2)' }
            );
        }
    }, [history]);

    useEffect(() => {
        api.trackerSearch()
            .then(setHistory)
            .catch((err) => console.error('tracker/search:', apiErrorMessage(err)));
    }, []);

    const handleRevisit = async (item: SearchRecord) => {
        try {
            await api.trackerRevisit(item.id);
        } catch (error) {
            // Non-fatal; we still navigate so operators can continue.
            console.error('tracker/revisit:', apiErrorMessage(error));
        }
        const [lon, lat] = item.center;
        const customAoiId = `custom_${lon.toFixed(4)}_${lat.toFixed(4)}`;
        navigate(`/drift/aoi/${customAoiId}`, {
            state: {
                highlightedId: item.id,
                coordinates: item.coordinates,
            },
        });
    };

    const handleClearHistory = async () => {
        if (history.length === 0 || isClearing) return;

        const shouldClear = window.confirm('Clear all deployment history records? This cannot be undone.');
        if (!shouldClear) return;

        setIsClearing(true);
        try {
            await api.trackerClearHistory();
            setHistory([]);
        } catch (error) {
            console.error(error);
            window.alert(`Failed to clear history: ${apiErrorMessage(error)}`);
        } finally {
            setIsClearing(false);
        }
    };

    const handleExportHistory = () => {
        if (history.length === 0) {
            window.alert('No history records available to export.');
            return;
        }

        const payload = {
            exportedAt: new Date().toISOString(),
            totalRecords: history.length,
            records: history,
        };

        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `d.r.i.f.t.-history-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    return (
        <div style={{ padding: isMobile ? '14px 14px 18px' : '28px 34px', background: 'var(--color-background)', minHeight: '100vh', color: 'var(--color-text-main)', boxSizing: 'border-box', fontFamily: 'var(--font-manrope)' }}>
            <h2 className="type-page-title" style={{ color: 'var(--color-text-main)', marginBottom: '1.1rem', fontWeight: 'bold' }}>Sector Deployment History</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '30px' }}>
                <button
                    onClick={() => navigate('/drift')}
                    className="btn-secondary"
                    style={{ flex: isMobile ? 1 : undefined, minWidth: isMobile ? '100%' : undefined }}
                >
                    &larr; Return to Dashboard
                </button>

                <button
                    onClick={() => navigate('/drift/dashboard')}
                    className="btn-primary"
                    style={{ flex: isMobile ? 1 : undefined, minWidth: isMobile ? '100%' : undefined }}
                >
                    Open Intel Dashboard
                </button>

                <button
                    onClick={handleExportHistory}
                    className="ghost-border card-hover"
                    style={{ background: 'var(--color-surface-container-high)', color: 'var(--color-primary)', border: 'none', padding: '10px 16px', borderRadius: '4px', cursor: 'pointer', fontWeight: 700, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flex: isMobile ? 1 : undefined, minWidth: isMobile ? '100%' : undefined }}
                >
                    <Download size={16} />
                    Export History
                </button>

                <button
                    onClick={handleClearHistory}
                    disabled={history.length === 0 || isClearing}
                    style={{ background: history.length === 0 || isClearing ? 'var(--color-surface-container)' : 'var(--color-error-container)', color: history.length === 0 || isClearing ? 'var(--color-text-muted)' : 'var(--color-error)', border: 'none', padding: '10px 16px', borderRadius: '4px', cursor: history.length === 0 || isClearing ? 'not-allowed' : 'pointer', fontWeight: 700, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flex: isMobile ? 1 : undefined, minWidth: isMobile ? '100%' : undefined }}
                >
                    <Trash2 size={16} />
                    {isClearing ? 'Clearing...' : 'Clear History'}
                </button>
            </div>

            <div ref={containerRef} style={{ display: 'grid', gap: '15px', width: '100%', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
                {history.slice().reverse().map((item) => (
                    <div key={item.id} className="ghost-border card-hover" style={{ padding: isMobile ? '14px' : '20px', borderRadius: '16px', background: 'var(--color-surface-container)', display: 'flex', justifyContent: 'space-between', flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center', gap: '16px', border: 'none' }}>
                        <div>
                            <h3 style={{ margin: '0 0 10px 0', color: 'var(--color-text-main)', fontWeight: 'bold', fontFamily: 'var(--font-jakarta)' }}>{item.id}</h3>
                            <div className="type-body-md" style={{ color: 'var(--color-text-muted)', lineHeight: '1.5' }}>
                                <strong style={{ color: 'var(--color-primary)' }}>Logged:</strong> {item.date} <br />
                                <strong style={{ color: 'var(--color-primary)' }}>Density Profile:</strong> {(item.density * 100).toFixed(1)}%<br />
                                <strong style={{ color: 'var(--color-primary)' }}>Coordinates:</strong> {item.center[1].toFixed(4)}&deg;N, {item.center[0].toFixed(4)}&deg;E
                            </div>
                        </div>
                        <button
                            onClick={() => handleRevisit(item)}
                            className="btn-primary"
                            style={{ padding: '12px 24px', width: isMobile ? '100%' : 'auto' }}
                        >
                            REVISIT
                        </button>
                    </div>
                ))}
                {history.length === 0 && (
                    <div style={{ color: 'var(--color-text-muted)' }}>No tracking sectors deployed yet.</div>
                )}
            </div>
        </div>
    );
};
