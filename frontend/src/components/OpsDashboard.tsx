import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Map from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { PolygonLayer, GeoJsonLayer } from '@deck.gl/layers';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, BarChart2, CheckCircle } from 'lucide-react';
import 'maplibre-gl/dist/maplibre-gl.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const INITIAL_VIEW_STATE = {
  longitude: 72.8,
  latitude: 19.0,
  zoom: 9,
  pitch: 30,
  bearing: 0
};

export const OpsDashboard: React.FC = () => {
  const { aoi_id } = useParams<{ aoi_id: string }>();
  const navigate = useNavigate();

  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [detectionData, setDetectionData] = useState<any>(null);
  const [forecastData, setForecastData] = useState<any>(null);
  const [metricsData, setMetricsData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [timeSlider, setTimeSlider] = useState(24);
  const [generatingMission, setGeneratingMission] = useState(false);

  const fetchDetection = React.useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE_URL}/api/v1/detect?aoi_id=${aoi_id}`);
      setDetectionData(res.data);
    } catch (err) {
      console.error('Error fetching detect data', err);
    } finally {
      setLoading(false);
    }
  }, [aoi_id]);

  const fetchForecast = React.useCallback(async () => {
    if (timeSlider === 0) {
      setForecastData(null);
      return;
    }
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE_URL}/api/v1/forecast?aoi_id=${aoi_id}&hours=${timeSlider}`);
      setForecastData(res.data);
    } catch (err) {
      console.error('Error fetching forecast data', err);
    } finally {
      setLoading(false);
    }
  }, [aoi_id, timeSlider]);

  const fetchDashboardMetrics = React.useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/v1/dashboard/metrics?aoi_id=${aoi_id}`);
      setMetricsData(res.data);
    } catch (err) {
      console.error('Error fetching metrics', err);
    }
  }, [aoi_id]);

  useEffect(() => {
    fetchDashboardMetrics();
    fetchDetection();
    fetchForecast();
    
    // Dynamically update viewState center based on custom string or available AOIs
    if (aoi_id && aoi_id.startsWith('custom_')) {
      const parts = aoi_id.split('_');
      if (parts.length === 3) {
        setViewState(prev => ({
          ...prev,
          longitude: parseFloat(parts[1]),
          latitude: parseFloat(parts[2])
        }));
      }
    } else {
      axios.get(`${API_BASE_URL}/api/v1/aois`).then(res => {
        const aois = res.data.aois;
        const matched = aois.find((a: any) => a.id === aoi_id);
        if (matched) {
          setViewState(prev => ({ ...prev, longitude: matched.center[0], latitude: matched.center[1] }));
        }
      }).catch(err => console.error(err));
    }
  }, [aoi_id, fetchDashboardMetrics, fetchDetection, fetchForecast]);

  const handleExportMission = () => {
    setGeneratingMission(true);
    window.open(`${API_BASE_URL}/api/v1/mission/export?aoi_id=${aoi_id}&format=gpx`, '_blank');
    setTimeout(() => setGeneratingMission(false), 2000);
  };

  // Rendering Layers
  const layers = [
    // Live detection polygons (what the AI found)
    detectionData && new GeoJsonLayer({
      id: 'ai-detections',
      data: detectionData,
      getFillColor: [245, 158, 11, 180], // Gold/Amber polygons
      getLineColor: [245, 158, 11, 255],
      stroked: true,
      filled: true,
      lineWidthMinPixels: 2,
      pickable: true
    }),

    // Forecast polygons (where it is going)
    forecastData && new GeoJsonLayer({
      id: 'drift-forecast',
      data: forecastData,
      getFillColor: [16, 185, 129, 100], // Emerald Green ghost polygons
      getLineColor: [16, 185, 129, 255],
      stroked: true,
      filled: true,
      lineWidthMinPixels: 2,
      getLineDashArray: [3, 3], // Dashed outlines for forecast
      dashJustified: true
    })
  ].filter(Boolean);

  return (
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '2rem', minHeight: '100vh', background: '#1e2229', color: '#e2e8f0', fontFamily: 'Inter, sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #38404d', paddingBottom: '1rem' }}>
        <h2 style={{ margin: 0, color: '#e2e8f0' }}><Activity size={24} style={{ marginRight: '8px', verticalAlign: 'middle', color: '#f59e0b' }} /> OPERATIONS: {aoi_id}</h2>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            onClick={handleExportMission}
            disabled={generatingMission || !detectionData}
            style={{ padding: '0.6rem 1.5rem', background: '#f59e0b', color: '#1e2229', border: 'none', borderRadius: '4px', cursor: (generatingMission || !detectionData) ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}>
            <CheckCircle size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
            {generatingMission ? 'GENERATING...' : 'EXPORT GPX MISSION'}
          </button>
          
          <button onClick={() => navigate('/drift')} style={{ padding: '0.6rem 1.5rem', background: '#272c35', color: '#cbd5e1', border: '1px solid #475569', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
            ABORT & RETURN
          </button>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
        <div style={{ background: '#272c35', borderRadius: '8px', border: '1px solid #38404d', display: 'flex', flexDirection: 'column', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.2)' }}>
          <div style={{ position: 'relative', flexGrow: 1, minHeight: '600px', backgroundColor: '#1e2229', borderRadius: '8px 8px 0 0', overflow: 'hidden' }}>
            <DeckGL
              initialViewState={viewState}
              onViewStateChange={({viewState}) => setViewState(viewState)}
              controller={true}
              layers={layers}
              getTooltip={({object}) => object && (object.properties?.id || "Detection Polygon")}
            >
              <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" />
            </DeckGL>
          </div>
          
          <div style={{ padding: '1.5rem', display: 'flex', gap: '2rem', alignItems: 'center', background: '#2a2f38', borderRadius: '0 0 8px 8px', borderTop: '1px solid #38404d' }}>
            <div style={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <label style={{ color: '#cbd5e1', fontWeight: 'bold' }}>T+ FORECAST (HOURS): <span style={{ color: '#10b981' }}>{timeSlider}h</span></label>
              <input type="range" min="0" max="72" step="24" value={timeSlider} onChange={e => setTimeSlider(Number(e.target.value))} style={{ flexGrow: 1, accentColor: '#10b981' }} />
            </div>
            <button disabled={loading} onClick={fetchForecast} style={{ padding: '0.75rem 2rem', background: '#10b981', color: '#1e2229', fontWeight: 'bold', border: 'none', borderRadius: '4px', cursor: 'pointer', boxShadow: '0 2px 4px rgba(16, 185, 129, 0.2)' }}>
              CALCULATE DRIFT PHYSICS
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ background: '#272c35', borderRadius: '8px', padding: '1.5rem', border: '1px solid #38404d', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.2)' }}>
            <h3 style={{ margin: '0 0 1rem 0', color: '#e2e8f0', fontWeight: 'bold' }}>RADAR LOGS</h3>
            
            <div style={{ padding: '1rem', background: 'rgba(245, 158, 11, 0.1)', borderLeft: '3px solid #f59e0b', marginBottom: '1rem' }}>
              <h4 style={{ margin: '0 0 0.5rem 0', color: '#f59e0b' }}>Current Intel</h4>
              {loading ? <p style={{ margin: 0, color: '#94a3b8' }}>Scanning...</p> : (
                <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: '#cbd5e1' }}>
                  {detectionData ? `Detected ${detectionData.features?.length || 0} anomaly clusters.` : 'No baseline data.'}
                </div>
              )}
            </div>

            <div style={{ padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderLeft: '3px solid #10b981' }}>
              <h4 style={{ margin: '0 0 0.5rem 0', color: '#10b981' }}>Simulation Output</h4>
              {loading ? <p style={{ margin: 0, color: '#94a3b8' }}>Processing vectors...</p> : (
                <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: '#cbd5e1' }}>
                  {forecastData ? `Generated ${forecastData.features?.length || 0} future drift paths.` : 'No active simulation.'}
                </div>
              )}
            </div>
          </div>

          <div style={{ background: '#272c35', borderRadius: '8px', padding: '1.5rem', border: '1px solid #38404d', flexGrow: 1, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.2)' }}>
            <h3 style={{ margin: '0 0 1.5rem 0', color: '#e2e8f0', fontWeight: 'bold' }}><BarChart2 size={18} style={{ marginRight: '8px', verticalAlign: 'middle', color: '#10b981' }} /> PLASTIC DEGRADATION MODEL</h3>
            {metricsData && metricsData.biofouling_chart_data && metricsData.biofouling_chart_data.length > 0 ? (
              <div style={{ height: '250px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsData.biofouling_chart_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#38404d" />
                    <XAxis dataKey="age_days" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#272c35', border: '1px solid #f59e0b', color: '#e2e8f0' }} />
                    <Line type="monotone" dataKey="simulated_confidence" stroke="#10b981" strokeWidth={3} dot={{ fill: '#10b981' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p style={{ color: '#94a3b8', textAlign: 'center', marginTop: '2rem' }}>No atmospheric degradation metrics logged.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
