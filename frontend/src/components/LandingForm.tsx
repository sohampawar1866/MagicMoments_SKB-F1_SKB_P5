import React, { useState, useCallback, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import MapLibreMap from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { LineLayer, PathLayer, PolygonLayer, ScatterplotLayer, GeoJsonLayer } from '@deck.gl/layers';
import * as turf from '@turf/turf';
import gsap from 'gsap';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { StyleSpecification } from 'maplibre-gl';
import api, { apiErrorMessage, type SearchRecord } from '../lib/api';

type CoastlineFeature = GeoJSON.Feature<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;
type CoastlineCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;
type TurfPolygonFeature = GeoJSON.Feature<GeoJSON.Polygon>;
type MapClickInfo = { coordinate?: number[] };
type BasemapMode = 'street' | 'satellite';

type SearchEventSummary = {
  id: string;
  date: string;
  center: [number, number];
  driftVector: [number, number];
  densityPct: number;
  patchAreaM2: number;
  driftDistanceKm: number;
};

type CoastImpactSummary = {
  segmentId: number;
  areaLabel: string;
  intensity: number;
  riskLabel: 'CRITICAL' | 'ELEVATED' | 'WATCH';
  representative: [number, number];
  nearestEvent: SearchEventSummary | null;
  segmentToImpactKm: number | null;
};

const INITIAL_VIEW_STATE = {
  longitude: 80.0,
  latitude: 18.0,
  zoom: 4.2,
  pitch: 0, // Flat 2D Map for clear tactical view
  bearing: 0
};

// Sentinel-2 Red and NIR bands are 10m resolution.
const SENTINEL2_PATCH_SIZE_METERS = 10;
const VISUAL_PREVIEW_PATCH_SIZE_METERS = 100;
const METERS_PER_DEG_LAT = 111320;

const MAP_STYLE_STREET = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const MAP_STYLE_SATELLITE: StyleSpecification = {
  version: 8,
  sources: {
    esri_satellite: {
      type: 'raster',
      tiles: ['https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      attribution: 'Tiles (c) Esri',
    },
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri_satellite',
    },
  ],
};

function haversineDistanceKm(a: [number, number], b: [number, number]): number {
  const [lon1, lat1] = a;
  const [lon2, lat2] = b;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const lat1Rad = toRad(lat1);
  const lat2Rad = toRad(lat2);
  const h =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return 6371 * (2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h)));
}

function coastlineAreaLabel(lon: number, lat: number): string {
  if (lon >= 86.5 && lat >= 20) return 'Sundarbans - Bengal Delta';
  if (lon >= 84 && lat >= 17.5) return 'Odisha - North Bay Coast';
  if (lon >= 80 && lat >= 13) return 'Andhra - Coromandel Belt';
  if (lon >= 78 && lat < 13) return 'Tamil Nadu - Coromandel South';
  if (lon >= 77 && lon < 80 && lat < 10.5) return 'Gulf of Mannar';
  if (lon >= 73 && lon < 77 && lat < 14.5) return 'Kerala - Malabar Coast';
  if (lon >= 72 && lon < 75.5 && lat >= 14.5) return 'Konkan - Goa Shelf';
  if (lon < 72 && lat >= 18) return 'Gujarat - Arabian Arc';
  if (lon < 72) return 'Arabian Offshore Edge';
  return 'Indian Coastline Sector';
}

function extractLinePoints(geometry: GeoJSON.Geometry | null | undefined): [number, number][] {
  if (!geometry) return [];
  if (geometry.type === 'LineString') {
    return geometry.coordinates as [number, number][];
  }
  if (geometry.type === 'MultiLineString') {
    return (geometry.coordinates as [number, number][][]).flat();
  }
  return [];
}

function formatUtc(isoDate: string): string {
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) return isoDate;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

function buildSentinelPatchSquare(lng: number, lat: number, patchMeters: number): [number, number][] {
  const half = patchMeters / 2;
  const latDelta = half / METERS_PER_DEG_LAT;
  const lonScale = Math.max(Math.cos((lat * Math.PI) / 180), 1e-6);
  const lonDelta = half / (METERS_PER_DEG_LAT * lonScale);

  return [
    [lng - lonDelta, lat + latDelta],
    [lng + lonDelta, lat + latDelta],
    [lng + lonDelta, lat - latDelta],
    [lng - lonDelta, lat - latDelta],
  ];
}

export const LandingForm: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 1024);
  const [basemapMode, setBasemapMode] = useState<BasemapMode>('satellite');
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<{ id: string, record: any } | null>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 1024);
    window.addEventListener('resize', onResize);
    
    if (sidebarRef.current) {
        gsap.fromTo(sidebarRef.current, { x: -30, opacity: 0 }, { x: 0, opacity: 1, duration: 0.8, ease: 'power3.out' });
        gsap.fromTo(sidebarRef.current.children, { y: 15, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6, stagger: 0.1, delay: 0.2, ease: 'power2.out' });
    }
    
    return () => window.removeEventListener('resize', onResize);
  }, []);

  React.useEffect(() => {
    const state = location.state as { highlightedId?: string } | null;
    if (state?.highlightedId) {
      const activateTimer = window.setTimeout(() => {
        setHighlightedId(state.highlightedId!);
      }, 0);
      const clearTimer = window.setTimeout(() => {
        setHighlightedId(null);
      }, 3000);
      return () => {
        window.clearTimeout(activateTimer);
        window.clearTimeout(clearTimer);
      };
    }
  }, [location.state]);

  // Single-point targeting; we auto-build a Sentinel-sized square patch.
  const [drawingPoints, setDrawingPoints] = useState<[number, number][]>([]);
  const [currentSelection, setCurrentSelection] = useState<TurfPolygonFeature | null>(null);
  const [processingSelection, setProcessingSelection] = useState<TurfPolygonFeature | null>(null);

  const [coastalGeoJson, setCoastalGeoJson] = useState<CoastlineCollection | null>(null);
  const [searchHistory, setSearchHistory] = useState<SearchRecord[]>([]);
  const [hoverInfo, setHoverInfo] = useState<{ lng: number, lat: number, x: number, y: number, hasObject: boolean } | null>(null);

  const processingBbox = React.useMemo(() => {
    if (!processingSelection?.geometry?.coordinates?.[0]) return null;
    const ring = processingSelection.geometry.coordinates[0].slice(0, 4) as [number, number][];
    const lons = ring.map(([lon]) => lon);
    const lats = ring.map(([, lat]) => lat);
    return {
      minLon: Math.min(...lons),
      maxLon: Math.max(...lons),
      minLat: Math.min(...lats),
      maxLat: Math.max(...lats),
    };
  }, [processingSelection]);

  React.useEffect(() => {
    // Fetch global history from the backend
    api.trackerSearch().then((res) => {
      setSearchHistory(res);
    }).catch((err) => console.error('tracker/search:', apiErrorMessage(err)));

    // Fetch dynamic coastline
    api.trackerCoastline().then((res) => {
      setCoastalGeoJson(res);
    }).catch((err) => console.error('tracker/coastline:', apiErrorMessage(err)));
  }, []);

  const getDensityColor = (density: number): [number, number, number, number] => {
    if (density > 0.7) return [245, 158, 11, 200]; // Gold/Amber
    if (density > 0.4) return [250, 204, 21, 200]; // Yellow gold
    return [16, 185, 129, 200]; // Emerald Green
  };

  const handleMapClick = useCallback((info: MapClickInfo) => {
    if (!info.coordinate || info.coordinate.length < 2) return;
    const [lng, lat] = info.coordinate;

    setScanResult(null);
    setDrawingPoints(() => {
      const nextPoints: [number, number][] = [[lng, lat]];
      const previewSquare = buildSentinelPatchSquare(lng, lat, VISUAL_PREVIEW_PATCH_SIZE_METERS);
      const processingSquare = buildSentinelPatchSquare(lng, lat, SENTINEL2_PATCH_SIZE_METERS);
      setCurrentSelection(turf.polygon([[...previewSquare, previewSquare[0]]]));
      setProcessingSelection(turf.polygon([[...processingSquare, processingSquare[0]]]));
      return nextPoints;
    });
  }, []);

  // Deck.GL Layers
  const activeHistory = searchHistory.slice(-5);
  const mapStyle = basemapMode === 'satellite' ? MAP_STYLE_SATELLITE : MAP_STYLE_STREET;

  const eventSummaries = React.useMemo<SearchEventSummary[]>(() => {
    return searchHistory.map((event) => {
      const ring: [number, number][] = [...event.coordinates];
      if (ring.length >= 3) {
        const [firstLon, firstLat] = ring[0];
        const [lastLon, lastLat] = ring[ring.length - 1];
        if (firstLon !== lastLon || firstLat !== lastLat) {
          ring.push([firstLon, firstLat]);
        }
      }

      const patchAreaM2 = ring.length >= 4 ? turf.area(turf.polygon([ring])) : 0;

      return {
        id: event.id,
        date: event.date,
        center: event.center,
        driftVector: event.driftVector,
        densityPct: event.density * 100,
        patchAreaM2,
        driftDistanceKm: haversineDistanceKm(event.center, event.driftVector),
      };
    });
  }, [searchHistory]);

  const impactedCoastSegments = React.useMemo<CoastImpactSummary[]>(() => {
    if (!coastalGeoJson?.features?.length) return [];

    return coastalGeoJson.features
      .map((feature) => {
        const props = feature.properties ?? {};
        const segmentId = Number((props as { segment_id?: unknown }).segment_id);
        const intensity = Number((props as { intensity?: unknown }).intensity ?? 0);
        const points = extractLinePoints(feature.geometry);
        if (!Number.isFinite(segmentId) || intensity <= 0 || points.length === 0) {
          return null;
        }

        const avgLon = points.reduce((sum, [lon]) => sum + lon, 0) / points.length;
        const avgLat = points.reduce((sum, [, lat]) => sum + lat, 0) / points.length;
        const representative: [number, number] = [avgLon, avgLat];

        let nearestEvent: SearchEventSummary | null = null;
        let minKm = Number.POSITIVE_INFINITY;

        for (const event of eventSummaries) {
          const distKm = haversineDistanceKm(event.driftVector, representative);
          if (distKm < minKm) {
            minKm = distKm;
            nearestEvent = event;
          }
        }

        const riskLabel: CoastImpactSummary['riskLabel'] =
          intensity >= 0.55 ? 'CRITICAL' : intensity >= 0.22 ? 'ELEVATED' : 'WATCH';

        return {
          segmentId,
          areaLabel: coastlineAreaLabel(avgLon, avgLat),
          intensity,
          riskLabel,
          representative,
          nearestEvent,
          segmentToImpactKm: Number.isFinite(minKm) ? minKm : null,
        };
      })
      .filter((item): item is CoastImpactSummary => Boolean(item))
      .sort((a, b) => b.intensity - a.intensity);
  }, [coastalGeoJson, eventSummaries]);

  const topImpactedSegments = React.useMemo(() => impactedCoastSegments.slice(0, 7), [impactedCoastSegments]);

  const impactedBySegmentId = React.useMemo(() => {
    const mapped = new Map<number, CoastImpactSummary>();
    for (const segment of impactedCoastSegments) {
      mapped.set(segment.segmentId, segment);
    }
    return mapped;
  }, [impactedCoastSegments]);

  const layers = [
    // --- COASTAL VULNERABILITY (JAGGED EXACT BORDERS) ---
    coastalGeoJson && new GeoJsonLayer({
      id: 'coastal-risk',
      data: coastalGeoJson,
      stroked: true,
      filled: false,
      getLineColor: (f: CoastlineFeature) => {
        const i =
          f.properties && typeof f.properties === 'object' && 'intensity' in f.properties
            ? Number(f.properties.intensity || 0)
            : 0;
        return i > 0.05 ? [245, 158, 11, 255] : [16, 185, 129, 255];
      },
      getLineWidth: (f: CoastlineFeature) => {
        const i =
          f.properties && typeof f.properties === 'object' && 'intensity' in f.properties
            ? Number(f.properties.intensity || 0)
            : 0;
        return Math.max(50, i * 400);
      },
      lineWidthMinPixels: 3,
      pickable: true,
      autoHighlight: true
    }),

    // --- PREDICTIVE DRIFT VECTOR (FLAT 2D LINE TO COAST) ---
    new LineLayer({
      id: 'drift-vectors',
      data: activeHistory,
      getSourcePosition: (d: SearchRecord) => d.center,
      getTargetPosition: (d: SearchRecord) => d.driftVector,
      getColor: (d: SearchRecord) => getDensityColor(d.density), // Origin box color
      getWidth: 5,
      pickable: true
    }),

    // --- IMPACT ZONES (DOTS ON COAST) ---
    new ScatterplotLayer({
      id: 'impact-zones',
      data: activeHistory,
      getPosition: (d: SearchRecord) => d.driftVector,
      getFillColor: [245, 158, 11, 255],
      getRadius: 6000, // 6 km radius hit marker
      radiusMinPixels: 4,
      pickable: true
    }),

    // --- HISTORICAL BOXES ---
    new PolygonLayer({
      id: 'historical-polygons',
      data: activeHistory,
      getPolygon: (d: SearchRecord) => d.coordinates,
      getFillColor: (d: SearchRecord) => {
        if (d.id === highlightedId) return [255, 255, 255, 200];
        return [...getDensityColor(d.density).slice(0, 3), 80] as [number, number, number, number];
      },
      getLineColor: (d: SearchRecord) => {
        if (d.id === highlightedId) return [255, 255, 255, 255];
        return getDensityColor(d.density);
      },
      getLineWidth: 100,
      stroked: true,
      filled: true,
      wireframe: true,
      pickable: true
    }),

    // --- DRAWING STATE UI ---
    // Draw the auto-generated Sentinel patch border.
    new PathLayer({
      id: 'drawing-border',
      data: currentSelection ? [{ path: currentSelection.geometry.coordinates[0] }] : [],
      getPath: (d: { path: [number, number][] }) => d.path,
      getColor: [16, 185, 129, 255],
      getWidth: 150,
      widthMinPixels: 2
    }),

    // Draw the selected center point as a glowing node.
    new ScatterplotLayer({
      id: 'drawing-nodes',
      data: drawingPoints.map(p => ({ position: p })),
      getPosition: (d: { position: [number, number] }) => d.position,
      getFillColor: [16, 185, 129, 255],
      getRadius: 500,
      radiusMinPixels: 7
    }),

    // Fill the generated patch polygon.
    currentSelection && new PolygonLayer({
      id: 'drawing-fill',
      data: [currentSelection],
      getPolygon: (d: TurfPolygonFeature) => d.geometry.coordinates[0],
      getFillColor: [16, 185, 129, 50],
      filled: true
    })
  ].filter(Boolean);

  const handleSubmit = async () => {
    if (processingSelection && drawingPoints.length === 1) {
      try {
        const patchCoordinates = processingSelection.geometry.coordinates[0].slice(0, 4);
        const record = await api.trackerSubmit(patchCoordinates as Array<[number, number]>);

        // Fetch globally updated search history instead of local hack
        const [histRes, coastRes] = await Promise.all([
          api.trackerSearch(),
          api.trackerCoastline(),
        ]);
        setSearchHistory(histRes);
        setCoastalGeoJson(coastRes);

        // Reset local selection drawing state so we can see the popups
        setDrawingPoints([]);
        setCurrentSelection(null);
        setProcessingSelection(null);

        const center = record.center ?? drawingPoints[0];
        const customAoiId = `custom_${center[0].toFixed(4)}_${center[1].toFixed(4)}`;
        setScanResult({ id: customAoiId, record });
      } catch (err: unknown) {
        console.error(err);
        const msg = apiErrorMessage(err);
        alert(msg.includes('ocean') || msg.includes('land')
          ? msg
          : 'Error deploying sector. Please try an oceanic location.');
        setDrawingPoints([]);
        setCurrentSelection(null);
        setProcessingSelection(null);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', width: '100%', minHeight: '100vh', backgroundColor: 'var(--color-background)', fontFamily: 'var(--font-manrope)' }}>

      {/* Sidebar Panel */}
      <div ref={sidebarRef} className="glass-panel ghost-border" style={{ width: isMobile ? '100%' : '400px', minHeight: isMobile ? 'auto' : '100%', backgroundColor: 'var(--color-surface-container)', borderRight: 'none', borderBottom: 'none', display: 'flex', flexDirection: 'column', zIndex: 10, borderRadius: isMobile ? '0 0 24px 24px' : '0 24px 24px 0', margin: isMobile ? '0 0 10px 0' : '0' }}>

        {/* Main Control Section */}
        <div style={{ padding: isMobile ? '1rem' : '2rem', borderBottom: '1px solid var(--color-surface-variant)' }}>
          <h1 style={{ margin: '0 0 1rem 0', fontSize: isMobile ? '1.2rem' : '1.8rem', color: 'var(--color-text-main)', textTransform: 'uppercase', letterSpacing: isMobile ? '1px' : '2px', fontFamily: 'var(--font-jakarta)' }}>D.R.I.F.T._OS v2.0</h1>

          <div style={{ marginBottom: '1.5rem', fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--color-text-muted)' }}>
            <strong style={{ color: 'var(--color-primary)' }}>&gt; SECTOR DEPLOYMENT</strong><br />
            Click 1 ocean point. D.R.I.F.T. previews 100m x 100m for visibility, but processes a 10m x 10m Sentinel-2 patch.
            <br /><br />
            <strong>Points Logged:</strong> <span style={{ color: 'var(--color-text-main)', fontWeight: 'bold' }}>{drawingPoints.length}/1</span>
          </div>

          {drawingPoints.length === 1 && (
            <div style={{ padding: '1rem', background: 'var(--color-surface-container)', borderLeft: '4px solid var(--color-primary)', marginBottom: '1.5rem', fontSize: '0.9rem', color: 'var(--color-primary)' }}>
              <span style={{ fontWeight: 'bold' }}>[ PATCH LOCKED ]</span><br />
              Center locked. 100m preview + 10m processing AOI generated.
            </div>
          )}

          {processingBbox && (
            <div style={{ padding: '0.8rem', background: 'var(--color-surface-container-low)', border: 'none', borderRadius: '12px', marginBottom: '1.1rem', fontFamily: 'monospace', fontSize: isMobile ? '0.72rem' : '0.78rem', color: 'var(--color-primary)', lineHeight: '1.5', overflowX: 'auto' }}>
              <div style={{ color: 'var(--color-primary)', fontWeight: 700, marginBottom: '0.25rem' }}>PROCESSING BBOX (10m)</div>
              <div>minLon: {processingBbox.minLon.toFixed(7)}</div>
              <div>minLat: {processingBbox.minLat.toFixed(7)}</div>
              <div>maxLon: {processingBbox.maxLon.toFixed(7)}</div>
              <div>maxLat: {processingBbox.maxLat.toFixed(7)}</div>
            </div>
          )}

          {!scanResult ? (
            <button
              onClick={handleSubmit}
              disabled={drawingPoints.length !== 1}
              style={{ width: '100%', padding: '1rem', background: drawingPoints.length === 1 ? 'var(--color-primary)' : 'var(--color-surface-high)', color: drawingPoints.length === 1 ? 'var(--color-on-primary)' : 'var(--color-text-muted)', border: 'none', borderRadius: '12px', cursor: drawingPoints.length === 1 ? 'pointer' : 'not-allowed', fontWeight: 'bold', fontSize: '1rem', textTransform: 'uppercase', transition: 'all 0.3s ease' }}>
              {drawingPoints.length === 1 ? 'Initialize AWS Deep Scan' : 'Awaiting Ocean Point...'}
            </button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ padding: '0.8rem', background: 'var(--color-surface-container)', borderLeft: '4px solid #10b981', fontSize: '0.9rem', color: '#10b981' }}>
                <span style={{ fontWeight: 'bold' }}>[ SCAN COMPLETE ]</span><br />
                Ocean drift trajectory charted on map.
              </div>
              <button
                onClick={() => navigate(`/drift/aoi/${scanResult.id}`, { state: { highlightedId: scanResult.record.id, coordinates: scanResult.record.coordinates } })}
                style={{ width: '100%', padding: '1rem', background: 'var(--color-primary)', color: 'var(--color-on-primary)', border: 'none', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', textTransform: 'uppercase', transition: 'all 0.3s ease', boxShadow: '0 0 15px var(--color-primary)' }}>
                Proceed to Dashboard Snapshot
              </button>
            </div>
          )}

          <button
            onClick={() => navigate('/drift/history')}
            className="btn-secondary"
            style={{ marginTop: '15px', width: '100%' }}
          >
            ACCESS DEPLOYMENT LOGS
          </button>

          <button
            onClick={() => navigate('/drift/dashboard')}
            className="btn-primary"
            style={{ marginTop: '10px', width: '100%' }}
          >
            OPEN INTEL DASHBOARD
          </button>
        </div>

        {/* Threat Legend Section */}
        <div style={{ padding: isMobile ? '1rem' : '2rem', flexGrow: 1 }}>
          <h4 style={{ margin: '0 0 1rem 0', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontSize: '0.85rem', fontWeight: 'bold', letterSpacing: '1px', fontFamily: 'var(--font-jakarta)' }}>Threat Legend & Analytics</h4>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', fontSize: '0.9rem', color: 'var(--color-text-main)' }}>
            <div style={{ width: '16px', height: '16px', background: '#f59e0b', borderRadius: '4px', boxShadow: '0 0 5px #f59e0b' }}></div>
            <span>Critical Coastline Accumulation</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', fontSize: '0.9rem', color: 'var(--color-text-main)' }}>
            <div style={{ width: '20px', height: '3px', background: 'linear-gradient(90deg, #10b981, #f59e0b)' }}></div>
            <span>Flat Vector: Predictive Drift Path</span>
          </div>

          <div style={{ marginTop: '1rem' }}>
            <h5 style={{ margin: '0 0 0.8rem 0', color: 'var(--color-primary)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'var(--font-jakarta)' }}>
              Predicted Coastline Impacts
            </h5>

            {topImpactedSegments.length === 0 ? (
              <div className="glass-panel ghost-border" style={{ borderRadius: '12px', padding: '0.75rem', color: 'var(--color-text-muted)', fontSize: '0.84rem' }}>
                No impacted coastline segments yet. Deploy a sector to generate projected impact intelligence.
              </div>
            ) : (
              <div style={{ display: 'grid', gap: '0.7rem', maxHeight: isMobile ? 240 : 320, overflowY: 'auto', paddingRight: '0.25rem' }}>
                {topImpactedSegments.map((segment) => {
                  const riskColor =
                    segment.riskLabel === 'CRITICAL' ? '#f59e0b' : segment.riskLabel === 'ELEVATED' ? '#facc15' : '#10b981';
                  const nearest = segment.nearestEvent;
                  return (
                    <div key={segment.segmentId} className="glass-panel ghost-border" style={{ borderRadius: '12px', padding: '0.75rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
                        <div style={{ color: 'var(--color-text-main)', fontWeight: 700, fontSize: '0.86rem' }}>{segment.areaLabel}</div>
                        <div style={{ color: riskColor, fontWeight: 800, fontSize: '0.72rem', letterSpacing: '0.06em' }}>{segment.riskLabel}</div>
                      </div>
                      <div style={{ marginTop: '0.35rem', color: 'var(--color-text-muted)', fontSize: '0.76rem', lineHeight: 1.45 }}>
                        Segment #{segment.segmentId} | Intensity {(segment.intensity * 100).toFixed(1)}%
                      </div>
                      {nearest && (
                        <div style={{ marginTop: '0.45rem', color: 'var(--color-text-main)', fontSize: '0.74rem', lineHeight: 1.5 }}>
                          Event {nearest.id} | Plastic signal {nearest.densityPct.toFixed(1)}% | Footprint {nearest.patchAreaM2.toFixed(1)} m2
                          <br />
                          Drift distance {nearest.driftDistanceKm.toFixed(1)} km | Segment offset {(segment.segmentToImpactKm ?? 0).toFixed(1)} km
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Map Content Section */}
      <div style={{ position: 'relative', flexGrow: 1, height: isMobile ? '62vh' : '100vh', minHeight: isMobile ? 420 : undefined }}>
        <DeckGL
          initialViewState={viewState}
          onViewStateChange={({ viewState: nextViewState }) => setViewState(nextViewState as typeof INITIAL_VIEW_STATE)}
          controller={true}
          layers={layers}
          onClick={handleMapClick}
          onHover={(info) => {
            if (info.coordinate) {
              setHoverInfo({
                lng: info.coordinate[0],
                lat: info.coordinate[1],
                x: info.x,
                y: info.y,
                hasObject: Boolean(info.object)
              });
            } else {
              setHoverInfo(null);
            }
          }}
          getTooltip={({ object }) => {
            if (!object) return null;

            const maybeProps = (object as { properties?: { segment_id?: unknown; intensity?: unknown } }).properties;
            const segmentId = Number(maybeProps?.segment_id);
            const coastlineIntel = Number.isFinite(segmentId) ? impactedBySegmentId.get(segmentId) : undefined;

            if (coastlineIntel) {
              const nearest = coastlineIntel.nearestEvent;
              const riskColor =
                coastlineIntel.riskLabel === 'CRITICAL'
                  ? '#f59e0b'
                  : coastlineIntel.riskLabel === 'ELEVATED'
                    ? '#facc15'
                    : '#10b981';

              return {
                html: `
                  <div style="font-family: var(--font-manrope); font-size: 12px; line-height: 1.45; max-width: 260px;">
                    <div style="font-family: var(--font-jakarta); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--color-primary); margin-bottom: 6px;">Coastline Impact Intel</div>
                    <div style="font-weight: 700; color: var(--color-text-main); margin-bottom: 4px;">${coastlineIntel.areaLabel}</div>
                    <div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Risk:</strong> <span style="color: ${riskColor}; font-weight: 800;">${coastlineIntel.riskLabel}</span></div>
                    <div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Impact Intensity:</strong> ${(coastlineIntel.intensity * 100).toFixed(1)}%</div>
                    ${nearest ? `<div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Event:</strong> ${nearest.id}</div>` : ''}
                    ${nearest ? `<div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Plastic Signal:</strong> ${nearest.densityPct.toFixed(1)}%</div>` : ''}
                    ${nearest ? `<div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Patch Footprint:</strong> ${nearest.patchAreaM2.toFixed(1)} m2</div>` : ''}
                    ${nearest ? `<div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Drift Distance:</strong> ${nearest.driftDistanceKm.toFixed(1)} km</div>` : ''}
                    ${coastlineIntel.segmentToImpactKm !== null ? `<div style="color: var(--color-text-main); margin-bottom: 3px;"><strong>Segment Offset:</strong> ${coastlineIntel.segmentToImpactKm.toFixed(1)} km</div>` : ''}
                    ${nearest ? `<div style="color: var(--color-text-muted); margin-top: 5px; font-size: 11px;">Updated: ${formatUtc(nearest.date)}</div>` : ''}
                  </div>
                `,
                style: {
                  backgroundColor: 'rgba(47, 54, 57, 0.65)',
                  border: '1px solid rgba(62, 72, 76, 0.28)',
                  color: 'var(--color-text-main)',
                  borderRadius: '14px',
                  padding: '10px 12px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.35)',
                  zIndex: 10000,
                  marginTop: '10px',
                  left: '10px',
                  backdropFilter: 'blur(14px)',
                  WebkitBackdropFilter: 'blur(14px)'
                }
              };
            }

            if (object.properties?.name) return object.properties.name;
            if (!object.density) return null;
            const risk = object.density > 0.7 ? "CRITICAL" : (object.density > 0.4 ? "ELEVATED" : "LOW");
            const color = risk === "CRITICAL" ? "#f59e0b" : risk === "ELEVATED" ? "#facc15" : "#10b981";
            return {
              html: `
                <div style="font-family: var(--font-manrope); font-size: 13px;">
                  <h4 style="margin: 0 0 5px 0; color: var(--color-primary); border-bottom: 1px solid var(--color-surface-variant); padding-bottom: 5px; font-family: var(--font-jakarta);">SECTOR: ${object.id}</h4>
                  <div style="margin-bottom: 3px; color: var(--color-text-main);"><strong>Risk Level:</strong> <span style="color: ${color}; font-weight: bold;">${risk}</span></div>
                  <div style="margin-bottom: 3px; color: var(--color-text-main);"><strong>Density:</strong> ${(object.density * 100).toFixed(1)}%</div>
                  <div style="margin-bottom: 3px; color: var(--color-text-main);"><strong>Bearing/Target:</strong> ${object.driftVector[1].toFixed(2)}&deg;N, ${object.driftVector[0].toFixed(2)}&deg;E</div>
                  <div style="color: var(--color-text-muted); margin-top: 5px; font-size: 11px;">Deployed: ${object.date}</div>
                </div>
              `,
              style: {
                backgroundColor: 'rgba(47, 54, 57, 0.65)',
                border: '1px solid rgba(62, 72, 76, 0.28)',
                color: 'var(--color-text-main)',
                borderRadius: '14px',
                padding: '12px',
                boxShadow: '0 8px 24px rgba(0,0,0,0.35)',
                zIndex: 10000,
                marginTop: '12px',
                left: '12px',
                backdropFilter: 'blur(14px)',
                WebkitBackdropFilter: 'blur(14px)'
              }
            };
          }}
        >
          <MapLibreMap mapStyle={mapStyle} />
        </DeckGL>

        <div
          className="glass-panel ghost-border"
          style={{
            position: 'absolute',
            right: 14,
            top: 14,
            zIndex: 5,
            borderRadius: 14,
            padding: 6,
            display: 'flex',
            gap: 6,
            boxShadow: '0 8px 22px rgba(0,0,0,0.28)'
          }}
        >
          <button
            onClick={() => setBasemapMode('street')}
            style={{
              border: 'none',
              borderRadius: 10,
              padding: '7px 12px',
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: '0.03em',
              cursor: 'pointer',
              background: basemapMode === 'street' ? 'var(--color-primary)' : 'transparent',
              color: basemapMode === 'street' ? 'var(--color-on-primary)' : 'var(--color-text-main)',
              transition: 'all 0.25s ease'
            }}
            aria-label="Switch to street view"
          >
            Street
          </button>
          <button
            onClick={() => setBasemapMode('satellite')}
            style={{
              border: 'none',
              borderRadius: 10,
              padding: '7px 12px',
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: '0.03em',
              cursor: 'pointer',
              background: basemapMode === 'satellite' ? 'var(--color-primary)' : 'transparent',
              color: basemapMode === 'satellite' ? 'var(--color-on-primary)' : 'var(--color-text-main)',
              transition: 'all 0.25s ease'
            }}
            aria-label="Switch to satellite view"
          >
            Satellite
          </button>
        </div>

        {hoverInfo && !hoverInfo.hasObject && (
          <div className="glass-panel" style={{
            position: 'absolute',
            left: hoverInfo.x + 15,
            top: hoverInfo.y + 15,
            background: 'var(--color-surface-container-high)',
            color: 'var(--color-primary)',
            padding: '6px 10px',
            borderRadius: '12px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1,
            fontFamily: 'monospace',
            border: 'none',
            boxShadow: '0 4px 10px rgba(0,0,0,0.5)'
          }}>
            {hoverInfo.lat.toFixed(4)}&deg;N<br />
            {hoverInfo.lng.toFixed(4)}&deg;E
          </div>
        )}
      </div>
    </div>
  );
};
