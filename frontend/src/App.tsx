import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { NewLandingPage } from './components/NewLandingPage';
import { DriftAppShell } from './components/DriftAppShell';
import { DRIFT_ROUTE_CONFIG } from './config/driftRouteConfig';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container" style={{ width: '100%', height: '100vh', margin: 0, padding: 0 }}>
        <Routes>
          <Route path="/" element={<NewLandingPage />} />
          <Route path="/drift" element={<DriftAppShell />}>
            {DRIFT_ROUTE_CONFIG.map((route) =>
              route.index ? (
                <Route key={route.key} index element={route.element} />
              ) : (
                <Route key={route.key} path={route.path} element={route.element} />
              )
            )}
          </Route>
        </Routes>
      </div>
    </Router>
  );
}

export default App;
