import React, { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { House, Menu } from 'lucide-react';
import { DRIFT_NAV_ITEMS } from '../config/driftRouteConfig';

export const DriftAppShell: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
      <aside
        style={{
          width: open ? 220 : 72,
          transition: 'width 0.2s ease',
          background: '#202631',
          borderRight: '1px solid #38404d',
          padding: '16px 10px 18px',
          boxSizing: 'border-box',
          position: 'sticky',
          top: 0,
          height: '100vh',
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <button
          onClick={() => setOpen((prev) => !prev)}
          style={{
            width: '100%',
            height: 42,
            borderRadius: 10,
            border: '1px solid #3f4959',
            background: '#2a3340',
            color: '#cbd5e1',
            display: 'flex',
            alignItems: 'center',
            justifyContent: open ? 'space-between' : 'center',
            padding: open ? '0 10px' : 0,
            cursor: 'pointer',
            marginBottom: 16,
          }}
        >
          {open && <span style={{ fontSize: 12, letterSpacing: '0.08em', fontWeight: 700 }}>DRIFT NAV</span>}
          <Menu size={16} />
        </button>

        <div style={{ display: 'grid', gap: 12 }}>
          {DRIFT_NAV_ITEMS.map((item) => {
            const active = item.activePrefixes.some((prefix) =>
              prefix === '/drift' ? location.pathname === '/drift' : location.pathname.startsWith(prefix)
            );
            const Icon = item.icon;
            return (
              <button
                key={item.label}
                onClick={() => navigate(item.to)}
                style={{
                  width: '100%',
                  minHeight: 44,
                  borderRadius: 11,
                  border: active ? '1px solid #279a74' : '1px solid #34404f',
                  background: active ? '#1f7a5d' : '#26303d',
                  color: active ? '#eaf8f3' : '#d6dfeb',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: open ? 'flex-start' : 'center',
                  gap: 12,
                  padding: open ? '0 12px' : 0,
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                <Icon size={16} />
                {open && <span style={{ fontSize: 13 }}>{item.label}</span>}
              </button>
            );
          })}
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button
            onClick={() => navigate('/')}
            style={{
              width: '100%',
              minHeight: 44,
              borderRadius: 11,
              border: '1px solid #415063',
              background: '#2a3340',
              color: '#e2e8f0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: open ? 'flex-start' : 'center',
              gap: 12,
              padding: open ? '0 12px' : 0,
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            <House size={16} />
            {open && <span style={{ fontSize: 13 }}>Landing</span>}
          </button>
        </div>
      </aside>

      <div style={{ flex: 1, minWidth: 0 }}>
        <Outlet />
      </div>
    </div>
  );
};
