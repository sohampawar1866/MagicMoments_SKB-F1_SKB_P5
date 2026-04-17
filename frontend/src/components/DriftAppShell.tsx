import React, { useEffect, useRef, useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Gauge, History, House, Map, Menu, X } from 'lucide-react';
import { DRIFT_NAV_ITEMS } from '../config/driftRouteConfig';
import gsap from 'gsap';

export const DriftAppShell: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 900);
  const navContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onResize = () => {
      const mobile = window.innerWidth <= 900;
      setIsMobile(mobile);
      if (!mobile) {
        setOpen(false);
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    if (navContainerRef.current) {
      gsap.fromTo(
        navContainerRef.current.children,
        { opacity: 0, x: -20 },
        { opacity: 1, x: 0, duration: 0.5, stagger: 0.1, ease: 'power2.out' }
      );
    }
  }, []);

  const mobileItems = [
    { label: 'Map', to: '/drift', icon: Map, activePrefixes: ['/drift'] },
    { label: 'History', to: '/drift/history', icon: History, activePrefixes: ['/drift/history'] },
    { label: 'Intel', to: '/drift/dashboard', icon: Gauge, activePrefixes: ['/drift/dashboard'] },
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', width: '100%', position: 'relative', backgroundColor: 'var(--color-background)' }}>
      {isMobile && (
        <button
          onClick={() => setOpen((prev) => !prev)}
          style={{
            position: 'fixed',
            top: 14,
            left: 12,
            zIndex: 120,
            width: 40,
            height: 40,
            borderRadius: '50%',
            border: 'none',
            background: 'var(--color-surface-container-highest)',
            color: 'var(--color-primary)',
            display: 'grid',
            placeItems: 'center',
            cursor: 'pointer',
            boxShadow: '0 8px 32px rgba(0,22,37,0.3)',
          }}
          aria-label="Toggle D.R.I.F.T. navigation"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      )}

      {isMobile && open && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(8, 15, 18, 0.7)',
            backdropFilter: 'blur(10px)',
            zIndex: 90,
          }}
        />
      )}

      <aside
        style={{
          width: isMobile ? 240 : open ? 220 : 72,
          transition: isMobile ? 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)' : 'width 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
          background: 'var(--color-surface-container-lowest)',
          padding: '16px 10px 18px',
          boxSizing: 'border-box',
          position: isMobile ? 'fixed' : 'sticky',
          top: 0,
          left: 0,
          height: '100vh',
          transform: isMobile ? (open ? 'translateX(0)' : 'translateX(-105%)') : 'none',
          zIndex: 100,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: isMobile ? '10px 0 30px rgba(0,0,0,0.5)' : 'none'
        }}
      >
        <button
          onClick={() => setOpen((prev) => !prev)}
          style={{
            width: '100%',
            height: 42,
            borderRadius: 9999,
            border: 'none',
            background: 'var(--color-surface-container-highest)',
            color: 'var(--color-text-main)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: open || isMobile ? 'space-between' : 'center',
            padding: open || isMobile ? '0 16px' : 0,
            cursor: 'pointer',
            marginBottom: 24,
            opacity: isMobile ? 0 : 1,
            pointerEvents: isMobile ? 'none' : 'auto',
            transition: 'background 0.3s'
          }}
          className="card-hover"
        >
          {(open || isMobile) && <span style={{ fontSize: 13, letterSpacing: '0.1em', fontWeight: 700, fontFamily: 'var(--font-jakarta)' }}>NAVIGATE</span>}
          <Menu size={16} />
        </button>

        <div ref={navContainerRef} style={{ display: 'grid', gap: 12 }}>
          {DRIFT_NAV_ITEMS.map((item) => {
            const active = item.activePrefixes.some((prefix) =>
              prefix === '/drift' ? location.pathname === '/drift' : location.pathname.startsWith(prefix)
            );
            const Icon = item.icon;
            return (
              <button
                key={item.label}
                onClick={() => {
                  navigate(item.to);
                  if (isMobile) setOpen(false);
                }}
                style={{
                  width: '100%',
                  minHeight: 48,
                  borderRadius: 9999,
                  border: 'none',
                  background: active ? 'var(--color-surface-container-highest)' : 'transparent',
                  color: active ? 'var(--color-primary)' : 'var(--color-text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: open || isMobile ? 'flex-start' : 'center',
                  gap: 16,
                  padding: open || isMobile ? '0 16px' : 0,
                  cursor: 'pointer',
                  fontWeight: 600,
                  transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                  boxShadow: active ? '0 4px 12px rgba(107, 212, 242, 0.1)' : 'none'
                }}
                className={!active ? "card-hover" : ""}
              >
                <Icon size={18} />
                {(open || isMobile) && <span style={{ fontSize: 13 }}>{item.label}</span>}
              </button>
            );
          })}
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button
            onClick={() => {
              navigate('/');
              if (isMobile) setOpen(false);
            }}
            style={{
              width: '100%',
              minHeight: 48,
              borderRadius: 9999,
              border: 'none',
              background: 'transparent',
              color: 'var(--color-text-muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: open || isMobile ? 'flex-start' : 'center',
              gap: 16,
              padding: open || isMobile ? '0 16px' : 0,
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'background 0.3s'
            }}
            className="card-hover ghost-border"
          >
            <House size={18} />
            {(open || isMobile) && <span style={{ fontSize: 13 }}>Landing</span>}
          </button>
        </div>
      </aside>

      <div style={{ flex: 1, minWidth: 0, paddingBottom: isMobile ? 72 : 0, zIndex: 1 }}>
        <Outlet />
      </div>

      {isMobile && (
        <nav
          className="glass-panel"
          style={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            height: 72,
            border: 'none',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            zIndex: 110,
            borderRadius: '24px 24px 0 0',
            boxShadow: '0 -10px 40px rgba(0,0,0,0.3)'
          }}
        >
          {mobileItems.map((item) => {
            const active = item.activePrefixes.some((prefix) =>
              prefix === '/drift' ? location.pathname === '/drift' : location.pathname.startsWith(prefix)
            );
            const Icon = item.icon;
            return (
              <button
                key={item.label}
                onClick={() => navigate(item.to)}
                style={{
                  border: 'none',
                  background: 'transparent',
                  color: active ? 'var(--color-primary)' : 'var(--color-text-muted)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  fontSize: 11,
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
                aria-label={`Open ${item.label}`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      )}
    </div>
  );
};

