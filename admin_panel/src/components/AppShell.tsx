import type { ReactNode } from 'react';
import { Activity, BookOpenText, Gauge, Headphones, LogOut, Settings2, Sparkles } from 'lucide-react';

import { cn } from '../lib/utils';

export type AdminView = 'dashboard' | 'voice-studio' | 'knowledge-hub' | 'provider-settings';

interface AppShellProps {
  activeView: AdminView;
  onViewChange: (view: AdminView) => void;
  onLogout: () => void;
  children: ReactNode;
}

const navigation: Array<{ view: AdminView; label: string; icon: typeof Gauge }> = [
  { view: 'dashboard', label: 'Command center', icon: Gauge },
  { view: 'voice-studio', label: 'Voice studio', icon: Headphones },
  { view: 'knowledge-hub', label: 'Knowledge hub', icon: BookOpenText },
  { view: 'provider-settings', label: 'Provider settings', icon: Settings2 },
];

/** Application frame with responsive navigation and a fixed operational context. */
export function AppShell({ activeView, onViewChange, onLogout, children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-mesh">
      <div className="mx-auto flex min-h-screen max-w-[1700px]">
        <aside className="hidden w-72 shrink-0 border-r border-slate-800/80 bg-slate-950/70 px-5 py-6 lg:flex lg:flex-col">
          <Brand />
          <nav className="mt-10 space-y-1.5" aria-label="Admin console navigation">
            {navigation.map(({ view, label, icon: Icon }) => (
              <button
                key={view}
                type="button"
                onClick={() => onViewChange(view)}
                aria-current={activeView === view ? 'page' : undefined}
                className={cn(
                  'flex w-full items-center gap-3 rounded-xl px-3.5 py-3 text-left text-sm font-medium transition',
                  activeView === view
                    ? 'bg-indigo-500/15 text-white shadow-[inset_0_0_0_1px_rgba(129,140,248,0.22)]'
                    : 'text-slate-400 hover:bg-slate-900 hover:text-slate-100',
                )}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
          <div className="mt-auto rounded-2xl border border-indigo-400/15 bg-indigo-500/[0.07] p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-indigo-100"><Sparkles className="h-4 w-4" /> Local-first control</div>
            <p className="mt-2 text-xs leading-5 text-indigo-200/70">Cloud providers remain optional. Credentials never enter this console.</p>
          </div>
          <button type="button" onClick={onLogout} className="mt-4 flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-400 hover:bg-slate-900 hover:text-white">
            <LogOut className="h-4 w-4" aria-hidden="true" />
            Sign out
          </button>
        </aside>

        <main className="min-w-0 flex-1 px-4 pb-8 pt-4 sm:px-7 sm:pt-7 lg:px-10">
          <header className="mb-7 flex items-center justify-between gap-4 lg:hidden">
            <Brand compact />
            <div className="flex items-center gap-2">
              <div className="flex rounded-xl border border-slate-800 bg-slate-950/70 p-1" aria-label="Mobile navigation">
                {navigation.map(({ view, label, icon: Icon }) => (
                  <button
                    key={view}
                    type="button"
                    onClick={() => onViewChange(view)}
                    className={cn('rounded-lg p-2', activeView === view ? 'bg-indigo-500/20 text-white' : 'text-slate-400')}
                    aria-label={label}
                    aria-current={activeView === view ? 'page' : undefined}
                  >
                    <Icon className="h-4 w-4" />
                  </button>
                ))}
              </div>
              <button type="button" onClick={onLogout} className="rounded-lg p-2 text-slate-400 hover:bg-slate-900 hover:text-white" aria-label="Sign out">
                <LogOut className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </header>
          {children}
        </main>
      </div>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-400 to-violet-600 shadow-lg shadow-indigo-950/70">
        <Activity className="h-5 w-5 text-white" aria-hidden="true" />
      </div>
      <div className={compact ? 'hidden sm:block' : ''}>
        <p className="text-base font-bold tracking-tight text-white">OmniVox</p>
        <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500">Command Center</p>
      </div>
    </div>
  );
}
