// Per-phase accent colors, shared between App.jsx and the phase tab bar.
export const PHASE_COLORS = {
  1: '#3b82f6',
  2: '#8b5cf6',
  3: '#10b981',
  4: '#f59e0b',
  5: '#ef4444',
  6: '#6366f1',
  7: '#14b8a6',
}

// Fixed categorical order for cluster identity (Analytics page) — reuses the
// app's existing phase-color hues plus one more for the 8-cluster cap in
// unsupervised.py. Assigned by cluster index, never cycled/reordered.
export const CLUSTER_COLORS = [
  '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
  '#ef4444', '#6366f1', '#14b8a6', '#ec4899',
]
