/**
 * Centralized Frontend Environment & Connection Configuration
 * Reads VITE_API_URL and VITE_WS_URL from environment with standard local fallbacks.
 */

export const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
export const WS_BASE_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000').replace(/\/$/, '');
