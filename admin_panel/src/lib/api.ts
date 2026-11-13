import type {
  DashboardResponse,
  KnowledgeDocument,
  KnowledgeSearchResult,
  ProviderSettingsResponse,
  RuntimeProviderConfig,
  VoiceProfile,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const ADMIN_API_KEY_STORAGE = 'omnivox-admin-api-key';

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export function getAdminApiKey(): string | null {
  return sessionStorage.getItem(ADMIN_API_KEY_STORAGE);
}

export function setAdminApiKey(value: string): void {
  sessionStorage.setItem(ADMIN_API_KEY_STORAGE, value);
}

export function clearAdminApiKey(): void {
  sessionStorage.removeItem(ADMIN_API_KEY_STORAGE);
}

async function fetchAdminApi(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const apiKey = getAdminApiKey();
  if (apiKey) headers.set('Authorization', `Bearer ${apiKey}`);

  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (response.status === 401) {
    clearAdminApiKey();
    window.dispatchEvent(new Event('omnivox-admin-auth-required'));
  }
  return response;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetchAdminApi(path, init);

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const message = isDetailResponse(payload) ? payload.detail : `Request failed with status ${response.status}.`;
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

function isDetailResponse(value: unknown): value is { detail: string } {
  return typeof value === 'object' && value !== null && 'detail' in value && typeof value.detail === 'string';
}

/** Typed admin API client used by TanStack Query hooks. */
export const adminApi = {
  getDashboard: (): Promise<DashboardResponse> => request('/api/v1/admin/dashboard'),
  getProviderSettings: (): Promise<ProviderSettingsResponse> => request('/api/v1/admin/providers'),
  updateProviderSettings: (payload: RuntimeProviderConfig): Promise<ProviderSettingsResponse> =>
    request('/api/v1/admin/providers', { method: 'PUT', body: JSON.stringify(payload) }),
  getVoices: (): Promise<VoiceProfile[]> => request('/api/v1/admin/voices'),
  getKnowledgeDocuments: (): Promise<KnowledgeDocument[]> => request('/api/v1/admin/knowledge/documents'),
  uploadKnowledgeDocument: (file: File): Promise<KnowledgeDocument> => {
    const body = new FormData();
    body.append('document', file);
    body.append('tenant_id', 'grand-meridian');
    return request('/api/v1/admin/knowledge/upload', { method: 'POST', body });
  },
  searchKnowledge: (query: string): Promise<KnowledgeSearchResult> =>
    request('/api/v1/admin/knowledge/search', {
      method: 'POST',
      body: JSON.stringify({ query, tenant_id: 'grand-meridian' }),
    }),
  auditionVoice: async (payload: { voice_id: string; text: string; speed: number; pitch: number }): Promise<Blob> => {
    const response = await fetchAdminApi('/api/v1/admin/voice/audition', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorPayload: unknown = await response.json().catch(() => null);
      throw new ApiError(response.status, isDetailResponse(errorPayload) ? errorPayload.detail : 'Voice audition failed.');
    }
    return response.blob();
  },
};
