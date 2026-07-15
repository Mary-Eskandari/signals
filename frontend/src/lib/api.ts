import type {
  BeatFeatures,
  ChamberEvents,
  ClinicalReport,
  DailyTelemetry,
  ModelsResponse,
  PatientListItem,
  PatientTrendSummary,
  ProcedureSummary,
  RecordListItem,
  WaveformResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`GET ${path} failed (${response.status}): ${body}`)
  }
  return response.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`POST ${path} failed (${response.status}): ${text}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  listRecords: () => get<RecordListItem[]>('/records'),
  getRecordSummary: (recordId: string) => get<ProcedureSummary>(`/records/${recordId}/summary`),
  getRecordBeats: (recordId: string) => get<BeatFeatures[]>(`/records/${recordId}/beats`),
  getRecordWaveform: (recordId: string, maxPoints = 3000) =>
    get<WaveformResponse>(`/records/${recordId}/waveform?max_points=${maxPoints}`),
  getChamberEvents: (recordId: string) => get<ChamberEvents>(`/records/${recordId}/chamber_events`),

  listPatients: () => get<PatientListItem[]>('/patients'),
  getPatientTrend: (patientId: string) => get<PatientTrendSummary>(`/patients/${patientId}/trend`),
  getPatientDaily: (patientId: string) => get<DailyTelemetry[]>(`/patients/${patientId}/daily`),

  listReportModels: () => get<ModelsResponse>('/reports/models'),
  generateReport: (recordId: string | null, patientId: string | null, model?: string) =>
    post<ClinicalReport>('/reports/generate', { record_id: recordId, patient_id: patientId, model }),
}
