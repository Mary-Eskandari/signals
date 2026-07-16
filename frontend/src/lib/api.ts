import type {
  BeatFeatures,
  ChamberEvents,
  ClassificationModelsResponse,
  ClassificationStatus,
  ClinicalReport,
  DailyTelemetry,
  ModelsResponse,
  PatientListItem,
  PatientTrendSummary,
  ProcedureSummary,
  RecordListItem,
  TrainProgressEvent,
  TrainRequest,
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

  classificationStatus: () => get<ClassificationStatus>('/classification/status'),
  classificationModels: () => get<ClassificationModelsResponse>('/classification/models'),
  classificationLabels: () => get<string[]>('/classification/labels'),
  classificationRecords: () => get<string[]>('/classification/records'),
  classificationFeatures: () => get<string[]>('/classification/features'),

  /** Consumes the newline-delimited-JSON training stream, calling onEvent for each
   * progress line (epoch/fold/fitting) and the final result line. */
  trainStream: async (req: TrainRequest, onEvent: (e: TrainProgressEvent) => void): Promise<void> => {
    const response = await fetch(`${API_BASE}/classification/train/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`train/stream failed (${response.status}): ${text}`)
    }
    if (!response.body) {
      throw new Error('train/stream failed: response had no body')
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line.trim()) onEvent(JSON.parse(line) as TrainProgressEvent)
      }
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer) as TrainProgressEvent)
  },
}
