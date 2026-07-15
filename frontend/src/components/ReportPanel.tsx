import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { ClinicalReport } from '../lib/types'

interface Props {
  recordId: string | null
  patientId: string | null
}

function Section({ section }: { section: { title: string; findings: string[] } | null | undefined }) {
  if (!section) return null
  return (
    <div className="report-section">
      <h4>{section.title}</h4>
      <ul>
        {section.findings.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ul>
    </div>
  )
}

export function ReportPanel({ recordId, patientId }: Props) {
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState<string>('claude-sonnet-5')
  const [report, setReport] = useState<ClinicalReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listReportModels().then((m) => {
      setModels(m.allowed_models)
      setModel(m.default)
    })
  }, [])

  useEffect(() => {
    setReport(null)
    setError(null)
  }, [recordId, patientId])

  const generate = () => {
    setLoading(true)
    setError(null)
    api
      .generateReport(recordId, patientId, model)
      .then(setReport)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }

  return (
    <div>
      <div className="report-controls">
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <button onClick={generate} disabled={loading || (!recordId && !patientId)}>
          {loading ? 'Generating…' : 'Generate Clinician Summary'}
        </button>
      </div>

      {error && <div className="panel-status panel-error">{error}</div>}

      {report && (
        <div className="report">
          <p className="report-summary">{report.summary}</p>
          <Section section={report.pa_pressure_findings} />
          <Section section={report.rhythm_hrv_findings} />
          <Section section={report.scg_findings} />
          <Section section={report.trend_findings} />
          {report.flags.length > 0 && (
            <div className="report-flags">
              {report.flags.map((f, i) => (
                <div key={i} className="flag-pill">
                  {f}
                </div>
              ))}
            </div>
          )}
          <p className="disclaimer">{report.disclaimer}</p>
        </div>
      )}
    </div>
  )
}
