import { useState } from 'react'
import { BeatFeaturesTable } from './BeatFeaturesTable'
import { DailyTelemetryTable } from './DailyTelemetryTable'
import { ProcedureSummaryPanel } from './ProcedureSummaryPanel'
import { ReportPanel } from './ReportPanel'
import { SelectorBar } from './SelectorBar'
import { TrendChart } from './TrendChart'
import { WaveformViewer } from './WaveformViewer'

export function DashboardView() {
  const [recordId, setRecordId] = useState('TRM278-RHC1')
  const [patientId, setPatientId] = useState('9')

  return (
    <>
      <SelectorBar
        recordId={recordId}
        patientId={patientId}
        onRecordChange={setRecordId}
        onPatientChange={setPatientId}
      />

      <section className="panel">
        <h2>Procedure Summary — {recordId}</h2>
        <ProcedureSummaryPanel recordId={recordId} />
      </section>

      <section className="panel">
        <h2>Waveform — PA Pressure / ECG / SCG</h2>
        <WaveformViewer recordId={recordId} />
        <details className="feature-details">
          <summary>Per-beat extracted features ({recordId})</summary>
          <BeatFeaturesTable recordId={recordId} />
        </details>
      </section>

      <section className="panel">
        <h2>Telemonitoring Trend — Patient {patientId}</h2>
        <TrendChart patientId={patientId} />
        <details className="feature-details">
          <summary>Daily telemetry features (Patient {patientId})</summary>
          <DailyTelemetryTable patientId={patientId} />
        </details>
      </section>

      <section className="panel">
        <h2>LLM Clinical Summary</h2>
        <ReportPanel recordId={recordId} patientId={patientId} />
      </section>
    </>
  )
}
