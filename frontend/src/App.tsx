import { useState } from 'react'
import './App.css'
import { BeatFeaturesTable } from './components/BeatFeaturesTable'
import { DailyTelemetryTable } from './components/DailyTelemetryTable'
import { ProcedureSummaryPanel } from './components/ProcedureSummaryPanel'
import { ReportPanel } from './components/ReportPanel'
import { SelectorBar } from './components/SelectorBar'
import { TrendChart } from './components/TrendChart'
import { WaveformViewer } from './components/WaveformViewer'

function App() {
  const [recordId, setRecordId] = useState('TRM278-RHC1')
  const [patientId, setPatientId] = useState('9')

  return (
    <div className="app">
      <header className="app-header">
        <h1>Cordella-Adjacent HF Signal Insights</h1>
        <p className="tagline">
          Pulmonary artery pressure, ECG, and seismocardiogram signal processing + LLM clinical reporting —
          a technical exploration of the problem space behind implantable PA pressure sensor systems
          (e.g. Cordella, CardioMEMS) used in heart failure management.
        </p>
      </header>

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

      <footer className="app-footer">
        <p>
          Datasets: <a href="https://physionet.org/content/scg-rhc-wearable-database/1.0.0/" target="_blank" rel="noreferrer">SCG-RHC Wearable + Right Heart Catheter DB</a> (PhysioNet, ODC-BY) ·{' '}
          <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190323" target="_blank" rel="noreferrer">Chiron CHF Telemonitoring</a> (Mlakar et al. 2018, CC BY)
        </p>
      </footer>
    </div>
  )
}

export default App
