import { useState } from 'react'
import './App.css'
import { ClassificationView } from './components/ClassificationView'
import { DashboardView } from './components/DashboardView'

type Tab = 'dashboard' | 'classification'

function App() {
  const [tab, setTab] = useState<Tab>('dashboard')

  return (
    <div className="app">
      <header className="app-header">
        <h1>Cordella-Adjacent HF Signal Insights</h1>
        <p className="tagline">
          Pulmonary artery pressure, ECG, and seismocardiogram signal processing + LLM clinical reporting + chamber-
          position classification — a technical exploration of the problem space behind implantable PA pressure
          sensor systems (e.g. Cordella, CardioMEMS) used in heart failure management.
        </p>
      </header>

      <nav className="tab-bar">
        <button className={tab === 'dashboard' ? 'tab-active' : ''} onClick={() => setTab('dashboard')}>
          Dashboard
        </button>
        <button className={tab === 'classification' ? 'tab-active' : ''} onClick={() => setTab('classification')}>
          Classification
        </button>
      </nav>

      {tab === 'dashboard' ? <DashboardView /> : <ClassificationView />}

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
