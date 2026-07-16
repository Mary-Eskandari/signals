import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type {
  ClassificationModelsResponse,
  ClassificationResult,
  ClassificationStatus,
  ModelTier,
  TrainProgressEvent,
} from '../lib/types'
import { ClassificationResults } from './ClassificationResults'
import { FeatureSelector } from './FeatureSelector'
import { HyperparameterControls } from './HyperparameterControls'
import { TrainingProgress } from './TrainingProgress'

const TIER_LABELS: Record<ModelTier, string> = {
  classic: 'Classic',
  ensemble: 'Ensemble',
  neural: 'Neural / Deep Learning',
}

export function ClassificationView() {
  const [modelsData, setModelsData] = useState<ClassificationModelsResponse | null>(null)
  const [status, setStatus] = useState<ClassificationStatus | null>(null)
  const [records, setRecords] = useState<string[]>([])
  const [allFeatures, setAllFeatures] = useState<string[]>([])
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])

  const [selectedModel, setSelectedModel] = useState('random_forest')
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({})
  const [splitMode, setSplitMode] = useState<'auto' | 'manual'>('auto')
  const [testSize, setTestSize] = useState(0.3)
  const [manualTrainIds, setManualTrainIds] = useState<string[]>([])
  const [manualTestIds, setManualTestIds] = useState<string[]>([])
  const [useCV, setUseCV] = useState(false)
  const [cvFolds, setCvFolds] = useState(5)

  const [training, setTraining] = useState(false)
  const [latestEvent, setLatestEvent] = useState<TrainProgressEvent | null>(null)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.classificationModels().then(setModelsData)
    api.classificationRecords().then(setRecords)
    api.classificationFeatures().then((features) => {
      setAllFeatures(features)
      setSelectedFeatures(features) // default to using every engineered feature
    })
    const poll = () => api.classificationStatus().then(setStatus)
    poll()
    const interval = setInterval(poll, 15000) // dataset build may still be running server-side
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!modelsData) return
    const defaults: Record<string, number> = {}
    for (const spec of modelsData.hyperparameters[selectedModel] ?? []) {
      defaults[spec.name] = spec.default
    }
    setHyperparams(defaults)
  }, [selectedModel, modelsData])

  const modelsByTier = useMemo(() => {
    if (!modelsData) return {}
    const grouped: Record<string, [string, string][]> = {}
    for (const [key, meta] of Object.entries(modelsData.models)) {
      grouped[meta.tier] ??= []
      grouped[meta.tier].push([key, meta.display_name])
    }
    return grouped
  }, [modelsData])

  // Raw-waveform models (cnn/lstm) always use the fixed raw signal channels —
  // feature selection only applies to engineered-feature models.
  const usesEngineeredFeatures = modelsData?.models[selectedModel]?.feature_set === 'engineered'

  const train = () => {
    setTraining(true)
    setError(null)
    setResult(null)
    setLatestEvent(null)

    api
      .trainStream(
        {
          model: selectedModel,
          split:
            splitMode === 'manual'
              ? { mode: 'manual', train_record_ids: manualTrainIds, test_record_ids: manualTestIds }
              : { mode: 'auto', test_size: testSize },
          cv_folds: useCV ? cvFolds : null,
          hyperparameters: hyperparams,
          feature_columns: usesEngineeredFeatures ? selectedFeatures : null,
        },
        (event) => {
          if (event.type === 'result' && event.data) {
            setResult(event.data)
          } else if (event.type === 'error') {
            setError(event.message ?? 'training failed')
          } else {
            setLatestEvent(event)
          }
        }
      )
      .catch((e) => setError(String(e)))
      .finally(() => setTraining(false))
  }

  const canTrain =
    !training &&
    (splitMode === 'auto' || (manualTrainIds.length > 0 && manualTestIds.length > 0)) &&
    (!usesEngineeredFeatures || selectedFeatures.length > 0)

  return (
    <div>
      <section className="panel">
        <h2>Dataset — Chamber Position Labels</h2>
        {status ? (
          <div className="legend-note">
            {status.n_records_covered}/{status.total_records} records processed · {status.n_beats.toLocaleString()} labeled
            beats
            {status.n_records_covered < status.total_records && ' (build still running server-side, refreshing every 15s)'}
            {' · '}
            {Object.entries(status.per_chamber)
              .map(([chamber, count]) => `${chamber}: ${count}`)
              .join(' · ')}
          </div>
        ) : (
          <div className="panel-status">Loading dataset status…</div>
        )}
      </section>

      <section className="panel">
        <h2>Configure Model</h2>

        <div className="classification-controls">
          <label className="hyperparam-field">
            <span>Model</span>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {Object.entries(modelsByTier).map(([tier, models]) => (
                <optgroup key={tier} label={TIER_LABELS[tier as ModelTier] ?? tier}>
                  {models.map(([key, displayName]) => (
                    <option key={key} value={key}>
                      {displayName}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>

          <label className="hyperparam-field">
            <span>Split mode</span>
            <select value={splitMode} onChange={(e) => setSplitMode(e.target.value as 'auto' | 'manual')}>
              <option value="auto">Auto (random, grouped by record)</option>
              <option value="manual">Manual (choose records)</option>
            </select>
          </label>

          {splitMode === 'auto' && !useCV && (
            <label className="hyperparam-field">
              <span>Test size</span>
              <input
                type="number"
                min={0.1}
                max={0.9}
                step={0.05}
                value={testSize}
                onChange={(e) => setTestSize(parseFloat(e.target.value))}
              />
            </label>
          )}

          <label className="hyperparam-field">
            <span>
              <input type="checkbox" checked={useCV} onChange={(e) => setUseCV(e.target.checked)} /> Use k-fold CV
            </span>
            {useCV && (
              <input type="number" min={2} max={10} value={cvFolds} onChange={(e) => setCvFolds(parseInt(e.target.value, 10))} />
            )}
          </label>
        </div>

        {splitMode === 'manual' && !useCV && (
          <div className="manual-split">
            <label>
              Train records ({manualTrainIds.length})
              <select
                multiple
                size={8}
                value={manualTrainIds}
                onChange={(e) => setManualTrainIds(Array.from(e.target.selectedOptions, (o) => o.value))}
              >
                {records.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Test records ({manualTestIds.length})
              <select
                multiple
                size={8}
                value={manualTestIds}
                onChange={(e) => setManualTestIds(Array.from(e.target.selectedOptions, (o) => o.value))}
              >
                {records.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}

        {usesEngineeredFeatures ? (
          <>
            <h4 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.4, color: 'var(--text-muted)', marginTop: 16 }}>
              Features
            </h4>
            <FeatureSelector allFeatures={allFeatures} selected={selectedFeatures} onChange={setSelectedFeatures} />
          </>
        ) : (
          <div className="legend-note" style={{ marginTop: 16 }}>
            This model trains on raw PA-pressure/ECG/SCG waveform snippets, not the engineered features below — feature
            selection doesn't apply here.
          </div>
        )}

        <h4 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.4, color: 'var(--text-muted)', marginTop: 16 }}>
          Hyperparameters
        </h4>
        <HyperparameterControls
          specs={modelsData?.hyperparameters[selectedModel] ?? []}
          values={hyperparams}
          onChange={(name, value) => setHyperparams((prev) => ({ ...prev, [name]: value }))}
        />

        <div className="report-controls" style={{ marginTop: 16 }}>
          <button onClick={train} disabled={!canTrain}>
            {training ? 'Training…' : 'Train & Evaluate'}
          </button>
        </div>
      </section>

      {training && (
        <section className="panel">
          <h2>Training Progress</h2>
          <TrainingProgress latestEvent={latestEvent} />
        </section>
      )}

      {error && (
        <section className="panel">
          <div className="panel-status panel-error">{error}</div>
        </section>
      )}

      {result && (
        <section className="panel">
          <h2>
            Results — {modelsData?.models[result.model]?.display_name ?? result.model}
          </h2>
          <ClassificationResults result={result} />
        </section>
      )}
    </div>
  )
}
