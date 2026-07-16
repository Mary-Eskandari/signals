import { ConfusionMatrixChart } from './ConfusionMatrixChart'
import { BEAT_FIELDS } from '../lib/glossary'
import type { ClassificationResult } from '../lib/types'

interface Props {
  result: ClassificationResult
}

export function ClassificationResults({ result }: Props) {
  const isCV = result.cv_results !== undefined

  return (
    <div>
      {result.feature_set === 'engineered' && result.feature_columns && (
        <div className="legend-note" style={{ marginBottom: 12 }}>
          Trained on {result.feature_columns.length} feature{result.feature_columns.length === 1 ? '' : 's'}:{' '}
          {result.feature_columns.map((f) => BEAT_FIELDS[f]?.label ?? f).join(', ')}
        </div>
      )}
      <div className="stat-grid">
        {isCV ? (
          <>
            <div className="stat-card" title="Mean accuracy across all folds, ± standard deviation">
              <div className="stat-label">Mean Accuracy</div>
              <div className="stat-value">
                {((result.mean_accuracy ?? 0) * 100).toFixed(1)}
                <span className="stat-unit">%</span>
              </div>
              <div className="stat-sub">± {((result.std_accuracy ?? 0) * 100).toFixed(1)}%</div>
            </div>
            <div className="stat-card" title="Mean macro-F1 across all folds — averages F1 equally across classes, not weighted by class size">
              <div className="stat-label">Mean Macro-F1</div>
              <div className="stat-value">
                {((result.mean_macro_f1 ?? 0) * 100).toFixed(1)}
                <span className="stat-unit">%</span>
              </div>
              <div className="stat-sub">± {((result.std_macro_f1 ?? 0) * 100).toFixed(1)}%</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">CV Folds</div>
              <div className="stat-value">{result.cv_folds}</div>
            </div>
          </>
        ) : (
          <>
            <div className="stat-card" title="Fraction of test beats correctly classified">
              <div className="stat-label">Accuracy</div>
              <div className="stat-value">
                {((result.accuracy ?? 0) * 100).toFixed(1)}
                <span className="stat-unit">%</span>
              </div>
            </div>
            <div className="stat-card" title="F1 score averaged equally across the 4 chamber classes, not weighted by how common each class is — a fairer summary than accuracy when classes are imbalanced">
              <div className="stat-label">Macro-F1</div>
              <div className="stat-value">
                {((result.macro_f1 ?? 0) * 100).toFixed(1)}
                <span className="stat-unit">%</span>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Train / Test Beats</div>
              <div className="stat-value">
                {result.n_train_beats}/{result.n_test_beats}
              </div>
            </div>
          </>
        )}
        <div className="stat-card">
          <div className="stat-label">Train Time</div>
          <div className="stat-value">
            {result.train_seconds.toFixed(1)}
            <span className="stat-unit">s</span>
          </div>
        </div>
      </div>

      {!isCV && result.confusion_matrix && result.labels && (
        <div style={{ marginTop: 20 }}>
          <h4 className="subsection-heading">Confusion Matrix</h4>
          <ConfusionMatrixChart matrix={result.confusion_matrix} labels={result.labels} />
        </div>
      )}

      {!isCV && result.per_class && (
        <div className="table-scroll" style={{ marginTop: 16, maxHeight: 'none' }}>
          <table className="feature-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Chamber</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {result.per_class.map((pc) => (
                <tr key={pc.label}>
                  <td style={{ textAlign: 'left' }}>{pc.label}</td>
                  <td>{pc.precision.toFixed(2)}</td>
                  <td>{pc.recall.toFixed(2)}</td>
                  <td>{pc.f1.toFixed(2)}</td>
                  <td>{pc.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {isCV && result.cv_results && (
        <div className="table-scroll" style={{ marginTop: 16, maxHeight: 'none' }}>
          <table className="feature-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Fold</th>
                <th>Accuracy</th>
                <th>Macro-F1</th>
                <th>Weighted-F1</th>
              </tr>
            </thead>
            <tbody>
              {result.cv_results.map((fold, i) => (
                <tr key={i}>
                  <td style={{ textAlign: 'left' }}>{i + 1}</td>
                  <td>{((fold.accuracy ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((fold.macro_f1 ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((fold.weighted_f1 ?? 0) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
