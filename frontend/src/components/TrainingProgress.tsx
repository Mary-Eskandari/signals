import type { TrainProgressEvent } from '../lib/types'

interface Props {
  latestEvent: TrainProgressEvent | null
}

export function TrainingProgress({ latestEvent }: Props) {
  if (!latestEvent) return <div className="panel-status">Starting…</div>

  if (latestEvent.type === 'fitting') {
    return <div className="panel-status">Fitting {latestEvent.model}…</div>
  }

  if (latestEvent.type === 'epoch') {
    const pct = ((latestEvent.epoch ?? 0) / (latestEvent.total_epochs ?? 1)) * 100
    return (
      <div className="training-progress">
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="progress-label">
          Epoch {latestEvent.epoch}/{latestEvent.total_epochs} · loss {latestEvent.loss?.toFixed(4)} · train accuracy{' '}
          {((latestEvent.train_accuracy ?? 0) * 100).toFixed(1)}%
        </div>
      </div>
    )
  }

  if (latestEvent.type === 'fold') {
    const pct = ((latestEvent.fold ?? 0) / (latestEvent.total_folds ?? 1)) * 100
    return (
      <div className="training-progress">
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="progress-label">
          Fold {latestEvent.fold}/{latestEvent.total_folds} complete · accuracy {((latestEvent.accuracy ?? 0) * 100).toFixed(1)}% ·
          macro-F1 {((latestEvent.macro_f1 ?? 0) * 100).toFixed(1)}%
        </div>
      </div>
    )
  }

  return <div className="panel-status">Working…</div>
}
