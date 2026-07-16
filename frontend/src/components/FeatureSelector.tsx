import { BEAT_FIELDS } from '../lib/glossary'

interface Props {
  allFeatures: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}

export function FeatureSelector({ allFeatures, selected, onChange }: Props) {
  if (allFeatures.length === 0) return null

  const toggle = (feature: string) => {
    onChange(selected.includes(feature) ? selected.filter((f) => f !== feature) : [...selected, feature])
  }

  return (
    <div>
      <div className="feature-selector-toolbar">
        <span className="legend-note">
          {selected.length}/{allFeatures.length} features selected
        </span>
        <button type="button" className="link-button" onClick={() => onChange(allFeatures)}>
          Select all
        </button>
        <button type="button" className="link-button" onClick={() => onChange([])}>
          Select none
        </button>
      </div>
      <div className="feature-selector-grid">
        {allFeatures.map((feature) => {
          const meta = BEAT_FIELDS[feature]
          return (
            <label key={feature} className="feature-checkbox" title={meta?.tooltip}>
              <input type="checkbox" checked={selected.includes(feature)} onChange={() => toggle(feature)} />
              <span>{meta?.label ?? feature}</span>
            </label>
          )
        })}
      </div>
    </div>
  )
}
