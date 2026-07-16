import { BEAT_FIELDS, SOTA_BEAT_FIELDS } from '../lib/glossary'

interface Props {
  allFeatures: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}

interface Category {
  key: string
  label: string
  description: string
}

const CATEGORIES: Category[] = [
  { key: 'basic', label: 'Basic', description: 'Core pressure, timing, and quality features from standard PA-catheter/ECG/SCG landmark detection.' },
  { key: 'sota', label: 'SOTA', description: 'Pulse-wave-analysis and SCG-ensemble features (dicrotic notch, upstroke slope, beat shape, detection confidence) layered on top of the basic set.' },
]

function categoryOf(feature: string): string {
  return SOTA_BEAT_FIELDS.has(feature) ? 'sota' : 'basic'
}

export function FeatureSelector({ allFeatures, selected, onChange }: Props) {
  if (allFeatures.length === 0) return null

  const toggle = (feature: string) => {
    onChange(selected.includes(feature) ? selected.filter((f) => f !== feature) : [...selected, feature])
  }

  const grouped = CATEGORIES.map((cat) => ({
    ...cat,
    features: allFeatures.filter((f) => categoryOf(f) === cat.key),
  })).filter((cat) => cat.features.length > 0)

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

      {grouped.map((cat) => {
        const selectedInCat = cat.features.filter((f) => selected.includes(f)).length
        return (
          <details key={cat.key} className="feature-category" open>
            <summary title={cat.description}>
              {cat.label} ({selectedInCat}/{cat.features.length})
            </summary>
            <div className="feature-selector-grid">
              {cat.features.map((feature) => {
                const meta = BEAT_FIELDS[feature]
                return (
                  <label key={feature} className="feature-checkbox" title={meta?.tooltip}>
                    <input type="checkbox" checked={selected.includes(feature)} onChange={() => toggle(feature)} />
                    <span>{meta?.label ?? feature}</span>
                  </label>
                )
              })}
            </div>
          </details>
        )
      })}
    </div>
  )
}
