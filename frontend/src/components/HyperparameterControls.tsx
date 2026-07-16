import type { HyperparamSpec } from '../lib/types'

interface Props {
  specs: HyperparamSpec[]
  values: Record<string, number>
  onChange: (name: string, value: number) => void
}

export function HyperparameterControls({ specs, values, onChange }: Props) {
  if (specs.length === 0) return null

  return (
    <div className="hyperparam-grid">
      {specs.map((spec) => (
        <label key={spec.name} className="hyperparam-field" title={spec.description}>
          <span>{spec.name}</span>
          <input
            type="number"
            step={spec.type === 'float' ? 'any' : 1}
            min={spec.min}
            max={spec.max}
            value={values[spec.name] ?? spec.default}
            onChange={(e) => onChange(spec.name, spec.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
          />
        </label>
      ))}
    </div>
  )
}
