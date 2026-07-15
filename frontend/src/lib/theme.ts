// Validated palette (dataviz skill reference instance) — light/dark categorical
// steps + status colors. Chart series read these directly since ECharts needs
// JS color values, not CSS custom properties; chrome (surfaces/ink) uses the
// CSS variables in index.css instead.

export function prefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches
}

const categorical = {
  blue: { light: '#2a78d6', dark: '#3987e5' },
  aqua: { light: '#1baf7a', dark: '#199e70' },
  yellow: { light: '#eda100', dark: '#c98500' },
  green: { light: '#008300', dark: '#008300' },
  violet: { light: '#4a3aa7', dark: '#9085e9' },
  red: { light: '#e34948', dark: '#e66767' },
  magenta: { light: '#e87ba4', dark: '#d55181' },
  orange: { light: '#eb6834', dark: '#d95926' },
}

const status = {
  critical: { light: '#d03b3b', dark: '#d03b3b' },
}

function pick(slot: { light: string; dark: string }): string {
  return prefersDark() ? slot.dark : slot.light
}

export const chartColors = {
  pap: () => pick(categorical.blue),
  ecg: () => pick(categorical.green),
  scg: () => pick(categorical.violet),
  annotation: () => pick(status.critical),
  weight: () => pick(categorical.blue),
  systolicBp: () => pick(categorical.blue),
  diastolicBp: () => pick(categorical.green),
  flagged: () => pick(status.critical),
}
