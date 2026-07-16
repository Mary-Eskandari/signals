import ReactECharts from 'echarts-for-react'
import { prefersDark } from '../lib/theme'

interface Props {
  matrix: number[][]
  labels: string[]
}

// Sequential blue ramp (dataviz skill reference palette) — magnitude encoding,
// light-to-dark, single hue.
const SEQUENTIAL_LIGHT = ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#184f95']
const SEQUENTIAL_DARK = ['#0d3661', '#104281', '#1c5cab', '#2a78d6', '#3987e5']

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

// Mirrors ECharts visualMap's own piecewise-linear interpolation across the ramp
// stops, so the label color always matches what's actually painted in the cell.
function interpolateRamp(ramp: string[], t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t))
  const scaled = clamped * (ramp.length - 1)
  const i = Math.floor(scaled)
  const frac = scaled - i
  const [r0, g0, b0] = hexToRgb(ramp[i])
  const [r1, g1, b1] = hexToRgb(ramp[Math.min(i + 1, ramp.length - 1)])
  return [r0 + (r1 - r0) * frac, g0 + (g1 - g0) * frac, b0 + (b1 - b0) * frac]
}

function relativeLuminance([r, g, b]: [number, number, number]): number {
  const [sr, sg, sb] = [r, g, b].map((v) => {
    const c = v / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * sr + 0.7152 * sg + 0.0722 * sb
}

// WCAG contrast ratio against pure black/white — picks whichever text color
// actually reads better on this specific cell, rather than assuming a fixed
// percentage threshold matches a given ramp's light/dark direction (that broke
// in dark mode, where the "low" end of the ramp is already a dark navy).
function readableTextColor(rgb: [number, number, number]): string {
  const bgLum = relativeLuminance(rgb)
  const contrastWithBlack = (bgLum + 0.05) / 0.05
  const contrastWithWhite = 1.05 / (bgLum + 0.05)
  return contrastWithBlack >= contrastWithWhite ? '#111' : '#fff'
}

export function ConfusionMatrixChart({ matrix, labels }: Props) {
  const ramp = prefersDark() ? SEQUENTIAL_DARK : SEQUENTIAL_LIGHT

  // Row-normalized (each true-class row sums to 100%) — the standard confusion-matrix
  // convention (matches sklearn's normalize='true'), and it colors by how well the
  // model does on each class rather than by how common that class is in the data.
  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0))

  const data: { value: [number, number, number]; count: number; label: { color: string } }[] = []
  for (let trueIdx = 0; trueIdx < matrix.length; trueIdx++) {
    for (let predIdx = 0; predIdx < matrix[trueIdx].length; predIdx++) {
      const count = matrix[trueIdx][predIdx]
      const pct = rowSums[trueIdx] > 0 ? (count / rowSums[trueIdx]) * 100 : 0
      // Computed per-point rather than via a series-level label.color callback —
      // that callback silently didn't apply for this heatmap series (verified by
      // sampling the rendered canvas: labels came out pure black regardless of
      // the function's return value), so each data item carries its own label
      // style instead, which is the well-supported per-item override pattern.
      const textColor = readableTextColor(interpolateRamp(ramp, pct / 100))
      data.push({ value: [predIdx, trueIdx, pct], count, label: { color: textColor } })
    }
  }

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p: { data: { value: [number, number, number]; count: number } }) => {
        const [predIdx, trueIdx, pct] = p.data.value
        return `True: ${labels[trueIdx]}<br/>Predicted: ${labels[predIdx]}<br/>Count: ${p.data.count} (${pct.toFixed(1)}% of true ${labels[trueIdx]})`
      },
    },
    grid: { top: 40, left: 70, right: 20, bottom: 50 },
    xAxis: {
      type: 'category',
      data: labels,
      name: 'Predicted',
      nameLocation: 'middle',
      nameGap: 30,
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category',
      data: labels,
      name: 'True (each row sums to 100%)',
      nameLocation: 'middle',
      nameGap: 40,
      splitArea: { show: true },
    },
    visualMap: {
      min: 0,
      max: 100,
      show: false,
      inRange: { color: ramp },
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: {
          show: true,
          fontSize: 12,
          fontWeight: 600,
          formatter: (p: { data: { value: [number, number, number]; count: number } }) =>
            `${p.data.count} (${p.data.value[2].toFixed(0)}%)`,
        },
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.3)' } },
      },
    ],
  }

  return <ReactECharts option={option} style={{ height: 340, width: '100%' }} notMerge />
}
