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

export function ConfusionMatrixChart({ matrix, labels }: Props) {
  const ramp = prefersDark() ? SEQUENTIAL_DARK : SEQUENTIAL_LIGHT

  // Row-normalized (each true-class row sums to 100%) — the standard confusion-matrix
  // convention (matches sklearn's normalize='true'), and it colors by how well the
  // model does on each class rather than by how common that class is in the data.
  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0))

  const data: { value: [number, number, number]; count: number }[] = []
  for (let trueIdx = 0; trueIdx < matrix.length; trueIdx++) {
    for (let predIdx = 0; predIdx < matrix[trueIdx].length; predIdx++) {
      const count = matrix[trueIdx][predIdx]
      const pct = rowSums[trueIdx] > 0 ? (count / rowSums[trueIdx]) * 100 : 0
      data.push({ value: [predIdx, trueIdx, pct], count })
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
          // Dark text on light (low-pct) cells, light text on dark (high-pct) cells —
          // a fixed white-with-border label washed out on the light end of the ramp.
          color: (p: { data: { value: [number, number, number] } }) =>
            p.data.value[2] > 55 ? '#fff' : '#111',
          formatter: (p: { data: { value: [number, number, number]; count: number } }) =>
            `${p.data.count} (${p.data.value[2].toFixed(0)}%)`,
        },
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.3)' } },
      },
    ],
  }

  return <ReactECharts option={option} style={{ height: 340, width: '100%' }} notMerge />
}
