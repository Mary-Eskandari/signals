import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { api } from '../lib/api'
import { chartColors } from '../lib/theme'
import type { BeatFeatures, WaveformResponse } from '../lib/types'

interface Props {
  recordId: string
}

const GRID_LAYOUT = [
  { top: '5%', height: '26%' },
  { top: '38%', height: '26%' },
  { top: '71%', height: '26%' },
]

function nearestIndex(times: number[], target: number): number {
  let lo = 0
  let hi = times.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] < target) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function WaveformViewer({ recordId }: Props) {
  const [waveform, setWaveform] = useState<WaveformResponse | null>(null)
  const [beats, setBeats] = useState<BeatFeatures[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setWaveform(null)
    setBeats(null)
    Promise.all([api.getRecordWaveform(recordId), api.getRecordBeats(recordId)])
      .then(([wf, b]) => {
        setWaveform(wf)
        setBeats(b)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [recordId])

  if (loading) return <div className="panel-status">Loading waveform…</div>
  if (error) return <div className="panel-status panel-error">{error}</div>
  if (!waveform || !beats) return null

  const t = waveform.time_s
  const pap = waveform.channels['RHC_pressure']
  const ecg = waveform.channels['ECG_lead_II']
  const scg = waveform.channels['patch_ACC_dv']

  const onsetPoints = beats
    .filter((b) => b.onset_time_s >= t[0] && b.onset_time_s <= t[t.length - 1])
    .map((b) => {
      const idx = nearestIndex(t, b.onset_time_s)
      return { coord: [t[idx], pap[idx]], value: 'onset' }
    })

  const aoPoints = beats
    .filter((b) => b.scg_ao_time_s !== null && b.scg_ao_time_s >= t[0] && b.scg_ao_time_s <= t[t.length - 1])
    .map((b) => {
      const idx = nearestIndex(t, b.scg_ao_time_s as number)
      return { coord: [t[idx], scg[idx]], value: 'AO', symbol: 'circle' }
    })

  const acPoints = beats
    .filter((b) => b.scg_ac_time_s !== null && b.scg_ac_time_s >= t[0] && b.scg_ac_time_s <= t[t.length - 1])
    .map((b) => {
      const idx = nearestIndex(t, b.scg_ac_time_s as number)
      return { coord: [t[idx], scg[idx]], value: 'AC', symbol: 'diamond' }
    })

  const papColor = chartColors.pap()
  const ecgColor = chartColors.ecg()
  const scgColor = chartColors.scg()
  const annotationColor = chartColors.annotation()

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: GRID_LAYOUT,
    tooltip: { trigger: 'axis' },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    xAxis: [0, 1, 2].map((i) => ({
      type: 'value',
      scale: true, // fit tightly to the actual data range instead of forcing origin=0
      gridIndex: i,
      name: i === 2 ? 'time (s)' : undefined,
      axisLabel: { show: i === 2 },
      axisLine: { lineStyle: { color: '#898781' } },
    })),
    yAxis: [
      { type: 'value', scale: true, gridIndex: 0, name: 'PAP (mmHg)', nameTextStyle: { color: papColor } },
      { type: 'value', scale: true, gridIndex: 1, name: 'ECG (mV)', nameTextStyle: { color: ecgColor } },
      { type: 'value', scale: true, gridIndex: 2, name: 'SCG (mg)', nameTextStyle: { color: scgColor } },
    ],
    series: [
      {
        name: 'PAP',
        type: 'line',
        data: t.map((time, i) => [time, pap[i]]),
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { color: papColor, width: 2 },
        markPoint: {
          symbolSize: 8,
          data: onsetPoints,
          itemStyle: { color: annotationColor, borderColor: '#fff', borderWidth: 1 },
          label: { show: false },
        },
      },
      {
        name: 'ECG',
        type: 'line',
        data: t.map((time, i) => [time, ecg[i]]),
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: ecgColor, width: 2 },
      },
      {
        name: 'SCG',
        type: 'line',
        data: t.map((time, i) => [time, scg[i]]),
        xAxisIndex: 2,
        yAxisIndex: 2,
        showSymbol: false,
        lineStyle: { color: scgColor, width: 2 },
        markPoint: {
          symbolSize: 7,
          data: [...aoPoints, ...acPoints],
          itemStyle: { color: annotationColor, borderColor: '#fff', borderWidth: 1 },
          label: { show: false },
        },
      },
    ],
  }

  return (
    <div>
      <ReactECharts option={option} style={{ height: 480, width: '100%' }} notMerge />
      <div className="legend-note">
        <span className="dot" style={{ background: annotationColor }} /> ● PAP pulse onset
        <span className="dot" style={{ background: annotationColor, marginLeft: 16 }} /> ◆ SCG AO (circle) / AC (diamond)
      </div>
    </div>
  )
}
