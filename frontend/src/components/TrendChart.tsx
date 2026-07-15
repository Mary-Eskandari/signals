import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { api } from '../lib/api'
import { chartColors } from '../lib/theme'
import type { DailyTelemetry, PatientTrendSummary } from '../lib/types'

interface Props {
  patientId: string
}

export function TrendChart({ patientId }: Props) {
  const [daily, setDaily] = useState<DailyTelemetry[] | null>(null)
  const [trend, setTrend] = useState<PatientTrendSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setDaily(null)
    setTrend(null)
    Promise.all([api.getPatientDaily(patientId), api.getPatientTrend(patientId)])
      .then(([d, s]) => {
        setDaily(d)
        setTrend(s)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [patientId])

  if (loading) return <div className="panel-status">Loading telemetry…</div>
  if (error) return <div className="panel-status panel-error">{error}</div>
  if (!daily || !trend) return null

  const dates = daily.map((d) => d.date)
  const flaggedDates = new Set(daily.filter((d) => d.flags.length > 0).map((d) => d.date))
  const weightColor = chartColors.weight()
  const systolicColor = chartColors.systolicBp()
  const diastolicColor = chartColors.diastolicBp()
  const flaggedColor = chartColors.flagged()

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: [{ top: '14%', height: '32%' }, { top: '62%', height: '32%' }],
    tooltip: { trigger: 'axis' },
    legend: { data: ['Systolic BP', 'Diastolic BP'], top: '52%', textStyle: { fontSize: 11 } },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    xAxis: [0, 1].map((i) => ({
      type: 'category',
      data: dates,
      gridIndex: i,
      axisLabel: { show: i === 1, rotate: 45, fontSize: 10 },
    })),
    yAxis: [
      { type: 'value', gridIndex: 0, name: 'Weight (kg)', nameTextStyle: { color: weightColor } },
      { type: 'value', gridIndex: 1, name: 'BP (mmHg)' },
    ],
    series: [
      {
        name: 'Weight',
        type: 'line',
        data: daily.map((d) => d.weight_kg),
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: true,
        symbolSize: (_: unknown, params: { dataIndex: number }) =>
          flaggedDates.has(dates[params.dataIndex]) ? 8 : 3,
        itemStyle: {
          color: (params: { dataIndex: number }) =>
            flaggedDates.has(dates[params.dataIndex]) ? flaggedColor : weightColor,
        },
        lineStyle: { color: weightColor, width: 2 },
      },
      {
        name: 'Systolic BP',
        type: 'line',
        color: systolicColor,
        data: daily.map((d) => d.systolic_bp_mmhg),
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: systolicColor, width: 2 },
        itemStyle: { color: systolicColor },
      },
      {
        name: 'Diastolic BP',
        type: 'line',
        color: diastolicColor,
        data: daily.map((d) => d.diastolic_bp_mmhg),
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: diastolicColor, width: 2 },
        itemStyle: { color: diastolicColor },
      },
    ],
  }

  return (
    <div>
      <ReactECharts option={option} style={{ height: 420, width: '100%' }} notMerge />
      <div className="legend-note">
        <span className="dot" style={{ background: flaggedColor }} /> flagged day (e.g. rapid weight gain)
        <span style={{ marginLeft: 16 }}>
          weight slope: {trend.weight_slope_kg_per_day.toFixed(3)} kg/day · {trend.flagged_events.length} flagged events
        </span>
      </div>
    </div>
  )
}
