import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { ProcedureSummary } from '../lib/types'
import { SummaryCards } from './SummaryCards'

interface Props {
  recordId: string
}

export function ProcedureSummaryPanel({ recordId }: Props) {
  const [summary, setSummary] = useState<ProcedureSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setSummary(null)
    api
      .getRecordSummary(recordId)
      .then(setSummary)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [recordId])

  if (loading) return <div className="panel-status">Processing record (first view can take a bit)…</div>
  if (error) return <div className="panel-status panel-error">{error}</div>
  if (!summary) return null

  return <SummaryCards summary={summary} />
}
