import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { PatientListItem, RecordListItem } from '../lib/types'

interface Props {
  recordId: string
  patientId: string
  onRecordChange: (id: string) => void
  onPatientChange: (id: string) => void
}

// A few records/patients known to have plenty of readings — good defaults for a fast demo.
const FEATURED_RECORDS = ['TRM278-RHC1', 'TRM178-RHC1', 'TRM220-RHC1', 'TRM155-RHC1']
const FEATURED_PATIENTS = ['9', '10', '2', '6']

export function SelectorBar({ recordId, patientId, onRecordChange, onPatientChange }: Props) {
  const [records, setRecords] = useState<RecordListItem[]>([])
  const [patients, setPatients] = useState<PatientListItem[]>([])

  useEffect(() => {
    api.listRecords().then(setRecords)
    api.listPatients().then(setPatients)
  }, [])

  const orderedRecords = [
    ...FEATURED_RECORDS.filter((id) => records.some((r) => r.record_id === id)),
    ...records.map((r) => r.record_id).filter((id) => !FEATURED_RECORDS.includes(id)),
  ]
  const orderedPatients = [
    ...FEATURED_PATIENTS.filter((id) => patients.some((p) => p.patient_id === id)),
    ...patients.map((p) => p.patient_id).filter((id) => !FEATURED_PATIENTS.includes(id)),
  ]

  return (
    <div className="selector-bar">
      <label>
        Catheterization record
        <select value={recordId} onChange={(e) => onRecordChange(e.target.value)}>
          {orderedRecords.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Telemonitoring patient
        <select value={patientId} onChange={(e) => onPatientChange(e.target.value)}>
          {orderedPatients.map((id) => (
            <option key={id} value={id}>
              Patient {id}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}
