import { Button, ConfirmDialog, Input } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { CanonicalGroupRoster, type CanonicalMemberPhase } from './canonical-group-activity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalRoom, CanonicalRoomContext } from './canonical-groups'

interface CanonicalGroupHeaderProps {
  busy: boolean
  canStop: boolean
  context?: CanonicalRoomContext
  deliveryTarget: string
  onBack?: () => void
  onDisband: () => Promise<void>
  onRename: (name: string) => void
  onSaveContext: (context: CanonicalRoomContext) => void
  onStop: () => void
  phases: Map<string, CanonicalMemberPhase>
  room?: CanonicalRoom
}

export function CanonicalGroupHeader({ busy, canStop, context, deliveryTarget, onBack, onDisband, onRename, onSaveContext, onStop, phases, room }: CanonicalGroupHeaderProps) {
  const labels = useCanonicalGroupLabels()
  const [renaming, setRenaming] = useState<string | null>(null)
  const [settings, setSettings] = useState<CanonicalRoomContext | null>(null)
  const [ending, setEnding] = useState(false)

  const rename = (name: string) => {
    setRenaming(null)

    if (name) {onRename(name)}
  }

  return <>
    <header className="flex items-center gap-2">
      {onBack && <Button onClick={onBack}>{labels.back}</Button>}
      <div className="min-w-0">
        <h2 className="text-sm font-medium">{room?.name || labels.loadingGroup}</h2>
        {context?.purpose && <p className="truncate text-xs text-(--ui-text-secondary)" title={context.purpose}>{context.purpose}</p>}
        <CanonicalGroupRoster label={labels.members} labels={labels} members={room?.members} phases={phases} />
      </div>
      <Button disabled={busy || !canStop} onClick={onStop}>{labels.stop}</Button>
      <Button disabled={busy || !room} onClick={() => setRenaming(room?.name ?? '')}>{labels.rename}</Button>
      <Button disabled={busy || !room || !context} onClick={() => setSettings({ purpose: context?.purpose ?? '', notes: context?.notes ?? '' })}>{labels.roomSettings}</Button>
      <Button disabled={busy || !room} onClick={() => setEnding(true)}>{labels.disband}</Button>
    </header>
    {renaming !== null && <form className="flex items-center gap-2" onSubmit={event => { event.preventDefault(); rename(renaming.trim()) }}>
      <Input aria-label={labels.renameLabel} autoFocus onChange={event => setRenaming(event.target.value)} value={renaming} />
      <Button disabled={busy || !renaming.trim()} type="submit">{labels.save}</Button>
      <Button onClick={() => setRenaming(null)} type="button">{labels.cancel}</Button>
    </form>}
    {settings !== null && <form aria-label={labels.roomSettings} className="grid gap-2" onSubmit={event => { event.preventDefault(); setSettings(null); onSaveContext(settings) }}>
      <label className="grid gap-1 text-xs">{labels.purposeLabel}
        <Input maxLength={500} onChange={event => setSettings({ ...settings, purpose: event.target.value })} placeholder={labels.purposePlaceholder} value={settings.purpose} />
      </label>
      <label className="grid gap-1 text-xs">{labels.notesLabel}
        <textarea className="min-h-24 rounded border border-(--ui-border) bg-transparent p-2 text-xs" onChange={event => setSettings({ ...settings, notes: event.target.value })}
          placeholder={labels.notesPlaceholder} value={settings.notes} />
      </label>
      <p className="text-xs text-(--ui-text-secondary)">{labels.deliveryTarget} <code data-selectable-text="true">{deliveryTarget}</code></p>
      <div className="flex gap-2">
        <Button disabled={busy} type="submit">{labels.save}</Button>
        <Button onClick={() => setSettings(null)} type="button">{labels.cancel}</Button>
      </div>
    </form>}
    {ending && <ConfirmDialog busyLabel={labels.disbanding} confirmLabel={labels.disband}
      description={<span>{labels.disbandDescPrefix}<span className="font-medium text-foreground">{room?.name}</span>{labels.disbandDescSuffix(room?.members?.length ?? 0)}</span>}
      destructive doneLabel={labels.disbandDone} onClose={() => setEnding(false)} onConfirm={onDisband} open title={labels.disbandTitle} />}
  </>
}
