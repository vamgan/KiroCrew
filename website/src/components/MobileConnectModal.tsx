import { useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Copy, Check, Smartphone, X, ArrowRight } from 'lucide-react'
import { api } from '../api/client'
import { Btn } from './ui'
import { useAppSelector } from '../store'
import { useDialogFocusTrap } from '../hooks/useDialogFocusTrap'
import { copyToClipboard } from '../utils/clipboard'

/**
 * "Connect your phone" — the sidebar entry's centered dialog (mockup A1).
 *
 * Renders one section per governed method the `/api/mobile-connect/methods`
 * endpoint returned (the CPP `mobile_connect` seam). Kinds this frontend
 * knows: `tailnet_qr` (server-rendered QR carrying a live session token,
 * minted ONLY on explicit click) and `login_link` (one-time sign-in URL).
 * An unrecognised kind renders nothing — an edition's new method degrades to
 * absent on this frontend, never to a broken panel.
 *
 * Credentials are minted on demand and never on mount: the QR/link responses
 * carry live tokens, so nothing here fetches one until the user asks.
 */
export default function MobileConnectModal({
  kinds,
  onClose,
}: {
  kinds: string[]
  onClose: () => void
}) {
  const { t } = useTranslation()
  const dialogRef = useRef<HTMLDivElement>(null)
  // The repo's modal keyboard contract: focus-in on mount, focus-restore on
  // close, Tab trapped inside the dialog, Escape dismisses.
  useDialogFocusTrap(dialogRef, onClose)

  const hasQr = kinds.includes('tailnet_qr')

  return (
    <div
      className="fixed inset-0 z-50 bg-bg/80 backdrop-blur-sm flex items-center justify-center animate-rise"
      role="presentation"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={t('components.mobileConnect.use_kiro_crew_on_your_phone')}
        className="bg-card border border-border rounded-xl shadow-xl w-[440px] max-w-[90vw] max-h-[85vh] overflow-y-auto outline-none p-6 relative"
      >
        <button
          onClick={onClose}
          aria-label={t('components.mobileConnect.close')}
          className="absolute top-3 right-3 text-muted hover:text-text bg-transparent border-none cursor-pointer p-1"
        >
          <X size={16} />
        </button>
        <div className="flex items-center gap-2 text-[15px] font-semibold text-text-strong mb-1.5">
          <Smartphone size={16} className="shrink-0" />
          {t('components.mobileConnect.use_kiro_crew_on_your_phone')}
        </div>
        {hasQr && (
          <p className="text-[12.5px] text-muted leading-relaxed mb-4">
            {t('components.mobileConnect.scan_with_your_phone_camera_to_continue_with_the_s')}
          </p>
        )}
        {hasQr && <TailnetQrSection onClose={onClose} />}
        {kinds.includes('login_link') && <LoginLinkSection standalone={!hasQr} />}
      </div>
    </div>
  )
}

/** Copy button with a transient confirmation tick. */
function CopyBtn({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    let ok = false
    try {
      ok = await copyToClipboard(value)
    } catch {
      ok = false
    }
    setCopied(ok)
    if (ok) setTimeout(() => setCopied(false), 2000)
  }
  return (
    <Btn onClick={copy}>
      {copied ? <Check size={14} /> : <Copy size={14} />} {label}
    </Btn>
  )
}

/** Tailnet QR: mint on explicit click when the machine state is `ready`;
 *  otherwise a one-line state + a path to the real setup card (Settings →
 *  Overview), never a rebuilt wizard. */
function TailnetQrSection({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  // Single probe on open (the card's gentle-poll contract): the modal is
  // short-lived, and minting re-validates server-side anyway.
  const { data: status, isPending: probing, isError: probeFailed } = useQuery({
    queryKey: ['mobile-connect-tailnet-probe'],
    queryFn: () => api.tailnetMobile(),
    staleTime: 30_000,
    retry: false,
  })
  const mintQr = useMutation({ mutationFn: () => api.tailnetMobileQr() })

  const ready = status?.step === 'ready'
  return (
    <div className="mb-4">
      {probing && (
        <p className="text-[11.5px] text-muted mb-2">{t('components.mobileConnect.checking_remote_access')}</p>
      )}
      {probeFailed && (
        <p className="text-[11.5px] text-danger mb-2">{t('components.mobileConnect.could_not_check_remote_access_reopen_this_dialog_t')}</p>
      )}
      {ready && !mintQr.data && (
        <div className="flex justify-center py-6">
          <Btn onClick={() => mintQr.mutate()} disabled={mintQr.isPending}>
            {mintQr.isPending
              ? t('components.mobileConnect.generating')
              : t('components.mobileConnect.show_qr_code')}
          </Btn>
        </div>
      )}
      {mintQr.data && (
        <div className="flex flex-col items-center gap-3 mb-2">
          <div className="bg-white p-2.5 rounded-lg leading-none">
            {/* Server-rendered PNG carrying a live session token — shown, never logged. */}
            <img src={mintQr.data.image} alt={t('components.mobileConnect.qr_code_for_mobile_access')} width={176} height={176} />
          </div>
          <div className="flex items-center gap-2 w-full">
            <input
              readOnly
              value={mintQr.data.url}
              aria-label={t('components.mobileConnect.mobile_access_link')}
              className="flex-1 min-w-0 bg-bg border border-border rounded-md text-muted text-[11.5px] px-2.5 py-2 font-mono overflow-hidden text-ellipsis"
            />
            <CopyBtn value={mintQr.data.url} label={t('components.mobileConnect.copy_link')} />
          </div>
        </div>
      )}
      {mintQr.isError && (
        <p className="text-[11.5px] text-danger mb-2">{t('components.mobileConnect.could_not_generate_a_code_try_again')}</p>
      )}
      {status && !ready && (
        <button
          onClick={() => { onClose(); navigate('/settings/overview') }}
          className="w-full flex items-center justify-between gap-2 bg-bg-elevated border border-border rounded-lg px-3 py-2.5 mb-2 text-left cursor-pointer hover:bg-bg-hover transition-colors"
        >
          <span className="text-[12px] text-muted">
            {t('components.mobileConnect.remote_access_is_not_set_up_yet_finish_setup_to_ge')}
          </span>
          <ArrowRight size={14} className="shrink-0 text-muted" />
        </button>
      )}
      {ready && mintQr.data && (
        <p className="text-[11.5px] text-muted">
          <span className="text-accent">● </span>
          {t('components.mobileConnect.remote_access_ready_the_code_carries_a_live_sign_i', {
            minutes: Math.max(1, Math.round((mintQr.data.ttl_secs ?? 3600) / 60)),
          })}
        </p>
      )}
    </div>
  )
}

/** One-time login link for the configured external origin. `standalone` means
 *  the QR section is absent (link-only editions / governance): no divider, and
 *  an intro that does not presuppose a camera alternative. */
function LoginLinkSection({ standalone }: { standalone: boolean }) {
  const { t } = useTranslation()
  // The active slot's key rides the request so the server's restricted-session
  // guard sees the REAL session, not the shared `dashboard:ui` default.
  const activeSlot = useAppSelector(s => s.chat.activeSlot)
  const sessionKey = activeSlot ? `dashboard:${activeSlot}` : undefined
  const createLink = useMutation({ mutationFn: () => api.mobileLoginLink(sessionKey) })

  return (
    <div className={standalone ? '' : 'border-t border-border pt-3'}>
      <div className="text-[12px] text-muted mb-2">
        {standalone
          ? t('components.mobileConnect.create_a_one_time_sign_in_link_to_send_to_yourself')
          : t('components.mobileConnect.or_create_a_one_time_sign_in_link_to_send_to_yours')}
      </div>
      {!createLink.data && (
        <Btn onClick={() => createLink.mutate()} disabled={createLink.isPending}>
          {createLink.isPending
            ? t('components.mobileConnect.creating')
            : t('components.mobileConnect.create_sign_in_link')}
        </Btn>
      )}
      {createLink.data && (
        <div className="flex items-center gap-2">
          <input
            readOnly
            value={createLink.data.url}
            aria-label={t('components.mobileConnect.one_time_sign_in_link')}
            className="flex-1 min-w-0 bg-bg border border-border rounded-md text-muted text-[11.5px] px-2.5 py-2 font-mono overflow-hidden text-ellipsis"
          />
          <CopyBtn value={createLink.data.url} label={t('components.mobileConnect.copy_link')} />
        </div>
      )}
      {createLink.isError && (
        <p className="text-[11.5px] text-danger mt-2">{t('components.mobileConnect.could_not_create_a_link_check_that_an_external_add')}</p>
      )}
      {createLink.data && (
        <p className="text-[11.5px] text-muted mt-2">
          {t('components.mobileConnect.the_link_works_once_and_expires_in_about_minutes', {
            minutes: Math.max(1, Math.round((createLink.data.expires_in ?? 900) / 60)),
          })}
        </p>
      )}
    </div>
  )
}
