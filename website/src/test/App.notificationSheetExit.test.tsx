/**
 * Notification Center sheet — exit animation.
 *
 * The bell sheet plays `nc-slide-in` on open; dismissal must keep it mounted
 * long enough to play `nc-slide-out` instead of ripping the portal out on the
 * same tick (the bug: "slides in but doesn't slide out on dismiss").
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { act, screen, fireEvent } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import tailwindConfig from '../../tailwind.config.js'

vi.mock('../pages/ChatPage', () => ({ default: () => <div data-testid="chat-page">ChatPage</div> }))
vi.mock('../pages/SystemPage', () => ({ default: () => null }))
vi.mock('../pages/AgentsPage', () => ({ default: () => null }))
vi.mock('../pages/ProjectsPage', () => ({ default: () => null }))
vi.mock('../pages/LogsPage', () => ({ default: () => null }))
vi.mock('../pages/KiroCrewAgentsPage', () => ({ default: () => null }))
vi.mock('../pages/NotificationsPage', () => ({ default: () => null }))
vi.mock('../pages/SchedulePage', () => ({ default: () => null }))
vi.mock('../hooks/useWebSocket', () => ({ useWebSocket: () => ({ subscribeLogs: () => {} }) }))
vi.mock('../hooks/useAgents', () => ({ useAgents: vi.fn(() => ({ agents: [{ name: 'kirocrew' }], defaultAgent: 'kirocrew' })) }))
vi.mock('../providers/context', () => ({ useProvider: () => ({ id: 'acp' }) }))
vi.mock('../components/MarkdownRenderer', () => ({ default: ({ content }: { content: string }) => <span>{content}</span>, Lightbox: () => null }))

vi.mock('../api/client', () => ({
  api: {
    chatSlots: vi.fn().mockResolvedValue([]),
    notifications: vi.fn().mockResolvedValue({ notifications: [] }),
    status: vi.fn().mockResolvedValue({ uptime: '1h', sessions: 0, messages: 0, cron_jobs: 0, subagents: 0, lessons: 0 }),
    listApps: vi.fn().mockResolvedValue([]),
    system: vi.fn().mockResolvedValue({ mem_used_gb: 4.0, mem_total_gb: 16.0, cpu_pct: 25.0, disk_total_gb: 100.0, disk_free_gb: 60.0 }),
    chatSlotAgent: vi.fn().mockResolvedValue({}),
    chatSlotReasoningEffort: vi.fn().mockResolvedValue({}),
    chatSlotModel: vi.fn().mockResolvedValue({}),
    chatMode: vi.fn().mockResolvedValue({}),
    listInstances: vi.fn().mockResolvedValue({ instances: [], warm_set_cap: 5 }),
  },
  isAuthBannerShown: vi.fn(() => false),
  ApiError: class ApiError extends Error {
    status: number
    constructor(status: number, message: string) { super(message); this.status = status }
  },
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: query === '(prefers-color-scheme: dark)',
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })),
})
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any

import App from '../App'

const sheet = () => document.querySelector('.animate-nc-slide-in, .animate-nc-slide-out') as HTMLElement | null

/**
 * Render, resolve the bell, THEN install fake timers.
 *
 * Order matters: `shouldAdvanceTime` lets fake time track the wall clock, so if
 * the timers were installed before the async `findByLabelText` await, a slow
 * runner could burn the 240ms unmount budget inside that await and tear the
 * portal down before the mid-exit assertions run.
 */
async function renderAndFindBell() {
  renderWithProviders(<App />, { route: '/chat' })
  const bell = await screen.findByLabelText('Notifications')
  vi.useFakeTimers({ shouldAdvanceTime: true })
  return bell
}

describe('Notification Center sheet — slide-out on dismiss', () => {
  afterEach(() => { vi.useRealTimers() })

  it('plays the exit animation before unmounting, then unmounts', async () => {
    const bell = await renderAndFindBell()

    fireEvent.click(bell)
    expect(sheet()?.classList.contains('animate-nc-slide-in')).toBe(true)

    // Dismiss: still mounted, now playing the exit animation and fully inert —
    // untouchable by pointer, keyboard and assistive tech.
    fireEvent.click(bell)
    const closingSheet = sheet()
    expect(closingSheet).toBeTruthy()
    expect(closingSheet!.classList.contains('animate-nc-slide-out')).toBe(true)
    expect(closingSheet!.classList.contains('pointer-events-none')).toBe(true)
    expect(closingSheet!.hasAttribute('inert')).toBe(true)
    expect(closingSheet!.getAttribute('aria-hidden')).toBe('true')

    // ...and gone once the animation has run.
    await act(async () => { vi.advanceTimersByTime(300) })
    expect(sheet()).toBeNull()
  })

  it('re-opening mid-exit cancels the pending unmount', async () => {
    const bell = await renderAndFindBell()

    fireEvent.click(bell)
    fireEvent.click(bell)
    expect(sheet()?.classList.contains('animate-nc-slide-out')).toBe(true)

    fireEvent.click(bell)
    await act(async () => { vi.advanceTimersByTime(300) })
    const reopened = sheet()
    expect(reopened?.classList.contains('animate-nc-slide-in')).toBe(true)
    // Re-opening must also lift the inert/aria-hidden guard.
    expect(reopened!.hasAttribute('inert')).toBe(false)
    expect(reopened!.hasAttribute('aria-hidden')).toBe(false)
  })

  it('returns focus to the bell when Escape dismisses the sheet', async () => {
    const bell = await renderAndFindBell()

    fireEvent.click(bell)
    expect(sheet()).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(document.activeElement).toBe(bell)
    expect(sheet()?.classList.contains('animate-nc-slide-out')).toBe(true)
  })
})

/**
 * Compositor guards — the sheet's slide must stay off the main thread.
 *
 * The keyframes originally animated `marginRight`: a layout property, so
 * every animation frame reflowed the sheet and its whole subtree on the main
 * thread — the same jank class as the old right-panel `width` animation, and
 * exactly what makes an open/close stutter while sessions are streaming. The
 * fix animates `transform` (compositor-driven). The old margin choice was
 * justified by a claim that a transformed ancestor becomes a backdrop root
 * and breaks the cards' backdrop-filter; a Chromium pixel probe (card blur
 * measured mid-animation under a transformed ancestor) disproved that —
 * transform is not in the backdrop-root trigger list.
 *
 * Percentage translateX resolves against the element's OWN width, which is
 * why one keyframe pair now covers both the desktop 400px sheet and the
 * mobile full-width sheet (the old `-full` variants are gone).
 */
describe('Notification Center sheet — compositor-driven slide', () => {
  const keyframes = (tailwindConfig as { theme: { extend: { keyframes: Record<string, Record<string, Record<string, string>>> } } })
    .theme.extend.keyframes

  it('animates transform only, never a layout property', () => {
    for (const name of ['nc-slide-in', 'nc-slide-out'] as const) {
      const frames = keyframes[name]
      expect(frames, `${name} keyframes must exist`).toBeTruthy()
      for (const [stop, decl] of Object.entries(frames)) {
        expect(Object.keys(decl), `${name} @${stop} must animate transform only`).toEqual(['transform'])
      }
    }
  })

  it('covers mobile via own-width percentages instead of a -full variant', () => {
    // translateX(%) resolves against the element's own width, so the single
    // pair covers the full-width mobile sheet; a resurrected -full variant
    // means the margin split came back.
    expect(keyframes['nc-slide-in-full']).toBeUndefined()
    expect(keyframes['nc-slide-out-full']).toBeUndefined()
    expect(keyframes['nc-slide-in'].from.transform).toContain('100%')
  })
})
