import { useEffect, useRef, useState } from 'react'
import { animate, type MotionValue } from 'framer-motion'
import { holdStreamingFlushes, releaseStreamingFlushes } from '../lib/streamHold'

/**
 * Finger-tracking open/close gesture for the mobile sessions drawer.
 *
 * Supersedes `useSwipeEdge`, which was a threshold DETECTOR: it read the
 * displacement once on `touchend` and fired a plain callback, so the panel
 * snapped open after the fact with nothing on screen following the finger, and
 * a drag begun and then reconsidered still committed. This hook drives the
 * panel's offset continuously instead, which is what makes a half-drag
 * readable and a drag-back cancellable.
 *
 * ONE binding covers both directions. The accept rule depends on whether the
 * panel is currently open, and `open` is deliberately read from a ref rather
 * than a dependency: the opening drag flips it mid-gesture (see
 * `onGestureOpen`), and a dep would tear the listeners down under the finger —
 * leaving the release to land on nothing while the panel stayed half-open.
 */

/** Travel, in px, before the gesture's axis is decided. */
const AXIS_LOCK = 10
/**
 * Inner edge of the band an OPENING drag may start in.
 *
 * Not 0. The platform's own back-swipe lives in the first ~20-30px and is not
 * cancellable from script — `preventDefault` does not reach it — so a band that
 * starts at the bezel loses a share of its gestures to the browser no matter
 * what this code does. Starting inboard of it means the two gestures mostly do
 * not contend for the same touch.
 */
const EDGE_START = 24
/**
 * Outer edge of that band.
 *
 * A band, not a third of the screen: the predecessor used 35% of the viewport
 * (137px on a 390px phone), so any rightward drag begun in the left third of a
 * message opened the drawer.
 */
const EDGE_END = 120
/** Release past this share of the travel commits to the new state. */
const COMMIT_DISTANCE = 0.5
/** Release faster than this (px/ms) commits regardless of how far it got. */
const COMMIT_VELOCITY = 0.4
/**
 * The settle curves, and their durations — ASYMMETRIC, one shape per direction.
 *
 * A cubic-bezier rather than a spring for two load-bearing reasons: the shape
 * is stated rather than tuned by feel, and the compositor path below hands the
 * curve to a `KeyframeEffect`, which a spring (no closed form) cannot express
 * without sampling itself on the main thread every frame — the exact thing the
 * compositor path exists to avoid.
 *
 * IN is iOS's sheet presentation curve: `(0.32, 0.72, 0, 1)`. That shape is not
 * invented here — it is the curve Ionic ships for its iOS transitions and the one
 * Vaul uses (a drawer library written to reproduce Apple's Sheet, whose author
 * states the curve and 500ms as the iOS match). It is also already this repo's
 * own drawer curve: `components/OverlayDrawer.tsx` has carried
 * `EASE = [0.32, 0.72, 0, 1]` all along, with the reason spelled out there —
 * "near-linear on purpose: a strong ease-out front-loads the travel, which
 * visually freezes the near edges while the far edges are still sweeping."
 *
 * That is the constraint to respect. Three progressively more front-loaded
 * easeOut shapes were tried on a device and each was rejected for reading as the
 * panel appearing rather than sliding: easeOutExpo `(0.19, 1, 0.22, 1)` at 340ms
 * (26% of the travel gone in the first painted frame), easeOutQuint
 * `(0.16, 1, 0.3, 1)` at 320ms (30%), and `(0.1, 0.9, 0.2, 1)` at 320ms (39%).
 * This pairing spends 10%. The FIRST FRAME is the measurement that tracks the
 * complaint, which is why the guard is stated that way.
 *
 * 420ms rather than the iOS 500ms: the shape is what was being judged, and the
 * shorter tail reads better on a side drawer than on a bottom sheet. Either is
 * affordable only because the settle reaches the compositor — a longer tween used
 * to mean more frames exposed to main-thread stalls, so length was a risk to be
 * minimized. It no longer is, and the duration can be chosen for how it reads.
 *
 * OUT stays its own shape — an easeIN, spending 2% of the travel in the first
 * 50ms and 10% by 100ms, so a dismissal starts from where the panel is and then
 * leaves quickly. Reusing an easeOut for the exit does the opposite: it jumps off
 * the edge and then crawls to a stop.
 */
const SETTLE_IN_EASE = [0.32, 0.72, 0, 1] as const
const SETTLE_IN_SECS = 0.42
const SETTLE_OUT_EASE = [0.3, 0, 0.8, 0.15] as const
const SETTLE_OUT_SECS = 0.24

/** The four control-point coordinates of a cubic-bezier. Kept as a TUPLE, not
 *  `number[]` — framer's `Easing` accepts only the fixed-arity form, so widening
 *  it here surfaces as a type error at the `animate` call. */
type Bezier = readonly [number, number, number, number]

/** Rest is offset 0 (see `scrimOpacity`), so the target alone says which
 *  direction a settle is going, and every call site already spells it that way:
 *  open animates to 0, close to ±travel. */
function settleFor(to: number): { ease: Bezier; secs: number } {
  return to === 0
    ? { ease: SETTLE_IN_EASE, secs: SETTLE_IN_SECS }
    : { ease: SETTLE_OUT_EASE, secs: SETTLE_OUT_SECS }
}

/** A settle in the spelling a `KeyframeEffect` takes — derived from the same
 *  arrays, so the compositor and main-thread paths cannot drift apart. */
function settleTiming(to: number): KeyframeAnimationOptions {
  const { ease, secs } = settleFor(to)
  return { duration: secs * 1000, easing: `cubic-bezier(${ease.join(', ')})`, fill: 'forwards' }
}
/** Reduced motion still needs the panel to ARRIVE — dragging is direct
 *  manipulation, and dropping the settle to 0 would teleport the panel out from
 *  under the finger. A short linear tween carries it there without the curve. */
const DRAWER_SETTLE_REDUCED = { duration: 0.12, ease: 'linear' as const }

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * The DOM the settle animates, when it can reach the compositor.
 *
 * `travel` is a function, not a number: the closed offset is the viewport width
 * and a rotation changes it between a register and a settle.
 */
export interface DrawerTargets {
  panel: () => HTMLElement | null
  scrim: () => HTMLElement | null
  travel: () => number
}

interface DrawerRuntime {
  targets: DrawerTargets
  /** Compositor animations currently carrying the panel, so a new gesture can
   *  take them over instead of writing the same channel underneath them. */
  running: Animation[]
}

/**
 * Per-MotionValue runtime, keyed weakly so a torn-down ChatPage takes its entry
 * with it. A WeakMap rather than a hook argument because `animateDrawer` is
 * called from four places that have no business knowing about DOM nodes (the
 * header toggle, the scrim tap, the session-selected close, and this hook's own
 * release), and threading the elements through all four would put the same
 * lookup in each of them.
 */
const runtimes = new WeakMap<MotionValue<number>, DrawerRuntime>()

/**
 * Point `x`'s settles at real DOM. Returns a deregister for unmount.
 *
 * PRECONDITION, and the reason the first compositor attempt shipped a visible
 * bug: nothing inside the registered panel may be a Framer layout-projection
 * node (`layout` / `layoutId`). Projection only stays correct while framer owns
 * every animated ancestor transform; under a WAAPI-driven ancestor it attributes
 * the panel's travel to the descendants themselves and compounds a corrective
 * transform per re-measure (measured: >4,000px — the sidebar rows visibly flew
 * in from the panel's right edge). ChatSidebar therefore renders its rows
 * WITHOUT projection when hosted in this drawer (`staticRows`), and that pairing
 * is what makes this registration safe — see ChatSidebar.staticRows.test.tsx.
 */
export function registerDrawerTargets(x: MotionValue<number>, targets: DrawerTargets): () => void {
  runtimes.set(x, { targets, running: [] })
  return () => runtimes.delete(x)
}

/** Scrim opacity for a panel offset: 0 fully closed, 1 at rest. Direction-
 *  agnostic — a LEFT drawer runs offset in [-travel, 0], a RIGHT drawer in
 *  [0, +travel], and |offset| is the distance from rest either way. */
function scrimOpacity(offset: number, travel: number): number {
  if (travel <= 0) return 1
  return Math.max(0, Math.min(1, 1 - Math.abs(offset) / travel))
}

/** Move `x` with NO animation and no inherited velocity, cancelling anything
 *  running on it. `set` alone does not cancel — see the note in `settle` — so
 *  `jump` is the correct verb here wherever the build provides it. */
function jumpTo(x: MotionValue<number>, to: number): void {
  if (typeof x.jump === 'function') x.jump(to)
  else x.set(to)
}

/** Current translateX of an element, read from its resolved matrix — including
 *  the value a running compositor animation is presenting this instant. */
function currentTranslateX(el: HTMLElement): number | null {
  if (typeof getComputedStyle !== 'function') return null
  const t = getComputedStyle(el).transform
  if (!t || t === 'none') return null
  const nums = t.slice(t.indexOf('(') + 1, -1).split(',').map(v => parseFloat(v))
  if (t.startsWith('matrix3d')) return Number.isFinite(nums[12]) ? nums[12] : null
  return Number.isFinite(nums[4]) ? nums[4] : null
}

/**
 * Hand `x` back the offset a compositor settle is presenting, and cancel it.
 *
 * Called wherever plain `x.stop()` used to go: a compositor animation is NOT an
 * animation ON the value, so `stop`/`jump` do not reach it — left running, it
 * would keep presenting its own offset over every write the finger makes, and
 * the panel would refuse to follow until the animation ran out.
 */
export function takeOverDrawer(x: MotionValue<number>): void {
  const rt = runtimes.get(x)
  if (rt && rt.running.length > 0) {
    const panel = rt.targets.panel()
    const at = panel ? currentTranslateX(panel) : null
    for (const a of rt.running) a.cancel()
    rt.running = []
    // Adopt the presented offset BEFORE anything repaints — into the element's
    // OWN inline style as well as the value, for the same reason publishArrival
    // does: cancel reverts to inline style, and only framer-bound panels get
    // theirs rewritten by jumpTo.
    if (at != null && panel) {
      panel.style.transform = `translate3d(${at}px, 0, 0)`
      const scrim = rt.targets.scrim()
      if (scrim) scrim.style.opacity = String(scrimOpacity(at, rt.targets.travel()))
      jumpTo(x, at)
    }
  }
  x.stop()
}

/** Animate a drawer offset to its resting place with the shared settle curve.
 *  Exported so the tap/backdrop/programmatic paths land identically to a
 *  released gesture — two curves for one panel is how the two paths visibly
 *  drift apart.
 *
 *  Runs on the COMPOSITOR when `registerDrawerTargets` has supplied elements:
 *  `transform` and `opacity` keyframes on the panel and the scrim, which the
 *  browser hands to the compositor thread. That is the point — the panel and
 *  the chat pane behind it share one main thread, and with sessions streaming
 *  that thread stalls unpredictably (chunk flushes are held, see
 *  lib/streamHold.ts, but tool events, subagent status pushes and the panel's
 *  own mount are not), so ONLY an animation that does not need the main thread
 *  holds its frame rate through the slide. Falls back to the main-thread tween
 *  when no element is registered (embed frames, tests) or under reduced motion.
 *
 *  The streaming-flush hold is kept on both paths: it starves the projection
 *  re-measures and the transcript repaints of their per-frame trigger, which
 *  is cheap insurance either way. */
export function animateDrawer(x: MotionValue<number>, to: number, onDone?: () => void) {
  const reduce = prefersReducedMotion()
  const rt = runtimes.get(x)

  const mainThread = () => {
    const curve = reduce ? DRAWER_SETTLE_REDUCED : { duration: settleFor(to).secs, ease: settleFor(to).ease }
    holdStreamingFlushes(curve.duration * 1000 + 100)
    /**
     * Paint the tween onto the ELEMENTS' OWN inline styles, not just `x`.
     *
     * Framer writes a transform only where it is BOUND to `x` (the sessions
     * drawer's `style={{ x }}`). The nav drawer is a plain <nav> and the right
     * overlay carries a template-string transform serialized once per React
     * render, so on this path their MotionValue would travel while their DOM
     * never moved — the panel stays at the mounted CLOSED offset and is simply
     * unreachable. Neither of those two has a drag gesture, so this fallback is
     * their ONLY path whenever the compositor one is unavailable: under
     * `prefers-reduced-motion`, and when the panel element has not appeared
     * within the mount grace frames. Same root cause as the cancel-fill bounce
     * (a non-framer-bound element needs the write spelled out), on the fallback
     * path instead of the compositor one. Redundant where framer IS bound — it
     * writes the same value from its own subscription.
     */
    const paint = rt
      ? (at: number) => {
        const panel = rt.targets.panel()
        if (panel) panel.style.transform = `translate3d(${at}px, 0, 0)`
        const scrim = rt.targets.scrim()
        if (scrim) scrim.style.opacity = String(scrimOpacity(at, rt.targets.travel()))
      }
      : null
    const unbind = paint ? x.on('change', paint) : null
    paint?.(x.get())
    const controls = animate(x, to, {
      ...curve,
      onComplete: () => { paint?.(to); unbind?.(); releaseStreamingFlushes(); onDone?.() },
    })
    return () => { unbind?.(); controls.stop() }
  }
  if (!rt || reduce) return mainThread()

  let raf: number | null = null
  let cancelled = false
  let stopMain: (() => void) | null = null
  /** Frames to wait for the panel to exist. The tap-open path calls this in the
   *  same tick as the setState that MOUNTS the panel, so on that path there is
   *  nothing to animate yet — and waiting is correct rather than merely
   *  tolerable, because the first painted frame is supposed to be the closed
   *  offset anyway. Bounded so a panel that never arrives degrades to the
   *  main-thread path instead of silently never settling. */
  let tries = 3

  const start = () => {
    if (cancelled) return
    raf = null
    /**
     * Adopt whatever the PREVIOUS settle is presenting, and cancel it, before
     * keyframing a replacement. Reversing mid-settle (close then re-open inside
     * the 420ms) otherwise breaks twice over, and nothing else cleans up: no
     * caller keeps this function's returned canceller.
     *
     * `x` does not move while the compositor owns the offset, so on a reversal
     * `from` would still read the offset the OUTGOING settle started from — a
     * close→open reversal keyframes 0→0 and snaps instead of sliding. And the
     * outgoing animation is `fill: 'forwards'`: once the replacement is
     * cancelled on arrival, that stale fill re-presents its own end state OVER
     * the published inline style, so the panel lands open and then goes back
     * offscreen while its phase still says `open` — unreachable.
     */
    takeOverDrawer(x)
    const panel = rt.targets.panel()
    if (!panel || typeof panel.animate !== 'function') {
      if (panel || tries-- <= 0) { stopMain = mainThread(); return }
      raf = requestAnimationFrame(start)
      return
    }
    const timing = settleTiming(to)
    holdStreamingFlushes(Number(timing.duration) + 100)
    const from = x.get()
    const travel = rt.targets.travel()
    const anims: Animation[] = [panel.animate(
      [{ transform: `translate3d(${from}px, 0, 0)` }, { transform: `translate3d(${to}px, 0, 0)` }],
      timing,
    )]
    const scrim = rt.targets.scrim()
    // The scrim's opacity is derived from the same offset, so it has to be
    // animated in lockstep here: its own binding reads `x`, which does not move
    // while the compositor owns the panel.
    if (scrim && typeof scrim.animate === 'function') {
      anims.push(scrim.animate(
        [{ opacity: scrimOpacity(from, travel) }, { opacity: scrimOpacity(to, travel) }],
        timing,
      ))
    }
    rt.running = anims
    /**
     * Publish the arrival into the ELEMENTS' OWN inline styles FIRST, then the
     * MotionValue, then cancel. Cancelling a `fill: 'forwards'` animation
     * reverts each element to its inline style — and `jumpTo(x)` rewrites that
     * style only where framer is BOUND to `x` (the sessions drawer's
     * `style={{ x }}`). The nav drawer is a plain <nav> and the right overlay
     * carries a template-string transform, so on those the inline style is
     * whatever their last React render serialized — the CLOSED offset, for a
     * panel that mounted closed. Cancelling without this write snapped the
     * just-arrived panel offscreen until the next unrelated re-render popped it
     * back: the recorded open→vanish→flash-open bounce.
     */
    const publishArrival = () => {
      panel.style.transform = `translate3d(${to}px, 0, 0)`
      if (scrim instanceof HTMLElement) scrim.style.opacity = String(scrimOpacity(to, travel))
      jumpTo(x, to)
    }
    const settled = () => {
      if (rt.running !== anims) return // taken over by a newer gesture
      publishArrival()
      for (const a of anims) a.cancel()
      rt.running = []
      releaseStreamingFlushes()
      onDone?.()
    }
    anims[0].onfinish = settled
    // A compositor animation on a hidden/backgrounded element can be cancelled
    // by the browser rather than finished; the panel must still arrive.
    anims[0].oncancel = () => { if (rt.running === anims) { rt.running = []; publishArrival(); releaseStreamingFlushes(); onDone?.() } }
  }

  start()
  return () => {
    cancelled = true
    if (raf != null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(raf)
    stopMain?.()
    for (const a of rt.running) a.cancel()
    rt.running = []
  }
}

/**
 * Nearest ancestor of `from`, up to and including `root`, that scrolls
 * horizontally. Returns null when the touch did not start inside one.
 */
function findHorizontalScroller(from: EventTarget | null, root: HTMLElement): HTMLElement | null {
  let node: Element | null = from instanceof Element ? from : null
  while (node) {
    if (node instanceof HTMLElement && node.scrollWidth - node.clientWidth > 1) {
      const overflowX = getComputedStyle(node).overflowX
      if (overflowX === 'auto' || overflowX === 'scroll') return node
    }
    if (node === root) break
    node = node.parentElement
  }
  return null
}

interface DrawerSwipeOptions {
  /** Bind the gesture at all. Mobile-only — a pointer device has the toggle. */
  enabled: boolean
  /** Whether the panel is currently open. Read live, never a dependency. */
  open: boolean
  /** Panel offset, in px: `-travel` fully closed, `0` fully open. */
  x: MotionValue<number>
  /** Put the panel in the DOM at its closed offset so the drag can reveal it. */
  onGestureOpen: () => void
  /** Released. `open` is the state the panel settled into, reported only once
   *  the settle animation has finished so an unmount cannot cut it short. */
  onSettle: (open: boolean) => void
}

/** @returns whether a drag currently owns the panel (suppress transitions). */
export function useDrawerSwipe(
  ref: React.RefObject<HTMLElement | null>,
  { enabled, open, x, onGestureOpen, onSettle }: DrawerSwipeOptions,
): boolean {
  const [dragging, setDragging] = useState(false)

  // Everything the move handler reads lives in a ref. A touchmove fires at
  // frame rate and this hook is bound inside the chat pane, so a re-render per
  // sample would drop frames on the very gesture it is meant to smooth.
  const openRef = useRef(open)
  openRef.current = open
  const onGestureOpenRef = useRef(onGestureOpen)
  onGestureOpenRef.current = onGestureOpen
  const onSettleRef = useRef(onSettle)
  onSettleRef.current = onSettle

  const phase = useRef<'idle' | 'pending' | 'locked'>('idle')
  const startX = useRef(0)
  const startY = useRef(0)
  const travel = useRef(0)
  /**
   * Offset the panel sat at when this gesture locked, and the base every later
   * sample is measured from.
   *
   * Latched, NOT re-read from `open` per move — and that is load-bearing. An
   * opening drag mounts the panel from inside the touchmove handler, React
   * flushes that synchronously, and `open` is therefore already true by the time
   * the SAME handler reaches the tracking line. Reading it there computed the
   * base for an already-open panel (0) instead of the closed one, so the offset
   * clamped to 0 and the panel appeared instantly at rest: a snap, i.e. exactly
   * the behaviour this hook replaces. The base belongs to the gesture, so it is
   * decided once, when the gesture is decided.
   */
  const gestureBase = useRef(0)
  const lastX = useRef(0)
  const lastT = useRef(0)
  const velocity = useRef(0)
  const scroller = useRef<HTMLElement | null>(null)
  const scrollerLeft = useRef(0)

  useEffect(() => {
    const el = ref.current
    if (!el || !enabled) return

    /** Offset the panel rests at while closed, and the gesture's full travel. */
    const closedOffset = () => -window.innerWidth

    /**
     * Take `x` over, then run the shared settle to `to`, reporting `open` once
     * it arrives.
     *
     * `x.stop()` rather than a stop handle this hook kept for its OWN
     * animations. Two writers to one value is the failure mode, and the other
     * writer is not always this hook: the consumer animates the same value
     * programmatically for the header toggle, the backdrop tap and the
     * session-selected close, and discards those handles. `x.set()` does NOT
     * cancel an animation running on the value — only `stop`/`jump` do — so a
     * drag begun inside one of those ~0.32s windows had the drag and the
     * animation both writing every frame, and the panel juddered until the
     * animation ran out. Stopping the VALUE covers every writer, including ones
     * added later, which no amount of handle-tracking here can.
     *
     * Stopping also suppresses the stopped animation's `onComplete`, so a
     * close that a new gesture interrupts cannot later report `closed` over the
     * gesture's own outcome.
     */
    const settle = (to: number, open: boolean) => {
      takeOverDrawer(x)
      animateDrawer(x, to, () => onSettleRef.current(open))
    }

    /** Drop a gesture that had not taken the panel over yet (still deciding its
     *  axis). Nothing was mounted and `x` was never written, so there is no
     *  visual state to put back. */
    const reset = () => {
      phase.current = 'idle'
      scroller.current = null
      setDragging(false)
    }

    /**
     * Give up a gesture that ALREADY owns the panel — a second finger, a
     * `touchcancel` from a system interruption, an incoming call.
     *
     * It is not enough to stop tracking. The panel is sitting wherever the
     * finger left it, and for an opening drag it is also MOUNTED, so merely
     * going idle strands it half-open with the scrim half-dimmed and no
     * animation coming: the release handler is the only other place that ever
     * settles it, and it never runs for a gesture that was cancelled rather
     * than released. So return the panel to the state the gesture STARTED from —
     * which is exactly `gestureBase`, 0 when it was open and the closed offset
     * when it was not — and report that state so the mount follows it.
     */
    const abandon = () => {
      if (phase.current === 'locked') settle(gestureBase.current, gestureBase.current === 0)
      reset()
    }

    const onTouchStart = (e: TouchEvent) => {
      // A second finger LANDING is a pinch or a two-finger scroll, not this
      // gesture — abandon on the spot. Checked before the phase guard on
      // purpose: a locked gesture would otherwise return here and only give the
      // panel up on the next touchmove, so a pinch that holds still would keep
      // it owned and stranded indefinitely.
      if (e.touches.length > 1) { abandon(); return }
      if (phase.current !== 'idle') return
      const touch = e.touches[0]
      if (!openRef.current) {
        const x0 = touch.clientX
        if (x0 < EDGE_START || x0 > EDGE_END) return
      }
      travel.current = window.innerWidth
      startX.current = touch.clientX
      startY.current = touch.clientY
      lastX.current = touch.clientX
      lastT.current = e.timeStamp
      velocity.current = 0
      phase.current = 'pending'
      scroller.current = findHorizontalScroller(e.target, el)
      scrollerLeft.current = scroller.current ? scroller.current.scrollLeft : 0
    }

    const onTouchMove = (e: TouchEvent) => {
      if (phase.current === 'idle') return
      if (e.touches.length > 1) { abandon(); return }
      const touch = e.touches[0]
      const dx = touch.clientX - startX.current
      const dy = touch.clientY - startY.current

      if (phase.current === 'pending') {
        // Vertical intent: the chat scroller owns this touch. Abandon outright
        // rather than staying armed, so a later horizontal wobble during a
        // scroll cannot retroactively claim the gesture.
        if (Math.abs(dy) > Math.abs(dx)) { reset(); return }
        if (Math.abs(dx) < AXIS_LOCK) return
        // A horizontal scroller under the finger (a wide code block, a
        // carousel) owns the gesture while it still has somewhere to go in
        // this direction. Checked at lock time, on the direction now known.
        const sc = scroller.current
        if (sc) {
          if (sc.scrollLeft !== scrollerLeft.current) { reset(); return }
          const maxScrollLeft = sc.scrollWidth - sc.clientWidth
          const canReveal = dx < 0 ? sc.scrollLeft < maxScrollLeft - 1 : sc.scrollLeft > 1
          if (canReveal) { reset(); return }
        }
        // Wrong direction for the current state: closed panels only open on a
        // rightward drag, open panels only close on a leftward one.
        if (openRef.current ? dx > 0 : dx < 0) { reset(); return }
        phase.current = 'locked'
        gestureBase.current = openRef.current ? 0 : closedOffset()
        // Take the value over from ANY animation still running on it — this
        // hook's own settle, or one the consumer started for the toggle, the
        // backdrop tap or a session-selected close. `x.set()` below does not
        // cancel an animation, so without this both write every frame. And a
        // compositor settle is not an animation ON the value at all, which is
        // why this goes through takeOverDrawer rather than `x.stop()`.
        takeOverDrawer(x)
        setDragging(true)
        if (!openRef.current) {
          // Seat the panel offscreen BEFORE it mounts, so the first painted
          // frame is the closed offset rather than a flash at rest position.
          x.set(closedOffset())
          onGestureOpenRef.current()
        }
      }

      if (phase.current !== 'locked') return
      // The finger owns the panel: keep the flush pipelines quiet one rolling
      // beat at a time, so the hold dies on its own if the gesture does.
      holdStreamingFlushes(250)
      const dt = e.timeStamp - lastT.current
      if (dt > 0) velocity.current = (touch.clientX - lastX.current) / dt
      lastX.current = touch.clientX
      lastT.current = e.timeStamp
      // Clamped to the panel's own range: dragging past open must not lift the
      // panel off its edge, and dragging past closed must not gap it further.
      x.set(Math.max(closedOffset(), Math.min(0, gestureBase.current + dx)))
    }

    const onTouchEnd = (e: TouchEvent) => {
      if (phase.current !== 'locked') { reset(); return }
      const touch = e.changedTouches[0]
      const dx = touch.clientX - startX.current
      // A release more than a frame after the last move is a hold, not a
      // flick — the stale sample would otherwise commit a gesture the finger
      // had already stopped.
      const v = e.timeStamp - lastT.current > 32 ? 0 : velocity.current
      const settledAt = Math.max(closedOffset(), Math.min(0, gestureBase.current + dx))
      const progress = 1 + settledAt / travel.current

      let target: boolean
      if (v > COMMIT_VELOCITY) target = true
      else if (v < -COMMIT_VELOCITY) target = false
      else target = progress > COMMIT_DISTANCE

      reset()
      // `onSettle(false)` unmounts the panel, so it is reported only once the
      // panel is offscreen — unmounting mid-slide is the snap this hook exists
      // to remove.
      settle(target ? 0 : closedOffset(), target)
    }

    el.addEventListener('touchstart', onTouchStart, { passive: true })
    el.addEventListener('touchmove', onTouchMove, { passive: true })
    el.addEventListener('touchend', onTouchEnd, { passive: true })
    el.addEventListener('touchcancel', abandon, { passive: true })
    return () => {
      el.removeEventListener('touchstart', onTouchStart)
      el.removeEventListener('touchmove', onTouchMove)
      el.removeEventListener('touchend', onTouchEnd)
      el.removeEventListener('touchcancel', abandon)
      // Unbinding mid-gesture (a viewport crossing out of mobile, an unmount)
      // must also END the gesture. `phase` is a ref, so it survives the
      // teardown: left at 'locked' it makes the next bind refuse every
      // touchstart, and lets the first touchmove resume from a stale startX —
      // the gesture would be dead until a remount, with one stray jump on the
      // way. The panel itself is not settled here; the consumer's own
      // leaving-mobile reset owns where it ends up.
      phase.current = 'idle'
      scroller.current = null
      setDragging(false)
    }
    // `open` is intentionally absent — see the header note. The callbacks are
    // held in refs for the same reason.
  }, [ref, enabled, x])

  return dragging
}
