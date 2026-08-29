/**
 * The OTHER two mobile panels ride the same compositor machinery as the
 * sessions drawer — pinned as SOURCE contracts, the way App.test.tsx pins the
 * drawer's own wrapper: these are pairings ("compositor slide" ⟷ "no framer
 * transform / no projection under it"), and one half regressing alone
 * reintroduces a bug the other half was built around.
 *
 *  LEFT (App.tsx mobile nav drawer):
 *   - the panel is a plain <nav> whose slide runs via animateDrawer — framer
 *     must not own a competing transform on it;
 *   - NavItem drops `layout` on mobile: a projection node under a
 *     compositor-driven ancestor compounds a corrective offset (the ChatSidebar
 *     rows measured >4,000px of it);
 *   - the scrim's opacity is animated in lockstep by animateDrawer, not by a
 *     framer fade of its own.
 *
 *  RIGHT (ChatPage inline side panel on mobile):
 *   - the mobile branch must NEVER animate `width` — a layout animation the
 *     compositor cannot take, which re-laid-out the squeezed chat pane every
 *     frame (the original 400ms width reveal);
 *   - it slides as a fixed overlay via sideOverlayX / animateDrawer;
 *   - keep-alive survives: a closed-but-alive panel stays mounted display:none
 *     so a live app tab's iframe is not torn down by the overlay conversion.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const app = readFileSync(resolve(__dirname, '../App.tsx'), 'utf8')
const chat = readFileSync(resolve(__dirname, '../pages/ChatPage.tsx'), 'utf8')

describe('mobile nav drawer (left) — compositor pairing', () => {
  const drawer = app.slice(app.indexOf('key="mobile-nav-drawer"'), app.indexOf('key="mobile-nav-drawer"') + 900)

  it('slides a plain <nav>, not a framer-owned motion.nav', () => {
    expect(app.indexOf('key="mobile-nav-drawer"')).toBeGreaterThan(0)
    // The element framer animates is the element the compositor cannot have:
    // two writers to one transform is the judder the sessions drawer already
    // debugged. The ref is what animateDrawer drives.
    expect(drawer).toContain('ref={mobileNavPanelRef}')
    expect(drawer).not.toContain('<motion.nav')
    expect(drawer).not.toContain('animate={{')
  })

  it('registers panel+scrim with the drawer runtime', () => {
    expect(app).toContain('registerDrawerTargets(mobileNavX')
    expect(app).toMatch(/panel: \(\) => mobileNavPanelRef\.current/)
    expect(app).toMatch(/scrim: \(\) => mobileNavScrimRef\.current/)
  })

  it('NavItem drops layout projection on mobile', () => {
    expect(app).toContain("layout={isMobileRow ? undefined : 'position'}")
  })

  it('the scrim has no framer fade of its own (animateDrawer owns it)', () => {
    const scrim = app.slice(app.indexOf('data-testid="nav-backdrop"') - 400, app.indexOf('data-testid="nav-backdrop"') + 400)
    expect(scrim).toContain('ref={mobileNavScrimRef}')
    expect(scrim).not.toContain('initial={{ opacity')
  })

  it('scrim is decorative (aria-hidden) and Escape is the keyboard dismissal', () => {
    // The scrim's click-to-dismiss is a pointer convenience. It must stay out
    // of the tab order and the accessibility tree (a focusable full-screen
    // scrim is a giant tab stop), so the keyboard path is an Escape handler
    // gated on the drawer being open — both halves pinned here.
    const scrim = app.slice(app.indexOf('data-testid="nav-backdrop"') - 400, app.indexOf('data-testid="nav-backdrop"') + 400)
    expect(scrim).toContain('aria-hidden="true"')
    const esc = app.slice(app.indexOf("mobileNavPhase !== 'open'") - 200, app.indexOf("mobileNavPhase !== 'open'") + 400)
    expect(esc).toContain("e.key === 'Escape'")
    expect(esc).toContain('closeMobileNavDrawer()')
  })
})

describe('inline side panel (right) — mobile overlay pairing', () => {
  const start = chat.indexOf('key="side-panel-inline"')
  // The mount predicate (with the keep-alive arm) sits a few hundred chars ABOVE the key.
  const block = chat.slice(start - 1400, start + 2400)

  it('mobile branch never animates width (layout animation)', () => {
    expect(start).toBeGreaterThan(0)
    // The width tween survives ONLY behind the desktop/embed ternary arm.
    expect(block).toContain('initial={isMobile ? false : { width: 0 }}')
    expect(block).toContain('animate={isMobile ? undefined : {')
  })

  it('mobile branch is a fixed overlay driven by the drawer runtime', () => {
    expect(block).toContain('ref={isMobile ? sideOverlayPanelRef : undefined}')
    expect(chat).toContain('registerDrawerTargets(sideOverlayX')
    // Seated at the closed offset inline so the first painted frame is offscreen.
    expect(block).toContain('translate3d(${sideOverlayX.get()}px, 0, 0)')
  })

  it('keep-alive: a closed-but-alive panel stays mounted display:none', () => {
    // The mount predicate keeps a hidden live-app panel in the tree…
    expect(block).toContain('|| (shouldMountSidePanel(')
    // …and the closed phase renders it invisible rather than unmounting it.
    expect(block).toMatch(/sideOverlayPhase === 'closed'\s*\n?\s*\? \{ display: 'none' \}/)
  })
})
