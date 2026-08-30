/**
 * MobileConnectModal — the sidebar "Connect your phone" dialog.
 *
 * Pins the credential-safety contract and the seam's forward-compat shape:
 *  1. a QR/link credential is minted ONLY on explicit click, never on mount
 *     (the responses carry live session tokens);
 *  2. sections render per `kinds` from the governed methods endpoint, and an
 *     unrecognised kind renders NOTHING (an edition's new method degrades to
 *     absent on this frontend, never to a broken panel);
 *  3. the not-ready tailnet state routes to the real setup card instead of
 *     minting.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'
import chatReducer from '../store/chatSlice'
import dashboardReducer from '../store/dashboardSlice'

const mocks = vi.hoisted(() => ({
  tailnetMobile: vi.fn(),
  tailnetMobileQr: vi.fn(),
  mobileLoginLink: vi.fn(),
}))
vi.mock('../api/client', () => ({ api: mocks }))

import MobileConnectModal from './MobileConnectModal'

function mount(kinds: string[]) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const store = configureStore({ reducer: { chat: chatReducer, dashboard: dashboardReducer } })
  return render(
    <Provider store={store}>
      <QueryClientProvider client={qc}>
        <MemoryRouter>
          <MobileConnectModal kinds={kinds} onClose={() => {}} />
        </MemoryRouter>
      </QueryClientProvider>
    </Provider>,
  )
}

beforeEach(() => {
  mocks.tailnetMobile.mockReset()
  mocks.tailnetMobileQr.mockReset()
  mocks.mobileLoginLink.mockReset()
  mocks.tailnetMobile.mockResolvedValue({ step: 'ready' })
})

describe('MobileConnectModal', () => {
  it('never mints a credential on mount — QR appears only after the explicit click', async () => {
    mocks.tailnetMobileQr.mockResolvedValue({
      url: 'https://host/?token=live',
      image: 'data:image/png;base64,x',
    })
    mount(['tailnet_qr'])
    await waitFor(() => expect(screen.getByText('Show QR code')).toBeInTheDocument())
    expect(mocks.tailnetMobileQr).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText('Show QR code'))
    await waitFor(() =>
      expect(screen.getByAltText('QR code for mobile access')).toBeInTheDocument(),
    )
    expect(mocks.tailnetMobileQr).toHaveBeenCalledTimes(1)
  })

  it('not-ready tailnet routes to setup instead of offering a mint', async () => {
    mocks.tailnetMobile.mockResolvedValue({ step: 'publish' })
    mount(['tailnet_qr'])
    await waitFor(() =>
      expect(
        screen.getByText(/Remote access is not set up yet/),
      ).toBeInTheDocument(),
    )
    expect(screen.queryByText('Show QR code')).not.toBeInTheDocument()
  })

  it('login_link mints only on click and shows the one-time URL', async () => {
    mocks.mobileLoginLink.mockResolvedValue({ url: 'https://ext/?token=once', expires_in: 300 })
    mount(['login_link'])
    expect(mocks.mobileLoginLink).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Create sign-in link'))
    await waitFor(() =>
      expect(screen.getByDisplayValue('https://ext/?token=once')).toBeInTheDocument(),
    )
  })

  it('an unrecognised kind renders nothing (forward compat with edition methods)', () => {
    mount(['some-enterprise-kind'])
    // Header renders; neither known section's affordance does.
    expect(screen.getByText('Use Kiro Crew on your phone')).toBeInTheDocument()
    expect(screen.queryByText('Show QR code')).not.toBeInTheDocument()
    expect(screen.queryByText('Create sign-in link')).not.toBeInTheDocument()
    expect(mocks.tailnetMobile).not.toHaveBeenCalled()
  })
})
