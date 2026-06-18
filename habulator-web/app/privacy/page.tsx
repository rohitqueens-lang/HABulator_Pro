import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import Header from '@/components/Header'

export const metadata = {
  title: 'Privacy · HABulator',
  description: 'How the HABulator research tool handles information.',
}

// Privacy notice reflecting the application's actual behavior: no accounts, cookies,
// analytics, or storage; entered values are used only to compute a result and are not
// retained. Hosting providers and standard server logs process limited connection data
// (e.g. IP address) for delivery and security, as disclosed below.
export default function PrivacyPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="mb-6 inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-400 transition-colors hover:text-ink-200"
        >
          <ArrowLeft size={14} /> Back to predictor
        </Link>

        <h1 className="font-display text-2xl font-semibold text-ink-100">Privacy Notice</h1>
        <p className="mt-1 text-[12px] text-ink-500">Last updated: June 2026</p>

        <div className="mt-6 flex flex-col gap-6 text-[14px] leading-relaxed text-ink-300">
          <p>
            HABulator is a non-commercial research tool that estimates Great Lakes phytoplankton
            biovolume from environmental parameters. This notice describes what information the
            application does and does not handle.
          </p>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">No personal information</h2>
            <p>
              The application does not require an account and does not ask for, collect, or store any
              personal information. It sets no cookies and uses no analytics, advertising, or
              third-party tracking technologies.
            </p>
          </section>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">The values you enter</h2>
            <p>
              The environmental parameters you set (temperature, nutrients, station depth, day of
              year) are transmitted to the prediction service only to compute a result for your
              current request. They are numeric environmental values, are not linked to your
              identity, and are not stored by the application after the result is returned.
            </p>
          </section>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">Hosting and technical data</h2>
            <p>
              The site is served by third-party hosting providers — the web interface via Vercel and
              the prediction service via Render. As with any website, these providers and standard
              server logs may automatically process limited technical connection data, such as your
              IP address, browser type, and request time, for the purpose of delivering the service
              and maintaining its security and reliability. This data is not used by us to identify
              you and is handled in accordance with those providers&apos; own privacy policies.
            </p>
          </section>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">Data sources</h2>
            <p>
              The underlying model was trained on publicly available U.S. EPA Great Lakes National
              Program Office (GLNPO / GLENDA) monitoring data (2001–2021). No personal data is
              involved in the model or its predictions.
            </p>
          </section>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">Changes to this notice</h2>
            <p>
              We may update this notice as the tool evolves. Any changes will be reflected on this
              page with a revised &ldquo;last updated&rdquo; date.
            </p>
          </section>

          <section>
            <h2 className="mb-1.5 text-[15px] font-semibold text-ink-100">Contact</h2>
            <p>
              Questions about this notice or the tool can be directed to the project team.
            </p>
          </section>
        </div>
      </main>

      <footer className="mt-6 border-t border-line py-6">
        <p className="mx-auto max-w-7xl px-4 text-center text-[11px] text-ink-400 sm:px-6 lg:px-8">
          © 2026 Rohit Shukla
        </p>
      </footer>
    </div>
  )
}
