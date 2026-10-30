import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Auralis",
  description: "Audio-Guided Vision Escalation Harness",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-6 py-4">
          <div className="flex items-center gap-4">
            <span className="bg-gradient-to-r from-brand-cyan to-brand-primary bg-clip-text text-3xl font-bold tracking-tight text-transparent">Auralis</span>
            <span className="hidden h-8 w-px bg-line sm:block" />
            <span className="hidden text-lg font-semibold text-brand-muted sm:block">Audio-Guided Vision Escalation Harness</span>
          </div>
          <div className="flex items-center gap-3 text-sm font-medium">
            <span className="flex items-center gap-2 rounded-full border border-brand-mint/50 bg-brand-mint/10 px-4 py-2 text-brand-mint">
              <span className="size-2.5 rounded-full bg-brand-mint" />Local AI Active
            </span>
            <span className="rounded-full border border-line bg-surface-2 px-4 py-2 text-brand-muted">&lt;/&gt; WebAssembly</span>
          </div>
        </header>
        <div className="flex flex-1 flex-col md:flex-row">
          <aside className="flex shrink-0 flex-col justify-between gap-6 border-b border-line p-3 md:w-56 md:border-r md:border-b-0">
            <nav className="flex gap-2 md:flex-col">
              <a href="#" className="rounded-xl bg-gradient-to-r from-brand-primary/60 to-brand-primary/20 px-4 py-3 font-semibold text-white">Live View</a>
              <a href="#" className="rounded-xl px-4 py-3 font-medium text-brand-muted hover:text-white">About</a>
            </nav>
            <div className="hidden rounded-xl border border-line bg-surface p-4 md:block">
              <p className="font-semibold leading-snug text-brand-mint">Lower Costs. Smarter Escalation. Auralis.</p>
              <p className="mt-3 text-sm text-brand-muted">Keep the intelligence local. Send to the cloud only when it matters.</p>
            </div>
          </aside>
          <div className="min-w-0 flex-1">{children}</div>
        </div>
      </body>
    </html>
  );
}
