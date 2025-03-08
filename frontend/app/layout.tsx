import { Inter } from 'next/font/google'
import './globals.css'
import { RunProvider } from '@/lib/runs'
import { Header } from '@/components/header'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <RunProvider>
          <Header />
          <main>
            {children}
          </main>
        </RunProvider>
      </body>
    </html>
  )
}
