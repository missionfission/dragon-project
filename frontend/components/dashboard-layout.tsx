import { Suspense } from "react"
import dynamic from "next/dynamic"
import { DashboardSidebar } from "./dashboard-sidebar"

// Import SidebarProvider dynamically to avoid SSR issues
const SidebarProvider = dynamic(
  () => import("@/components/ui/sidebar").then((mod) => mod.SidebarProvider),
  { ssr: false }
)

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <SidebarProvider defaultOpen>
      <div className="flex h-screen bg-black">
        <Suspense fallback={<div className="w-64 bg-black" />}>
          <DashboardSidebar />
        </Suspense>
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </SidebarProvider>
  )
} 