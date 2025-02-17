import { Suspense } from "react"
import dynamic from "next/dynamic"
import DashboardLayout from "@/components/dashboard-layout"

const DashboardContent = dynamic(
  () => import("@/components/dashboard-content").then((mod) => mod.DashboardContent),
  {
    loading: () => (
      <div className="flex items-center justify-center h-full">Loading...</div>
    ),
    ssr: false
  }
)

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <DashboardContent />
    </DashboardLayout>
  )
} 