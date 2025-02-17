import { Suspense } from "react"
import DashboardLayout from "@/components/dashboard-layout"
import { DashboardContent } from "@/components/dashboard-content"

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <Suspense fallback={<div>Loading...</div>}>
        <DashboardContent />
      </Suspense>
    </DashboardLayout>
  )
} 