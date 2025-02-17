import { Suspense } from "react"
import DashboardLayout from "@/components/dashboard-layout"

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <Suspense fallback={<div>Loading...</div>}>
        <DashboardContent />
      </Suspense>
    </DashboardLayout>
  )
}

"use client"

import { useSearchParams } from "next/navigation"
import ChipDesigner from "../../chip-designer"
import ChipPerformanceEstimator from "@/components/chip-performance-estimator"

const DashboardContent = () => {
  const searchParams = useSearchParams()
  const view = searchParams.get("view") || "estimator"

  const renderContent = () => {
    switch (view) {
      case "designer":
        return <ChipDesigner />
      case "estimator":
      default:
        return <ChipPerformanceEstimator />
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {renderContent()}
    </div>
  )
} 