"use client"

import { useSearchParams } from "next/navigation"
import ChipDesigner from "../chip-designer"
import ChipPerformanceEstimator from "@/components/chip-performance-estimator"

export const DashboardContent = () => {
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