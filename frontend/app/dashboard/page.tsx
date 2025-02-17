"use client"

import { Suspense } from "react"
import DashboardLayout from "@/components/dashboard-layout"
import { DashboardContent } from "@/components/dashboard-content"

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <Suspense fallback={<div className="flex items-center justify-center h-full">Loading...</div>}>
        <DashboardContent />
      </Suspense>
    </DashboardLayout>
  )
} 