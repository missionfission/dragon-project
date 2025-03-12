"use client"

import { Suspense } from "react"
import DashboardLayout from "@/components/dashboard-layout"
import { DashboardContent } from "@/components/dashboard-content"
import { RunHeader } from "@/components/run-header"
import { AuthGuard } from "@/components/auth-guard"

export default function DashboardPage() {
  return (
    <AuthGuard>
      <DashboardLayout>
        <RunHeader />
        <Suspense fallback={<div className="flex items-center justify-center h-full">Loading...</div>}>
          <DashboardContent />
        </Suspense>
      </DashboardLayout>
    </AuthGuard>
  )
} 