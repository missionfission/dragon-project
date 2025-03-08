"use client"

import { useSearchParams } from "next/navigation"
import ChipDesigner from "../chip-designer"

export const DashboardContent = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <ChipDesigner />
    </div>
  )
} 