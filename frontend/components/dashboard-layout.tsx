"use client"

import { useState } from "react"
import { Sidebar, SidebarContent, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Cpu, Activity, FileJson } from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { useSearchParams } from "next/navigation"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const searchParams = useSearchParams()
  const currentView = searchParams.get("view") || "estimator"

  const menuItems = [
    {
      id: "estimator",
      label: "Performance Estimator",
      icon: Activity,
      href: "/dashboard?view=estimator"
    },
    {
      id: "designer",
      label: "Chip Designer",
      icon: Cpu,
      href: "/dashboard?view=designer"
    },
    {
      id: "api-docs",
      label: "API Documentation",
      icon: FileJson,
      href: "http://localhost:8000/docs",
      external: true
    }
  ]

  return (
    <SidebarProvider defaultOpen>
      <div className="flex h-screen bg-black">
        <Sidebar className="border-r border-gray-800" side="left">
          <SidebarContent>
            <div className="flex h-[60px] items-center px-6">
              <div className="flex items-center gap-2">
                <Cpu className="h-6 w-6 text-blue-500" />
                <span className="text-lg font-semibold text-white">Dragon Systems</span>
              </div>
              <SidebarTrigger className="ml-auto lg:hidden" />
            </div>
            <ScrollArea className="flex-1 px-4">
              <div className="space-y-2 py-4">
                {menuItems.map((item) => (
                  <Button
                    key={item.id}
                    variant="ghost"
                    className={cn(
                      "w-full justify-start gap-2",
                      currentView === item.id && "bg-gray-800 text-white"
                    )}
                    asChild
                  >
                    <Link 
                      href={item.href}
                      target={item.external ? "_blank" : undefined}
                      rel={item.external ? "noopener noreferrer" : undefined}
                    >
                      <item.icon className="h-4 w-4" />
                      {item.label}
                    </Link>
                  </Button>
                ))}
              </div>
            </ScrollArea>
          </SidebarContent>
        </Sidebar>
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </SidebarProvider>
  )
} 