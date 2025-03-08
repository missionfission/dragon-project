import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { useRouter } from "next/navigation"

export function MainNav() {
  const router = useRouter()

  return (
    <div className="flex items-center justify-between w-full">
      <nav className="flex items-center space-x-4 lg:space-x-6">
        <Link
          href="/runs/active"
          className="text-sm font-medium transition-colors hover:text-primary"
        >
          Current Run
        </Link>
        <Link
          href="/runs"
          className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
        >
          Run History
        </Link>
      </nav>

      <Button 
        onClick={() => router.push('/runs/new')}
        className="flex items-center gap-2"
      >
        <Plus className="w-4 h-4" />
        New Run
      </Button>
    </div>
  )
} 