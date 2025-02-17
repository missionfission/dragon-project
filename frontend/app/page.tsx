"use client"

import { useState } from 'react'
import { signInWithPopup, GoogleAuthProvider } from 'firebase/auth'
import { auth } from '@/lib/firebase'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useRouter } from 'next/navigation'
import { FaGoogle } from 'react-icons/fa'

export default function HomePage() {
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleGoogleSignIn = async () => {
    setLoading(true)
    try {
      const provider = new GoogleAuthProvider()
      await signInWithPopup(auth, provider)
      router.push('/dashboard')
    } catch (error) {
      console.error('Error signing in with Google:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center space-y-6">
          <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
            DRAGON SYSTEMS
          </h1>
          <p className="text-xl md:text-2xl text-gray-300">
            Full Stack Chip Design Solutions
          </p>
          <div className="flex justify-center gap-4 mt-8">
            <Button 
              size="lg"
              onClick={handleGoogleSignIn}
              disabled={loading}
              className="bg-white text-black hover:bg-gray-200 flex items-center gap-2"
            >
              <FaGoogle className="w-5 h-5" />
              Sign in with Google
            </Button>
          </div>
        </div>
      </div>

      {/* Solutions Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="space-y-24">
          {/* Solution 1 */}
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-4 mb-6">
              <span className="text-4xl font-bold text-blue-500">01</span>
              <h2 className="text-3xl font-bold">Optimize Workloads</h2>
            </div>
            <Card className="p-8 bg-gray-800/50 border-gray-700">
              <h3 className="text-xl font-semibold mb-4">
                Optimize Workloads to run faster on your Custom Chip
              </h3>
              <p className="text-gray-400">
                Configure your compilers with best-in-class software optimizers for maximum performance.
              </p>
            </Card>
          </div>

          {/* Solution 2 */}
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-4 mb-6">
              <span className="text-4xl font-bold text-blue-500">02</span>
              <h2 className="text-3xl font-bold">AI Architect</h2>
            </div>
            <Card className="p-8 bg-gray-800/50 border-gray-700">
              <div className="space-y-4">
                <p className="text-gray-300">- Evaluate Your Micro-Architecture</p>
                <p className="text-gray-300">
                  - Explore a large design space for evaluating Architecture Choices, in seconds not months
                </p>
              </div>
            </Card>
          </div>

          {/* Solution 3 */}
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-4 mb-6">
              <span className="text-4xl font-bold text-blue-500">03</span>
              <h2 className="text-3xl font-bold">Technology Derivations</h2>
            </div>
            <Card className="p-8 bg-gray-800/50 border-gray-700">
              <div className="space-y-4">
                <p className="text-gray-300">- Evaluate different foundry technology choices</p>
                <p className="text-gray-300">
                  - Configure for the best choices of Packaging, Thermal Conduits, and Connectivity Integration
                </p>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 text-center text-gray-400">
        © 2024 Dragon Systems. All rights reserved.
      </footer>
    </div>
  )
}