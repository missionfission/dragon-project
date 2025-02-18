"use client"

import { useState } from 'react'
import { signInWithPopup, GoogleAuthProvider } from 'firebase/auth'
import { auth } from '@/lib/firebase'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useRouter } from 'next/navigation'
import { FaGoogle } from 'react-icons/fa'
import Image from 'next/image'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Performance comparison data
const performanceData = [
  { name: 'Baseline', performance: 100 },
  { name: 'Dragon Systems', performance: 285 },
]

// Testimonials data
const testimonials = [
  {
    quote: "Dragon Systems helped us achieve 3x performance improvement in our chip design process.",
    author: "Sarah Chen",
    role: "Chief Architect, TechCorp"
  },
  {
    quote: "The AI Architect feature saved us months of evaluation time.",
    author: "Michael Rodriguez",
    role: "VP Engineering, ChipWorks"
  }
]

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
        <div className="flex flex-col md:flex-row items-center gap-12">
          <div className="flex-1 text-left space-y-6">
            <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
              DRAGON SYSTEMS
            </h1>
            <p className="text-xl md:text-2xl text-gray-300">
              Revolutionizing Chip Design with AI-Powered Solutions
            </p>
            <div className="flex gap-4 mt-8">
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
          {/* <div className="flex-1">
            <Image 
              src="/acc.png"
              alt="Performance Comparison"
              width={600}
              height={400}
              className="rounded-lg shadow-2xl"
            />
          </div> */}
        </div>
      </div>

      {/* Performance Metrics Section */}
      <div className="container mx-auto px-4 py-16 bg-gray-800/30">
        <h2 className="text-3xl font-bold text-center mb-12">Performance Impact</h2>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis label={{ value: 'Performance (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="performance" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Solutions Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">Our Solutions</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/acc.png"
              alt="Performance Acceleration"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">Performance Estimation</h3>
            <p className="text-gray-400">
              Achieve up to 90% accuracy with significant speedup compared to traditional methods.
            </p>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/designspace.jpg"
              alt="Design Space Exploration"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">Design Space Exploration</h3>
            <p className="text-gray-400">
              Explore multiple technology options and design parameters to find optimal solutions.
            </p>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/system.png"
              alt="System Architecture"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">System Integration</h3>
            <p className="text-gray-400">
              Comprehensive system-level evaluation combining logic, memory, and interconnects.
            </p>
          </Card>
        </div>
      </div>

      {/* System Architecture Section */}
      <div className="container mx-auto px-4 py-16 bg-gray-800/30">
        <h2 className="text-3xl font-bold text-center mb-12">System Architecture</h2>
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div>
            <Image
              src="/framework-diagram.jpg"
              alt="Framework Architecture"
              width={600}
              height={400}
              className="rounded-lg shadow-2xl"
            />
            <p className="text-center text-gray-300 mt-4">Framework Architecture</p>
          </div>
          <div>
            <Image
              src="/symbolic-diagram.jpg"
              alt="Symbolic Execution"
              width={600}
              height={400}
              className="rounded-lg shadow-2xl"
            />
            <p className="text-center text-gray-300 mt-4">Symbolic Execution</p>
          </div>
        </div>
        <p className="text-center text-gray-300 mt-8 max-w-2xl mx-auto">
          Our integrated approach combines device-level parameters, advanced simulators, and system-level evaluation for comprehensive chip design optimization.
        </p>
      </div>

      {/* Testimonials Section */}
      <div className="container mx-auto px-4 py-20 bg-gray-800/30">
        <h2 className="text-4xl font-bold text-center mb-16">What Our Clients Say</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="p-8 bg-gray-800/50 border-gray-700">
              <div className="space-y-4">
                <p className="text-gray-300 italic mb-4">{testimonial.quote}</p>
                <div>
                  <p className="font-semibold">{testimonial.author}</p>
                  <p className="text-sm text-gray-400">{testimonial.role}</p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 text-center text-gray-400">
        © 2024 Dragon Systems. All rights reserved.
      </footer>
    </div>
  )
}