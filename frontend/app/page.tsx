"use client"

import { useState } from 'react'
import { signInWithEmailAndPassword } from 'firebase/auth'
import { auth } from '@/lib/firebase'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useRouter } from 'next/navigation'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import Image from 'next/image'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Update the performance data section
const developmentMetrics = [
  { name: 'Development Time', reduction: 70 },
  { name: 'Design Iterations', reduction: 65 },
  { name: 'Development Costs', reduction: 60 },
]

const timeToMarketMetrics = [
  { name: 'Traditional Process', months: 24 },
  { name: 'With DragonX', months: 8 },
]

// Testimonials data
const testimonials = [
  {
    quote: "Dragon Systems cut our chip design evaluation time by 70%, saving us millions in development costs.",
    role: "Chief Architect, Fujitsu Research"
  },
  {
    quote: "Whether it's AI accelerators or traditional processors, Dragon Systems consistently delivers accurate performance predictions.",
    role: "TSMC Novel Computing Group"
  }
  // ,
  // {
  //   quote: "The rapid architecture evaluation helped us identify optimal designs in weeks instead of months.",
  //   role: "Chief Architect, Qualcomm"
  // }
]

export default function HomePage() {
  const [loading, setLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const router = useRouter()

  const handleSignIn = async () => {
    setLoading(true)
    setError('')
    try {
      await signInWithEmailAndPassword(auth, email, password)
      router.push('/dashboard')
    } catch (error: any) {
      console.error('Error signing in:', error)
      setError(error.message || 'Failed to sign in')
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
              DRAGONX SYSTEMS
            </h1>
            <p className="text-xl md:text-2xl text-gray-300">
              Accelerate Chip Design Decisions with AI-Powered Architecture Evaluation
            </p>
            <div className="space-y-4 text-gray-300">
              <div className="flex items-center gap-2">
                <span className="text-blue-500">✓</span>
                <p>70% faster architecture evaluation time</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-500">✓</span>
                <p>90% accuracy for both AI and traditional workloads</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-500">✓</span>
                <p>Reduce development costs by up to 60%</p>
              </div>
            </div>
            <Card className="p-6 bg-gray-800/50 border-gray-700 max-w-md">
              <form onSubmit={(e) => { e.preventDefault(); handleSignIn(); }} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    required
                  />
                </div>
                {error && (
                  <div className="text-red-500 text-sm">{error}</div>
                )}
                <Button 
                  type="submit"
                  className="w-full bg-blue-500 hover:bg-blue-600"
                  disabled={loading}
                >
                  {loading ? 'Signing in...' : 'Sign in'}
                </Button>
              </form>
            </Card>
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
        <h2 className="text-3xl font-bold text-center mb-12">Our Impact</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <Card className="p-6 bg-gray-800/50 border-gray-700">
            <h3 className="text-xl font-semibold mb-6 text-center">Development Improvements</h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={developmentMetrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#4B5563' }}
                  />
                  <YAxis 
                    label={{ 
                      value: 'Reduction (%)', 
                      angle: -90, 
                      position: 'insideLeft',
                      fill: '#9CA3AF'
                    }}
                    tick={{ fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#4B5563' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#fff'
                    }}
                  />
                  <Bar 
                    dataKey="reduction" 
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={80}
                  >
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="p-6 bg-gray-800/50 border-gray-700">
            <h3 className="text-xl font-semibold mb-6 text-center">Time to Market Comparison</h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={timeToMarketMetrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#4B5563' }}
                  />
                  <YAxis 
                    label={{ 
                      value: 'Time to Market (Months)', 
                      angle: -90, 
                      position: 'insideLeft',
                      fill: '#9CA3AF'
                    }}
                    tick={{ fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#4B5563' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#fff'
                    }}
                  />
                  <Bar 
                    dataKey="months" 
                    fill="#8b5cf6"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={80}
                  >
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </div>

      {/* Products Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">Our Products</h2>
        <div className="grid md:grid-cols-2 gap-12">
          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <div className="flex flex-col h-full">
              <h3 className="text-2xl font-semibold mb-4 text-blue-400">DragonX Performance Estimator</h3>
              <p className="text-gray-300 mb-6">
                Fast and accurate performance estimation for chip designs. Evaluate your architecture decisions in minutes instead of months.
              </p>
              <div className="bg-gray-900/50 p-4 rounded-lg mb-6">
                <h4 className="font-semibold mb-2 text-blue-300">Key Features:</h4>
                <ul className="space-y-2 text-gray-300">
                  <li>• Rapid performance modeling</li>
                  <li>• Power and area estimates</li>
                  <li>• Support for AI and traditional workloads</li>
                  <li>• Cloud-based simulation engine</li>
                </ul>
              </div>
              <div className="mt-auto">
                <Button 
                  className="w-full bg-blue-500 hover:bg-blue-600"
                  onClick={() => router.push('/dashboard')}
                >
                  Try Performance Estimator
                </Button>
              </div>
            </div>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <div className="flex flex-col h-full">
              <h3 className="text-2xl font-semibold mb-4 text-purple-400">DragonX Chip Designer</h3>
              <p className="text-gray-300 mb-6">
                AI-powered chip design optimization platform. Automatically explore design spaces and find optimal configurations.
              </p>
              <div className="bg-gray-900/50 p-4 rounded-lg mb-6">
                <h4 className="font-semibold mb-2 text-purple-300">Key Features:</h4>
                <ul className="space-y-2 text-gray-300">
                  <li>• Automated design space exploration</li>
                  <li>• Multi-objective optimization</li>
                  <li>• Cost-performance trade-off analysis</li>
                  <li>• Integration with major EDA tools</li>
                </ul>
              </div>
              <div className="mt-auto">
                <Button 
                  className="w-full bg-purple-500 hover:bg-purple-600"
                  onClick={() => router.push('/dashboard')}
                >
                  Try Chip Designer
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Solutions Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">Comprehensive Solutions</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/acc.png"
              alt="Performance Acceleration"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">Rapid Architecture Evaluation</h3>
            <p className="text-gray-400">
              Evaluate chip architectures 70% faster with our AI-powered platform. Support for both neural networks and traditional workloads.
            </p>
            <ul className="mt-4 space-y-2 text-gray-400">
              <li>• Multi-scenario performance modeling</li>
              <li>• Power and area estimation</li>
              <li>• Workload-specific optimization</li>
            </ul>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/designspace.jpg"
              alt="Design Space Exploration"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">Intelligent Design Space Exploration</h3>
            <p className="text-gray-400">
              Automatically explore and evaluate thousands of design options to find the optimal solution for your requirements.
            </p>
            <ul className="mt-4 space-y-2 text-gray-400">
              <li>• Technology node comparison</li>
              <li>• Memory hierarchy optimization</li>
              <li>• Cost-performance trade-offs</li>
            </ul>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 transform hover:scale-105 transition-transform">
            <Image
              src="/system.png"
              alt="System Architecture"
              width={200}
              height={200}
              className="mx-auto mb-6"
            />
            <h3 className="text-xl font-semibold mb-4">ROI-Focused Integration</h3>
            <p className="text-gray-400">
              Comprehensive system-level evaluation that helps reduce development costs and time-to-market.
            </p>
            <ul className="mt-4 space-y-2 text-gray-400">
              <li>• Early-stage cost estimation</li>
              <li>• Time-to-market optimization</li>
              <li>• Risk assessment and mitigation</li>
            </ul>
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

      {/* Blog Preview Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">Latest from Our Blog</h2>
        
        <div className="max-w-4xl mx-auto space-y-8">
          <Card className="p-8 bg-gray-800/50 border-gray-700 hover:bg-gray-800/70 transition-colors cursor-pointer" onClick={() => router.push('/blog')}>
            <h3 className="text-2xl font-semibold mb-4">Introducing DragonX: Revolutionary AI-Powered Chip Design Tools</h3>
            <div className="mb-6">
              <Image
                src="/acc.png"
                alt="AI Accelerator Performance"
                width={800}
                height={400}
                className="rounded-lg shadow-xl mb-4"
              />
            </div>
            <p className="text-gray-300 mb-6">
              Announcing the launch of DragonX Systems with industry-leading 99% accuracy for transformer models and 
              97% for CNN architectures. Our AI-powered tools deliver unprecedented speed and precision in chip design optimization.
            </p>
            <div className="flex items-center text-blue-400 hover:text-blue-300">
              Read full article
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </Card>

          <Card className="p-8 bg-gray-800/50 border-gray-700 hover:bg-gray-800/70 transition-colors cursor-pointer" onClick={() => router.push('/blog')}>
            <h3 className="text-2xl font-semibold mb-4">Performance Validation: Beyond AI Workloads</h3>
            <p className="text-gray-300 mb-6">
              Discover how DragonX achieves 95% accuracy compared to industry-standard simulators like gem5, while delivering 
              100-1000x faster simulation speeds. Our comprehensive validation study covers SPEC CPU2017, PARSEC, and various 
              industrial workloads across 200+ processor configurations.
            </p>
            <div className="flex items-center text-blue-400 hover:text-blue-300">
              Read full article
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </Card>
        </div>
      </div>

      {/* Code & Library Support Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">Developer Tools & Library Support</h2>
        <div className="grid md:grid-cols-2 gap-12">
          {/* Python SDK Card */}
          <Card className="p-8 bg-gray-800/50 border-gray-700">
            <h3 className="text-2xl font-semibold mb-6">Python SDK & Libraries</h3>
            <div className="space-y-4">
              <div className="bg-gray-900/50 p-4 rounded-lg">
                <code className="text-sm text-gray-300">
                  pip install dragonx-optimizer==0.1.1
                </code>
              </div>
              <p className="text-gray-300">Comprehensive Python libraries for:</p>
              <ul className="list-disc list-inside text-gray-300 space-y-2">
                <li>Workload analysis and profiling</li>
                <li>Neural network optimization</li>
                <li>Auto-tuning and parameter optimization</li>
                <li>Performance prediction and modeling</li>
              </ul>
            </div>
          </Card>

          {/* Compiler Optimizations Card */}
          <Card className="p-8 bg-gray-800/50 border-gray-700">
            <h3 className="text-2xl font-semibold mb-6">Compiler Optimizations (Coming Soon!)</h3>
            <div className="space-y-4">
              <div className="bg-gray-900/50 p-4 rounded-lg">
                <code className="text-sm text-gray-300">
                  dragonx-compile --target=accelerator --opt-level=3 workload.py
                </code>
              </div>
              <p className="text-gray-300">Advanced compilation features:</p>
              <ul className="list-disc list-inside text-gray-300 space-y-2">
                <li>Hardware-specific code generation</li>
                <li>Automatic vectorization and parallelization</li>
                <li>Memory access pattern optimization</li>
                <li>Dynamic runtime adaptation</li>
              </ul>
            </div>
          </Card>
        </div>

        {/* Code Example Section */}
        <div className="mt-12">
          <Card className="p-8 bg-gray-800/50 border-gray-700">
            <h3 className="text-2xl font-semibold mb-6">Quick Start Example</h3>
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <pre className="text-sm text-gray-300">
                <code>{`import dragonx_optimizer.src_main as dx

# Initialize optimizer with architecture config
optimizer = dx.initialize(arch_config="custom_accelerator.yaml")

# Analyze workload
graph = dx.analyze_workload(model)

# Optimize design for target metrics
optimized_config = dx.optimize_design(
    graph,
    target_metrics={
        "latency": "minimal",
        "power": "<5W"
    }
)

# Get performance estimates
perf_stats = dx.estimate_performance(graph, optimized_config)`}</code>
              </pre>
            </div>
          </Card>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="container mx-auto px-4 py-20 bg-gray-800/30">
        <h2 className="text-4xl font-bold text-center mb-16">What Our Clients Say</h2>
        {/* <div className="grid md:grid-cols-2 gap-8">
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
          ))} */}
        {/* </div> */}
      </div>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 text-center text-gray-400">
        © 2024 Dragon Systems. All rights reserved.
      </footer>
    </div>
  )
}