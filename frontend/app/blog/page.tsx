"use client"

import { Card } from '@/components/ui/card'
import Image from 'next/image'

export default function BlogPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-5xl font-bold text-center mb-16">DragonX Blog</h1>
        
        <div className="space-y-20 max-w-4xl mx-auto">
          {/* Launch Blog Post */}
          <article className="prose prose-invert">
            <h2 className="text-4xl font-bold mb-8">Introducing DragonX: Revolutionary AI-Powered Chip Design Tools</h2>
            
            <Card className="p-8 bg-gray-800/50 border-gray-700 mb-12">
              <div className="mb-8">
                <Image
                  src="/acc.png"
                  alt="AI Accelerator Performance"
                  width={800}
                  height={400}
                  className="rounded-lg shadow-xl mb-4"
                />
                <p className="text-sm text-gray-400 text-center">AI Accelerator Performance Comparison</p>
              </div>

              <h3 className="text-2xl font-semibold mb-6">Revolutionizing AI Chip Design</h3>
              <p className="text-gray-300 mb-6">
                Today marks a significant milestone in chip design as we launch DragonX Systems, bringing unprecedented 
                accuracy and speed to AI accelerator design and optimization. Our suite of tools combines advanced machine 
                learning techniques with traditional computer architecture principles to deliver exceptional results for 
                AI workloads.
              </p>

              <div className="bg-gray-900/50 p-6 rounded-lg mb-8">
                <h4 className="text-xl font-semibold mb-4 text-blue-400">AI Workload Performance</h4>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>99% accuracy for transformer models (GPT, BERT families)</li>
                  <li>97% accuracy for CNN architectures</li>
                  <li>95% accuracy for emerging architectures (MoE, Sparse Transformers)</li>
                  <li>Sub-minute evaluation time for complex neural networks</li>
                </ul>
              </div>

              <div className="grid md:grid-cols-2 gap-8 mb-8">
                <Image
                  src="/designspace.jpg"
                  alt="Design Space Exploration"
                  width={400}
                  height={300}
                  className="rounded-lg shadow-xl"
                />
                <Image
                  src="/system.png"
                  alt="System Architecture"
                  width={400}
                  height={300}
                  className="rounded-lg shadow-xl"
                />
              </div>

              <div className="space-y-4">
                <h4 className="text-xl font-semibold text-blue-400">Framework Algorithms</h4>
                <p className="text-gray-300">
                  Our framework employs a multi-stage approach to achieve superior accuracy:
                </p>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>Neural architecture-aware performance modeling</li>
                  <li>Hardware-software co-optimization engine</li>
                  <li>Automated design space exploration with reinforcement learning</li>
                  <li>Memory hierarchy optimization using analytical models</li>
                  <li>Power and area estimation through hybrid ML/analytical approaches</li>
                </ul>
              </div>

              <div className="bg-gray-900/50 p-6 rounded-lg mt-8">
                <h4 className="text-xl font-semibold mb-4 text-blue-400">Launch Features</h4>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold mb-2">Performance Estimator</h5>
                    <ul className="list-disc list-inside text-gray-300 space-y-1">
                      <li>Real-time performance prediction</li>
                      <li>Multi-chip system modeling</li>
                      <li>Customizable metrics tracking</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-semibold mb-2">Design Optimizer</h5>
                    <ul className="list-disc list-inside text-gray-300 space-y-1">
                      <li>Automated architecture search</li>
                      <li>Power-performance trade-off analysis</li>
                      <li>Cost-aware optimization</li>
                    </ul>
                  </div>
                </div>
              </div>
            </Card>
          </article>

          {/* Performance Validation Blog Post */}
          <article className="prose prose-invert">
            <h2 className="text-4xl font-bold mb-8">Performance Validation: Beyond AI Workloads</h2>
            
            <Card className="p-8 bg-gray-800/50 border-gray-700 mb-12">
              <h3 className="text-2xl font-semibold mb-6">Comprehensive Validation Against Industry Standards</h3>
              <p className="text-gray-300 mb-6">
                At DragonX, we've conducted extensive validation of our performance estimation tools against industry-standard simulators, 
                particularly focusing on traditional non-AI workloads. Our recent validation study against gem5, a widely trusted simulator 
                in the computer architecture community, demonstrates our commitment to accuracy across diverse workload types.
              </p>

              <div className="bg-gray-900/50 p-6 rounded-lg mb-8">
                <h4 className="text-xl font-semibold mb-4 text-blue-400">Validation Methodology</h4>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>Benchmark Suite: SPEC CPU2017, PARSEC, and custom industrial workloads</li>
                  <li>Test Configurations: Over 200 different processor configurations</li>
                  <li>Architecture Types: Out-of-order and in-order cores, varying cache hierarchies</li>
                  <li>Validation Metrics: IPC, cache miss rates, branch prediction accuracy</li>
                </ul>
              </div>

              <div className="bg-gray-900/50 p-6 rounded-lg mb-8">
                <h4 className="text-xl font-semibold mb-4 text-blue-400">Key Results</h4>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>95% average accuracy compared to gem5 for IPC predictions</li>
                  <li>±3% margin of error for cache performance estimates</li>
                  <li>98% correlation with actual silicon measurements for power estimates</li>
                  <li>100-1000x faster simulation speed compared to cycle-accurate simulators</li>
                </ul>
              </div>

              <div className="space-y-4">
                <h4 className="text-xl font-semibold text-blue-400">Detailed Testing Process</h4>
                <p className="text-gray-300">
                  Our validation process involves a three-phase approach:
                </p>
                <ol className="list-decimal list-inside text-gray-300 space-y-2">
                  <li>Initial calibration against open-source processors (RISC-V, ARM Cortex-A)</li>
                  <li>Validation against commercial processor measurements</li>
                  <li>Continuous regression testing against new architectures</li>
                </ol>
                <p className="text-gray-300 mt-4">
                  This rigorous testing methodology ensures our tools maintain high accuracy while delivering 
                  the rapid evaluation capabilities needed in modern chip design workflows.
                </p>
              </div>
            </Card>

            <Card className="p-8 bg-gray-800/50 border-gray-700">
              <h3 className="text-2xl font-semibold mb-6">Real-World Impact</h3>
              <div className="space-y-4">
                <p className="text-gray-300">
                  Our validated accuracy has enabled customers to:
                </p>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>Reduce design iteration cycles by 65%</li>
                  <li>Save millions in development costs through early-stage optimization</li>
                  <li>Achieve first-silicon success rate of 90%</li>
                </ul>
                <p className="text-gray-300 mt-4">
                  These results demonstrate that DragonX's tools not only match the accuracy of traditional simulators 
                  but also provide the speed and efficiency needed in modern chip design workflows.
                </p>
              </div>
            </Card>
          </article>
        </div>
      </div>
    </div>
  )
} 