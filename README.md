# DragonX

DragonX is a hardware-software co-design framework for AI accelerators. It provides comprehensive tools for workload analysis, neural network optimization, auto-tuning, and performance modeling.

```
pip install dragonx-optimizer==0.1.1
```
## Quick Start


```
import src_main as dx
```
Initialize optimizer with architecture config
```
optimizer = dx.initialize(arch_config="custom_accelerator.yaml")
```
Analyze workload
```
graph = dx.analyze_workload(model)
```
Optimize design for target metrics
```
optimized_config = dx.optimize_design(
graph,
target_metrics={
"latency": "minimal",
"power": "<5W"
}
)
```
Get performance estimates
```
perf_stats = dx.estimate_performance(graph, optimized_config)
```
## Installation
## Features

- Workload analysis and profiling
- Neural network optimization
- Auto-tuning and parameter optimization  
- Performance prediction and modeling
- Compiler optimizations

## Documentation

See [docs/](docs/) for detailed documentation and API reference.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.



