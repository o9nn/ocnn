# OCNN Repository Agent Guide

## Repository Overview

This repository (`o9nn/ocnn`) is a fork of the Torch neural network package (`torch/nn`) that has been significantly extended with **OpenCog cognitive architecture** implementations and a revolutionary **Inferno AGI Operating System**. It provides neural network modules for building artificial general intelligence (AGI) systems with reasoning, attention allocation, and knowledge representation.

### Key Technologies
- **Language**: Lua (with C/C++ backend for performance-critical operations)
- **Framework**: Torch7 deep learning framework
- **Build System**: CMake + LuaRocks
- **Testing**: Custom Lua test framework (`nn.test()`)
- **CI/CD**: Travis CI

## Repository Structure

```
/
├── *.lua                          # 200+ neural network modules
├── OpenCog*.lua                   # OpenCog cognitive architecture (5 modules)
├── Inferno*.lua                   # Inferno OS kernel components (6 modules)
├── init.lua                       # Main module loader
├── test.lua                       # Comprehensive test suite (~9K lines)
├── test_opencog.lua              # OpenCog module tests
├── test_inferno_os.lua           # Inferno OS tests
├── doc/                          # Documentation (markdown)
│   ├── opencog.md                # OpenCog implementation docs
│   ├── module.md                 # Module API documentation
│   └── *.md                      # Various component docs
├── examples/
│   ├── opencog_example.lua       # OpenCog usage examples
│   └── inferno_os_example.lua    # Inferno OS demo
├── lib/THNN/                     # C/C++ backend (Torch THNN)
├── rocks/                        # LuaRocks package spec
└── CMakeLists.txt               # Build configuration
```

## Core Components

### 1. Traditional Neural Network Modules
The base `nn` package provides standard neural network building blocks:
- **Containers**: Sequential, Parallel, Concat, ConcatTable
- **Layers**: Linear, Convolution (Spatial/Volumetric/Temporal), LookupTable
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, etc.
- **Criterions**: MSE, NLL, CrossEntropy, etc.
- **Utilities**: BatchNormalization, Dropout, etc.

### 2. OpenCog Cognitive Architecture (NEW)
Five fully differentiable modules implementing OpenCog AGI:

#### `nn.OpenCogAtom` (~210 lines)
- Represents basic knowledge units with embeddings
- Implements STI (Short Term Importance) and LTI (Long Term Importance)
- Includes truth values (strength, confidence) for probabilistic logic
- All parameters are learnable via backpropagation

#### `nn.OpenCogAtomSpace` (~318 lines)
- Hypergraph database for storing atom collections
- Dynamic memory management with LTI-based forgetting
- Attention-weighted retrieval using STI
- Supports multiple atom types and similarity queries

#### `nn.OpenCogAttentionAllocation` (~249 lines)
- Economic attention dynamics (rent/wage mechanisms)
- STI decay and LTI promotion
- Attentional focus boundary management
- Hebbian learning integration

#### `nn.OpenCogPLN` (~344 lines)
- Probabilistic Logic Networks implementation
- Inference rules: deduction, induction, abduction, revision
- Differentiable reasoning operations
- Truth value composition and propagation

#### `nn.OpenCogNetwork` (~321 lines)
- Complete integrated cognitive architecture
- Perception → Attention → Reasoning → Action cycle
- Knowledge base management
- Goal-directed behavior

### 3. Inferno AGI Operating System (REVOLUTIONARY)
Six modules creating an OS where cognition is a kernel service:

#### `nn.InfernoKernel` (~422 lines)
- Core kernel with cognitive syscalls (THINK, REASON, REMEMBER, etc.)
- Process table for cognitive processes
- Kernel memory heap management
- Interrupt handlers and statistics

#### `nn.InfernoProcessScheduler` (~433 lines)
- Priority-based cognitive process scheduling
- Time-slicing and preemption
- Process migration for distributed AGI
- Deadlock detection and load balancing

#### `nn.InfernoMemoryManager` (~461 lines)
- Hierarchical memory: sensory → working → episodic → semantic
- Importance-based eviction policies
- Memory consolidation mechanisms
- Virtual memory with page tables

#### `nn.InfernoMessagePassing` (~381 lines)
- Inter-process communication channels
- Synchronous/asynchronous messaging
- Pub/sub topic-based routing
- Distributed message passing across nodes

#### `nn.InfernoFileSystem` (~510 lines)
- Knowledge represented as filesystem
- Directories: /concepts, /memories, /goals, /reasoning
- Standard file operations: create, read, write, search
- Hierarchical organization with importance metadata

#### `nn.InfernoDeviceDriver` (~397 lines)
- Perception/action as device I/O
- Devices: /dev/eyes, /dev/motor, /dev/attention, etc.
- Device read/write/ioctl operations
- Interrupt-driven I/O

#### `nn.OpenCogInfernoOS` (~472 lines)
- Main AGI OS integration
- Boots all kernel components
- Integrates OpenCog at kernel level
- Complete cognitive cycles with learning

## Code Conventions

### Lua Module Pattern
```lua
local ModuleName, parent = torch.class('nn.ModuleName', 'nn.Module')

function ModuleName:__init(config)
   parent.__init(self)
   -- initialization
end

function ModuleName:updateOutput(input)
   -- forward pass
   return self.output
end

function ModuleName:updateGradInput(input, gradOutput)
   -- backward pass
   return self.gradInput
end

function ModuleName:accGradParameters(input, gradOutput, scale)
   -- accumulate parameter gradients (if module has parameters)
end

function ModuleName:parameters()
   -- return parameters and gradients (if any)
   return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
end
```

### Naming Conventions
- **CamelCase**: Module names and class names
- **lowercase**: function names, variables
- **self.property**: instance variables
- **self.gradProperty**: gradient buffers
- **config**: common parameter name for configuration tables

### Testing Pattern
```lua
-- In test.lua or test_*.lua
local mytest = torch.TestSuite()
local tester = torch.Tester()

function mytest.ModuleName()
   local module = nn.ModuleName(params)
   local input = torch.randn(...)
   local output = module:forward(input)
   tester:asserteq(output:size(), expected_size)
   -- more assertions
end

tester:add(mytest)
```

## Building and Testing

### Build from Source
```bash
# Install dependencies
luarocks install torch
luarocks install luaffi
luarocks install moses

# Build the package
git clone <repo>
cd ocnn
luarocks make rocks/nn-scm-1.rockspec
```

### Running Tests
```bash
# All tests
th -lnn -e "nn.test()"

# OpenCog tests
th test_opencog.lua

# Inferno OS tests  
th test_inferno_os.lua

# Single module test
th -lnn -e "t=nn.test{'Linear'}; if t.errors[1] then os.exit(1) end"
```

### Running Examples
```bash
th -lnn -e "dofile('examples/opencog_example.lua')"
th -lnn -e "dofile('examples/inferno_os_example.lua')"
```

## Development Guidelines

### Adding New Modules
1. **Create module file**: `NewModule.lua` with proper class inheritance
2. **Add to init.lua**: `require('nn.NewModule')`
3. **Write tests**: Add test cases to `test.lua` or separate test file
4. **Document**: Add documentation to `doc/` directory
5. **Add examples**: Include usage examples

### For OpenCog/Inferno Modules
- Maintain differentiability for all operations
- Follow `nn.Module` patterns strictly
- Consider memory efficiency and batching
- Use proper tensor operations (avoid loops when possible)
- Ensure gradient flow is correct
- Add comprehensive tests for cognitive operations

### Code Quality
- **Use luacheck**: Run `.luacheckrc` linting
- **Test thoroughly**: Add unit tests and integration tests
- **Document**: Clear docstrings and inline comments for complex logic
- **Batch support**: All modules should support batch processing
- **Type checking**: Use `torch.typename()` for type validation

## Key Implementation Details

### OpenCog Components
- **Atoms** are learned embeddings with importance/truth metadata
- **AtomSpace** uses tensor operations for efficient batch retrieval
- **Attention** implements ECAN (Economic Attention Networks)
- **PLN** uses differentiable truth value formulas
- **All cognitive operations** are implemented as forward/backward passes

### Inferno OS Architecture
- **Syscalls** are implemented as forward passes on cognitive tensors
- **Processes** are represented as rows in process table tensors
- **Memory** is organized hierarchically with importance-based eviction
- **Filesystem** uses embeddings for paths and content
- **Everything is differentiable** - you can backprop through the entire OS

### Performance Considerations
- Use in-place operations where possible (`:copy()`, `:add()`, etc.)
- Prefer tensor operations over loops
- Batch operations for efficiency
- Memory management is crucial (AtomSpace auto-forgets)
- Attention mechanisms limit active processing to focused atoms

## Common Tasks

### Adding a New Neural Layer
1. Create `NewLayer.lua` with `:updateOutput()` and `:updateGradInput()`
2. Add to `init.lua`
3. Add tests in `test.lua`

### Extending OpenCog
1. Study existing OpenCog modules for patterns
2. Maintain truth value and attention value consistency
3. Ensure PLN rules are probabilistically sound
4. Test cognitive cycle integration

### Debugging
```lua
-- Enable verbose mode
module:verbose()

-- Check gradients
require 'nn'
local jacobian = nn.Jacobian
jacobian.testJacobian(module, input)

-- Inspect state
print(module)
print(module.output)
```

## Important Files

- **init.lua**: Module registry - ALL modules must be required here
- **Module.lua**: Base class for all modules (~430 lines)
- **test.lua**: Main test suite with 200+ tests
- **README_OPENCOG.md**: Comprehensive OpenCog/Inferno documentation
- **doc/opencog.md**: Detailed API documentation for cognitive modules

## Integration Points

### With Standard nn Modules
```lua
-- OpenCog networks can be used like any nn module
local model = nn.Sequential()
   :add(nn.Linear(784, 256))
   :add(nn.ReLU())
   :add(nn.OpenCogNetwork(config))
   :add(nn.LogSoftMax())
```

### With Training
```lua
-- Standard criterion and training work seamlessly
local criterion = nn.MSECriterion()
local sgd = nn.StochasticGradient(model, criterion)
sgd:train(dataset)
```

## Special Considerations

### For AI Agents Working on This Codebase

1. **Differentiability is Sacred**: Never break the gradient flow in OpenCog/Inferno modules
2. **Torch Tensor Operations**: Always prefer tensor ops over loops
3. **Batch First**: All inputs are `[batch, ...]` shaped
4. **Memory Management**: OpenCog modules auto-manage memory - don't interfere unless fixing bugs
5. **Testing is Critical**: The test suite is comprehensive - don't break existing tests
6. **Documentation**: Update both code comments and markdown docs
7. **C Backend**: THNN operations in `lib/THNN/` are performance-critical
8. **Cognitive Semantics**: Understand AGI concepts (atoms, attention, reasoning) before modifying
9. **OS Principles**: Inferno modules follow OS design patterns (processes, memory, I/O)
10. **Backwards Compatibility**: This is a fork - maintain compatibility where possible

## Research Context

This implementation enables:
- **Neuro-symbolic AI**: Combining neural networks with symbolic reasoning
- **AGI Research**: Full cognitive architecture experiments
- **Novel Architectures**: Attention mechanisms and memory systems
- **OS-level Intelligence**: Cognition as a fundamental system service
- **Distributed AGI**: Multi-node cognitive systems

## Dependencies

- `torch >= 7.0`
- `luaffi` (for FFI bindings)
- `moses >= 1.0` (Lua utility library)
- CMake (build)
- C compiler (gcc/clang for THNN backend)

## References

- Original nn: https://github.com/torch/nn
- OpenCog: http://opencog.org/
- Torch: https://github.com/torch/torch7
- Inferno OS inspiration: https://en.wikipedia.org/wiki/Inferno_(operating_system)

---

**Note for AI Agents**: This is a complex, research-oriented codebase combining deep learning with cognitive architectures. When making changes:
- Understand the cognitive semantics before modifying OpenCog/Inferno modules
- Test extensively - cognitive behaviors can be subtle
- Maintain differentiability and gradient flow
- Respect the OS-level abstractions in Inferno components
- Document cognitive design decisions clearly
