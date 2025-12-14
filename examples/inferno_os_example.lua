--[[
OpenCog Inferno OS Example
===========================

Demonstrates the revolutionary AGI operating system where cognitive processing
is a fundamental kernel service.

This example shows:
1. Booting the Inferno AGI OS
2. Making system calls for cognitive operations
3. Process scheduling and management
4. Hierarchical memory management
5. Message passing between cognitive processes
6. Knowledge as a filesystem
7. Device I/O for perception and action
8. Complete cognitive cycles with learning

Usage:
  th -lnn -e "dofile('examples/inferno_os_example.lua')"
]]

require('nn')

print("=" .. string.rep("=", 60))
print("OpenCog Inferno AGI Operating System Demo")
print("=" .. string.rep("=", 60))
print()

-- Configuration for the AGI OS
local config = {
   maxProcesses = 128,
   cognitiveResourceSize = 32,
   thoughtVectorSize = 64,
   perceptionSize = 128,
   actionSize = 64,
   heapSize = 1024,
   sensoryBufferSize = 64,
   workingMemorySize = 256,
   episodicMemorySize = 512,
   semanticMemorySize = 2048,
   maxChannels = 128,
   maxInodes = 2048,
   schedulingPolicy = 'priority',
   nodeName = 'agi-node-0'
}

print("Initializing OpenCog Inferno OS with configuration:")
print("  - Max Processes: " .. config.maxProcesses)
print("  - Cognitive Resource Size: " .. config.cognitiveResourceSize)
print("  - Thought Vector Size: " .. config.thoughtVectorSize)
print("  - Scheduling Policy: " .. config.schedulingPolicy)
print()

-- Create the AGI Operating System
local agiOS = nn.OpenCogInfernoOS(config)

print()
print("=" .. string.rep("=", 60))
print("Example 1: System Call Interface")
print("=" .. string.rep("=", 60))
print()

-- Example syscalls
print("Making system calls to the AGI kernel...")

-- THINK syscall: Generate thought from perception
local perception = torch.randn(4, 32)
local thought = agiOS:syscall('THINK', {input = perception})
print("✓ THINK syscall: Generated thoughts of size " .. thought:size(1) .. "x" .. thought:size(2))

-- REASON syscall: Apply reasoning
local premises = torch.randn(3, 64)
local conclusion = agiOS:syscall('REASON', {premises = premises, rule = 'deduction'})
print("✓ REASON syscall: Applied deductive reasoning")

-- SPAWN syscall: Create cognitive process
local newPID = agiOS:syscall('SPAWN', {
   name = 'test-process',
   priority = 60,
   entryPoint = function(input) return input end
})
print("✓ SPAWN syscall: Created process with PID " .. newPID)

-- INTROSPECT syscall: Examine kernel state
local kernelState = agiOS:syscall('INTROSPECT', {query = 'all'})
print("✓ INTROSPECT syscall: Kernel has " .. kernelState.processes.processCount .. " processes")

print()
print("=" .. string.rep("=", 60))
print("Example 2: Cognitive Process Scheduling")
print("=" .. string.rep("=", 60))
print()

print("Creating multiple cognitive processes...")

-- Spawn several cognitive processes with different priorities
local pids = {}
local priorities = {100, 75, 50, 25}
local names = {'critical-thought', 'important-reasoning', 'normal-learning', 'background-consolidation'}

for i = 1, 4 do
   local pid = agiOS:syscall('SPAWN', {
      name = names[i],
      priority = priorities[i],
      entryPoint = function(input)
         return torch.randn(input:size())
      end,
      resources = {memory = 64}
   })
   table.insert(pids, pid)
   print("  Created process '" .. names[i] .. "' (PID " .. pid .. ", Priority " .. priorities[i] .. ")")
end

local schedStats = agiOS.scheduler:getStats()
print("\nScheduler status:")
print("  Total processes: " .. schedStats.totalProcesses)
print("  Ready processes: " .. schedStats.readyProcesses)
print("  Context switches: " .. schedStats.contextSwitches)

print()
print("=" .. string.rep("=", 60))
print("Example 3: Hierarchical Memory Management")
print("=" .. string.rep("=", 60))
print()

print("Demonstrating hierarchical cognitive memory...")

-- Allocate memory in different hierarchies
local sensoryData = torch.randn(32)
local sensoryAddr = agiOS.memory:allocate('sensory', sensoryData, {
   importance = 0.3,
   pid = 1
})
print("✓ Allocated sensory memory at address " .. sensoryAddr)

local workingData = torch.randn(32)
local workingAddr = agiOS.memory:allocate('working', workingData, {
   importance = 0.7,
   pid = 1
})
print("✓ Allocated working memory at address " .. workingAddr)

local semanticData = torch.randn(32)
local semanticAddr = agiOS.memory:allocate('semantic', semanticData, {
   importance = 0.9,
   generality = 0.95,
   pid = 1
})
print("✓ Allocated semantic memory at address " .. semanticAddr)

-- Read from memory
local retrieved = agiOS.memory:read(semanticAddr)
print("✓ Retrieved semantic memory: " .. retrieved:size(1) .. " dimensions")

-- Memory statistics
local memStats = agiOS.memory:getStats()
print("\nMemory hierarchy utilization:")
print("  Sensory buffer: " .. string.format("%.1f%%", memStats.sensoryUtilization * 100))
print("  Working memory: " .. string.format("%.1f%%", memStats.workingUtilization * 100))
print("  Episodic memory: " .. string.format("%.1f%%", memStats.episodicUtilization * 100))
print("  Semantic memory: " .. string.format("%.1f%%", memStats.semanticUtilization * 100))
print("  Page hit rate: " .. string.format("%.1f%%", memStats.hitRate * 100))

print()
print("=" .. string.rep("=", 60))
print("Example 4: Inter-Process Message Passing")
print("=" .. string.rep("=", 60))
print()

print("Creating communication channels...")

-- Create channels for cognitive communication
local thoughtChannel = agiOS.messaging:createChannel('thought-stream', {
   bufferSize = 32,
   messageType = agiOS.messaging.messageTypes.THOUGHT
})
print("✓ Created 'thought-stream' channel (ID " .. thoughtChannel .. ")")

local memoryChannel = agiOS.messaging:createChannel('memory-sync', {
   bufferSize = 16,
   messageType = agiOS.messaging.messageTypes.MEMORY
})
print("✓ Created 'memory-sync' channel (ID " .. memoryChannel .. ")")

-- Send messages
local thoughtMessage = torch.randn(64)
local success = agiOS.messaging:send(thoughtChannel, thoughtMessage)
print("✓ Sent thought message: " .. tostring(success))

-- Receive messages
local received = agiOS.messaging:receive(thoughtChannel, false)
if received then
   print("✓ Received message from " .. received.sender)
end

-- Broadcast
local broadcastMsg = torch.randn(64)
local sent = agiOS.messaging:broadcast(broadcastMsg, agiOS.messaging.messageTypes.ATTENTION)
print("✓ Broadcast attention signal to " .. sent .. " channels")

local msgStats = agiOS.messaging:getStats()
print("\nMessage passing statistics:")
print("  Active channels: " .. msgStats.channels)
print("  Messages sent: " .. msgStats.messagesSent)
print("  Messages received: " .. msgStats.messagesReceived)

print()
print("=" .. string.rep("=", 60))
print("Example 5: Knowledge Filesystem")
print("=" .. string.rep("=", 60))
print()

print("Organizing knowledge as a filesystem...")

-- Create concept files
local catConcept = torch.randn(32)
local catInode = agiOS.filesystem:create('/concepts/cat', catConcept, {
   importance = 0.8,
   truthValue = {strength = 0.9, confidence = 0.85}
})
print("✓ Created concept '/concepts/cat' (inode " .. catInode .. ")")

local animalConcept = torch.randn(32)
local animalInode = agiOS.filesystem:create('/concepts/animal', animalConcept, {
   importance = 0.85,
   truthValue = {strength = 0.95, confidence = 0.9}
})
print("✓ Created concept '/concepts/animal' (inode " .. animalInode .. ")")

-- Create relationship
local relationData = torch.randn(32)
agiOS.filesystem:create('/relations/is-a', relationData, {
   from = 'cat',
   to = 'animal',
   strength = 0.9
})
print("✓ Created relation '/relations/is-a'")

-- Create episodic memory
local episodeData = torch.randn(32)
agiOS.filesystem:create('/memories/episodic/event-001', episodeData, {
   timestamp = os.time(),
   importance = 0.7
})
print("✓ Created episodic memory '/memories/episodic/event-001'")

-- List concepts
local concepts = agiOS.filesystem:list('/concepts')
print("\nConcepts in filesystem:")
for _, entry in ipairs(concepts) do
   print("  - " .. entry.name .. " (" .. entry.type .. ")")
end

local fsStats = agiOS.filesystem:getStats()
print("\nFilesystem statistics:")
print("  Total inodes: " .. fsStats.totalInodes)
print("  Files: " .. fsStats.files)
print("  Directories: " .. fsStats.directories)

print()
print("=" .. string.rep("=", 60))
print("Example 6: Device I/O for Perception and Action")
print("=" .. string.rep("=", 60))
print()

print("Interacting with cognitive devices...")

-- Write to perception devices
local visualInput = torch.randn(1, 128)
agiOS.devices:write('eyes', visualInput)
print("✓ Wrote to visual perception device (/dev/eyes)")

-- Read from devices
local attentionState = agiOS.devices:read('attention', 1)
print("✓ Read from attention device (/dev/attention)")

-- Device control
local controlResult = agiOS.devices:ioctl('reasoning', 'infer', {premises = {}})
print("✓ Sent control command to reasoning device")

local devStats = agiOS.devices:getStats()
print("\nDevice statistics:")
print("  Total devices: " .. devStats.totalDevices)
print("  Active devices: " .. devStats.activeDevices)
print("  I/O operations: " .. devStats.ioOperations)
print("  Interrupts: " .. devStats.interrupts)

print()
print("=" .. string.rep("=", 60))
print("Example 7: Complete Cognitive Cycle")
print("=" .. string.rep("=", 60))
print()

print("Running complete perception-action cognitive cycle...")

-- Simulate sensory input
local perceptionInput = torch.randn(8, config.perceptionSize)
print("Input perception: " .. perceptionInput:size(1) .. " samples of " .. perceptionInput:size(2) .. " dimensions")

-- Process through AGI OS (forward pass)
local actions = agiOS:forward(perceptionInput)
print("Output actions: " .. actions:size(1) .. " samples of " .. actions:size(2) .. " dimensions")

-- Simulate learning from feedback
local targetActions = torch.randn(actions:size())
local criterion = nn.MSECriterion()
local loss = criterion:forward(actions, targetActions)
print("Action loss: " .. string.format("%.6f", loss))

-- Backward pass through entire OS
local gradActions = criterion:backward(actions, targetActions)
local gradPerception = agiOS:backward(perceptionInput, gradActions)
print("Gradients computed: " .. gradPerception:size(1) .. "x" .. gradPerception:size(2))

-- Get learnable parameters
local params, gradParams = agiOS:parameters()
print("Total parameter tensors: " .. #params)
local totalParams = 0
for _, p in ipairs(params) do
   totalParams = totalParams + p:nElement()
end
print("Total learnable parameters: " .. totalParams)

print()
print("=" .. string.rep("=", 60))
print("Example 8: System Status and Monitoring")
print("=" .. string.rep("=", 60))
print()

print("Comprehensive system status:")
local status = agiOS:getSystemStatus()

print("\nOS Information:")
print("  Version: " .. status.version)
print("  Uptime: " .. status.uptime .. " seconds")
print("  Clock cycle: " .. status.clockCycle)
print("  System load: " .. string.format("%.1f%%", status.systemLoad * 100))

print("\nKernel Statistics:")
print("  Processes: " .. status.kernel.processes.processCount)
print("  Memory utilization: " .. string.format("%.1f%%", status.kernel.memory.utilizationPercent))
print("  Syscalls: " .. status.kernel.stats.syscallCount)

print("\nScheduler:")
print("  Total processes: " .. status.scheduler.totalProcesses)
print("  CPU utilization: " .. string.format("%.1f%%", status.scheduler.cpuUtilization * 100))
print("  Context switches: " .. status.scheduler.contextSwitches)

print("\nMemory:")
print("  Working memory: " .. string.format("%.1f%%", status.memory.workingUtilization * 100))
print("  Semantic memory: " .. string.format("%.1f%%", status.memory.semanticUtilization * 100))
print("  Consolidations: " .. status.memory.consolidations)

print("\nMessaging:")
print("  Active channels: " .. status.messaging.channels)
print("  Messages sent: " .. status.messaging.messagesSent)
print("  Messages received: " .. status.messaging.messagesReceived)

print("\nFilesystem:")
print("  Total inodes: " .. status.filesystem.totalInodes)
print("  Files: " .. status.filesystem.files)
print("  Directories: " .. status.filesystem.directories)

print("\nDevices:")
print("  Active: " .. status.devices.activeDevices .. "/" .. status.devices.totalDevices)
print("  I/O operations: " .. status.devices.ioOperations)

print("\nCognitive Statistics:")
print("  Total thoughts: " .. status.stats.totalThoughts)
print("  Total inferences: " .. status.stats.totalInferences)
print("  Memory consolidations: " .. status.stats.memoryConsolidations)

print()
print("=" .. string.rep("=", 60))
print("Summary")
print("=" .. string.rep("=", 60))
print()

print("OpenCog Inferno AGI OS demonstration completed!")
print()
print("This revolutionary operating system makes cognitive processing")
print("a fundamental kernel service where:")
print()
print("  ✓ Thinking, reasoning, and intelligence emerge from the OS itself")
print("  ✓ Thoughts are processes that can be scheduled and managed")
print("  ✓ Memory is hierarchical with automatic consolidation")
print("  ✓ Knowledge is organized as a cognitive filesystem")
print("  ✓ Perception and action are device I/O operations")
print("  ✓ Cognitive processes communicate via message passing")
print("  ✓ The entire system is differentiable and learnable")
print()
print("Applications:")
print("  • Autonomous robots with embedded cognition")
print("  • Distributed AGI across multiple nodes")
print("  • Real-time intelligent systems")
print("  • Cognitive operating systems for AGI research")
print("  • Hybrid symbolic-neural architectures")
print()
print("System object:")
print(agiOS)
print()

print("For more information, see README_OPENCOG.md")
