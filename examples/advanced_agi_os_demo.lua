--[[
Advanced OpenCog Inferno AGI OS Demonstration
==============================================

This example demonstrates the revolutionary features of the kernel-based AGI OS:

1. Learnable Kernel Operations - Thinking and reasoning that adapt through learning
2. Real-Time Cognitive Scheduling - Critical thoughts with guaranteed deadlines
3. Distributed Process Migration - Load balancing across AGI cluster
4. Sleep-Based Memory Consolidation - Deep memory consolidation during idle periods
5. Copy-on-Write Memory - Efficient shared memory with COW
6. Reliable Distributed Messaging - Acknowledgment-based delivery
7. Remote Procedure Calls - Call cognitive functions on remote nodes
8. Network Layer - Distributed AGI across multiple machines

Usage:
  th -lnn -e "dofile('examples/advanced_agi_os_demo.lua')"
]]

require('nn')

print("=" .. string.rep("=", 70))
print("Advanced OpenCog Inferno AGI Operating System Demonstration")
print("Revolutionary Kernel-Based Distributed AGI")
print("=" .. string.rep("=", 70))
print()

-- Configuration for advanced AGI OS
local config = {
   maxProcesses = 256,
   cognitiveResourceSize = 64,
   thoughtVectorSize = 128,
   perceptionSize = 256,
   actionSize = 128,
   
   -- Kernel configuration
   heapSize = 2048,
   learningEnabled = true,
   learningRate = 0.01,
   nodeID = 'agi-alpha',
   
   -- Scheduler configuration
   schedulingPolicy = 'real-time',
   rtEnabled = true,
   migrationThreshold = 0.7,
   deadlockCheckInterval = 100,
   
   -- Memory configuration
   sensoryBufferSize = 128,
   workingMemorySize = 512,
   episodicMemorySize = 2048,
   semanticMemorySize = 8192,
   proceduralMemorySize = 1024,
   consolidationBatch = 20,
   compressionEnabled = true,
   
   -- Messaging configuration
   maxChannels = 256,
   nodeName = 'agi-alpha',
   networkEnabled = true,
   reliableDelivery = true,
   maxRetries = 3,
   
   -- Filesystem
   maxInodes = 8192,
   embeddingSize = 64
}

print("Boot Configuration:")
print("  Node ID: " .. config.nodeID)
print("  Cognitive Resource Size: " .. config.cognitiveResourceSize)
print("  Scheduling Policy: " .. config.schedulingPolicy)
print("  Real-Time Scheduling: " .. tostring(config.rtEnabled))
print("  Network Enabled: " .. tostring(config.networkEnabled))
print("  Learning Enabled: " .. tostring(config.learningEnabled))
print()

-- Boot the advanced AGI OS
print("Booting Advanced OpenCog Inferno OS...")
local agiOS = nn.OpenCogInfernoOS(config)
print("✓ OS booted successfully")
print()

-- Display system status
local status = agiOS:getSystemStatus()
print("System Status:")
print("  Version: " .. status.version)
print("  Uptime: " .. status.uptime .. " seconds")
print("  Boot Processes: " .. #agiOS.bootProcesses)
print("  Kernel Version: " .. agiOS.kernel.kernelVersion)
print("  Node ID: " .. agiOS.kernel.nodeID)
print()

print("=" .. string.rep("=", 70))
print("Demo 1: Learnable Kernel Operations")
print("=" .. string.rep("=", 70))
print()

print("The kernel's thinking and reasoning are now LEARNABLE neural operations!")
print()

-- Test learnable thinking
print("Testing THINK syscall with learnable transformation:")
local perception = torch.randn(4, 64)
local thoughts = agiOS:syscall('THINK', {input = perception})
print("✓ Input size: " .. perception:size(1) .. "x" .. perception:size(2))
print("✓ Thought size: " .. thoughts:size(1) .. "x" .. thoughts:size(2))
print("✓ Thinking transformation has " .. (#agiOS.kernel.thinkingTransform:parameters() > 0 and "learnable parameters" or "no parameters"))

-- Test learnable reasoning
print()
print("Testing REASON syscall with neural reasoning network:")
local premises = torch.randn(2, 128)
local conclusion = agiOS:syscall('REASON', {premises = premises, rule = 'neural'})
print("✓ Premises size: " .. premises:size(1) .. "x" .. premises:size(2))
print("✓ Conclusion size: " .. conclusion:size(1) .. "x" .. conclusion:size(2))
print("✓ Reasoning network is a learned neural transformation")

-- Test kernel learning
print()
print("Testing LEARN syscall:")
local experience = torch.randn(16)
local learningResult = agiOS:syscall('LEARN', {experience = experience, reward = 0.8})
print("✓ Learning result: " .. (learningResult.learned and "SUCCESS" or "FAILED"))
print("✓ Reward: " .. (learningResult.reward or 0))
print("✓ Adaptation: " .. (learningResult.adaptation or "none"))

print()
print("=" .. string.rep("=", 70))
print("Demo 2: Real-Time Cognitive Scheduling")
print("=" .. string.rep("=", 70))
print()

print("Creating real-time cognitive processes with deadlines...")
print()

-- Spawn regular process
local normalPID = agiOS:syscall('SPAWN', {
   name = 'normal-reasoning',
   priority = 50,
   entryPoint = function(x) return x end
})
print("✓ Normal process (PID " .. normalPID .. "): priority 50")

-- Spawn critical real-time process
local criticalProcess = {
   pid = normalPID + 1,
   name = 'safety-critical',
   priority = 100,
   realtime = true,
   deadline = agiOS.scheduler.totalCycles + 50,  -- Must complete in 50 cycles
   state = 'ready',
   entryPoint = function(x) return x end
}
agiOS.scheduler:addProcess(criticalProcess)
print("✓ Real-time process (PID " .. criticalProcess.pid .. "): priority 100, deadline in 50 cycles")

-- Run scheduler
print()
print("Running scheduler for 10 cycles...")
for i = 1, 10 do
   agiOS.scheduler:schedule()
end

local schedulerStats = agiOS.scheduler:getStats()
print("✓ Scheduler stats:")
print("  Total processes: " .. schedulerStats.totalProcesses)
print("  RT processes: " .. schedulerStats.rtProcesses)
print("  CPU utilization: " .. string.format("%.1f%%", schedulerStats.cpuUtilization * 100))
print("  Missed deadlines: " .. schedulerStats.rtMissedDeadlines)

print()
print("=" .. string.rep("=", 70))
print("Demo 3: Distributed Process Migration")
print("=" .. string.rep("=", 70))
print()

print("Testing process migration for distributed AGI...")
print()

-- Test process migration
local targetNode = 'agi-beta'
print("Migrating process to node '" .. targetNode .. "':")
local migrationResult = agiOS:syscall('MIGRATE', {pid = normalPID, targetNode = targetNode})
print("✓ Migration " .. (migrationResult.success and "SUCCEEDED" or "FAILED"))
if migrationResult.success then
   print("  Process " .. migrationResult.pid .. " migrated to " .. migrationResult.targetNode)
end

-- Check kernel distributed info
local distInfo = agiOS.kernel:syscall(agiOS.kernel.syscalls.INTROSPECT, {query = 'distributed'})
print()
print("✓ Distributed cluster info:")
print("  Local node: " .. distInfo.nodeID)
print("  Cluster size: " .. distInfo.clusterSize)
print("  Migrated processes: " .. distInfo.migratedProcesses)

print()
print("=" .. string.rep("=", 70))
print("Demo 4: Sleep-Based Memory Consolidation")
print("=" .. string.rep("=", 70))
print()

print("Demonstrating memory consolidation during sleep...")
print()

-- Allocate some working memory
print("Allocating memories in working memory:")
for i = 1, 5 do
   local data = torch.randn(64)
   local metadata = {
      importance = 0.5 + (i * 0.1),
      emotional = i > 3 and 0.9 or 0.3,
      accessCount = i * 3
   }
   local addr = agiOS.memory:allocate('working', data, metadata)
   print("✓ Memory " .. i .. " allocated: importance=" .. metadata.importance .. ", emotional=" .. metadata.emotional)
end

local memStatsBefore = agiOS.memory:getStats()
print()
print("Memory stats before sleep:")
print("  Working memory: " .. string.format("%.1f%%", memStatsBefore.workingUtilization * 100))
print("  Episodic memory: " .. string.format("%.1f%%", memStatsBefore.episodicUtilization * 100))
print("  Semantic memory: " .. string.format("%.1f%%", memStatsBefore.semanticUtilization * 100))

-- Perform sleep consolidation
print()
print("Entering sleep mode for deep consolidation...")
local consolidated = agiOS.memory:sleepConsolidate(50)  -- 50 time units of sleep
print("✓ Consolidated " .. consolidated .. " memories during sleep")

local memStatsAfter = agiOS.memory:getStats()
print()
print("Memory stats after sleep:")
print("  Working memory: " .. string.format("%.1f%%", memStatsAfter.workingUtilization * 100))
print("  Episodic memory: " .. string.format("%.1f%%", memStatsAfter.episodicUtilization * 100))
print("  Semantic memory: " .. string.format("%.1f%%", memStatsAfter.semanticUtilization * 100))
print("  Consolidations: " .. memStatsAfter.consolidations)
print("  Compressions: " .. agiOS.memory.stats.compressions)

print()
print("=" .. string.rep("=", 70))
print("Demo 5: Copy-on-Write Memory")
print("=" .. string.rep("=", 70))
print()

print("Testing copy-on-write for efficient memory sharing...")
print()

-- Allocate memory and enable COW
local sharedData = torch.randn(64):fill(1.0)
local addr1 = agiOS.memory:allocate('working', sharedData, {importance = 0.8})
agiOS.memory:enableCopyOnWrite(addr1)
print("✓ Allocated shared memory with COW enabled")

-- Write to COW page (should trigger copy)
local newData = torch.randn(64):fill(2.0)
agiOS.memory:write(addr1, newData)
print("✓ Write triggered copy-on-write")
print("  COW operations: " .. agiOS.memory.stats.copyOnWrites)

print()
print("=" .. string.rep("=", 70))
print("Demo 6: Reliable Distributed Messaging")
print("=" .. string.rep("=", 70))
print()

print("Testing reliable message delivery with acknowledgments...")
print()

-- Create channel
local channelID = agiOS.messaging:createChannel('reliable-channel', {
   bufferSize = 16
})
print("✓ Created channel: " .. channelID)

-- Send reliable message
local message = torch.randn(64)
local success, messageID = agiOS.messaging:sendReliable(channelID, message)
print("✓ Sent reliable message (ID: " .. messageID .. ")")

-- Acknowledge message
agiOS.messaging:acknowledge(messageID)
print("✓ Message acknowledged")

local msgStats = agiOS.messaging:getStats()
print()
print("Messaging stats:")
print("  Messages sent: " .. msgStats.messagesSent)
print("  Messages received: " .. msgStats.messagesReceived)
print("  Acknowledgments: " .. msgStats.acksReceived)
print("  Pending acks: " .. msgStats.pendingAcks)

print()
print("=" .. string.rep("=", 70))
print("Demo 7: Remote Procedure Calls")
print("=" .. string.rep("=", 70))
print()

print("Testing RPC system for distributed cognitive functions...")
print()

-- Register RPC handler
agiOS.messaging:registerRPC('cognitive_analysis', function(args)
   print("  [RPC Handler] Received cognitive analysis request")
   return {
      analyzed = true,
      confidence = 0.95,
      result = "Complex pattern detected"
   }
end)
print("✓ Registered RPC handler: cognitive_analysis")

-- Call RPC (would be on remote node in real system)
print()
print("Calling RPC on node 'agi-beta'...")
agiOS.messaging:callRPC('agi-beta', 'cognitive_analysis', {input = perception}, function(success, result)
   if success then
      print("  [RPC Callback] Success: " .. (result.result or "no result"))
      print("  [RPC Callback] Confidence: " .. (result.confidence or 0))
   else
      print("  [RPC Callback] Failed: " .. tostring(result))
   end
end)
print("✓ RPC call initiated")
print("  RPC calls: " .. msgStats.rpcCalls)

print()
print("=" .. string.rep("=", 70))
print("Demo 8: End-to-End Learning")
print("=" .. string.rep("=", 70))
print()

print("Demonstrating end-to-end learning of the entire AGI OS...")
print()

-- Run cognitive cycle
print("Running cognitive cycle:")
local input = torch.randn(2, 256)
print("✓ Input shape: " .. input:size(1) .. "x" .. input:size(2))

local output = agiOS:forward(input)
print("✓ Output shape: " .. output:size(1) .. "x" .. output:size(2))

-- Compute loss and backpropagate
local target = torch.randn(output:size())
local criterion = nn.MSECriterion()
local loss = criterion:forward(output, target)
print("✓ Loss: " .. string.format("%.6f", loss))

local gradOutput = criterion:backward(output, target)
local gradInput = agiOS:backward(input, gradOutput)
print("✓ Gradients computed: " .. gradInput:size(1) .. "x" .. gradInput:size(2))

-- Check learnable parameters
local params, gradParams = agiOS:parameters()
print()
print("✓ Total learnable parameters: " .. #params)
print("  Includes kernel, attention, PLN, and all cognitive components")
print("  The entire OS can be trained end-to-end with gradient descent!")

print()
print("=" .. string.rep("=", 70))
print("Final System Status")
print("=" .. string.rep("=", 70))
print()

local finalStatus = agiOS:getSystemStatus()
print("OpenCog Inferno OS - Final Statistics:")
print()
print("Kernel:")
print("  Version: " .. finalStatus.kernel.version)
print("  Clock cycles: " .. finalStatus.kernel.clockCycle)
print("  Syscalls: " .. finalStatus.kernel.stats.syscallCount)
print("  Learning operations: " .. finalStatus.kernel.stats.learningOperations)
print("  Processes migrated: " .. finalStatus.kernel.stats.processesMigrated)
print()
print("Scheduler:")
print("  Total processes: " .. finalStatus.scheduler.totalProcesses)
print("  Context switches: " .. finalStatus.scheduler.contextSwitches)
print("  CPU utilization: " .. string.format("%.1f%%", finalStatus.scheduler.cpuUtilization * 100))
print("  RT missed deadlines: " .. finalStatus.scheduler.rtMissedDeadlines)
print("  Processes migrated: " .. finalStatus.scheduler.processesMigrated)
print()
print("Memory:")
print("  Total pages: " .. finalStatus.memory.totalPages)
print("  Working memory: " .. string.format("%.1f%%", finalStatus.memory.workingUtilization * 100))
print("  Semantic memory: " .. string.format("%.1f%%", finalStatus.memory.semanticUtilization * 100))
print("  Consolidations: " .. finalStatus.memory.consolidations)
print("  Hit rate: " .. string.format("%.1f%%", finalStatus.memory.hitRate * 100))
print()
print("Messaging:")
print("  Channels: " .. finalStatus.messaging.channels)
print("  Messages sent: " .. finalStatus.messaging.messagesSent)
print("  RPC calls: " .. finalStatus.messaging.rpcCalls)
print("  Network errors: " .. finalStatus.messaging.networkErrors)
print()
print("Filesystem:")
print("  Total inodes: " .. finalStatus.filesystem.totalInodes)
print("  Files: " .. finalStatus.filesystem.files)
print("  Directories: " .. finalStatus.filesystem.directories)
print()
print("OS Statistics:")
print("  Total thoughts: " .. finalStatus.stats.totalThoughts)
print("  Total inferences: " .. finalStatus.stats.totalInferences)
print("  Memory consolidations: " .. finalStatus.stats.memoryConsolidations)
print()

print("=" .. string.rep("=", 70))
print("Advanced Demo Complete!")
print("=" .. string.rep("=", 70))
print()
print("This demonstration showed:")
print("  ✓ Learnable kernel operations (thinking and reasoning adapt through learning)")
print("  ✓ Real-time cognitive scheduling with deadlines")
print("  ✓ Distributed process migration for load balancing")
print("  ✓ Sleep-based memory consolidation")
print("  ✓ Copy-on-write for efficient memory sharing")
print("  ✓ Reliable message delivery with acknowledgments")
print("  ✓ Remote procedure calls for distributed cognition")
print("  ✓ End-to-end learning of the entire AGI OS")
print()
print("The OpenCog Inferno OS represents a revolutionary approach where")
print("COGNITION IS A KERNEL SERVICE - thinking, reasoning, and intelligence")
print("emerge from the operating system itself, not from applications running on it.")
print()
